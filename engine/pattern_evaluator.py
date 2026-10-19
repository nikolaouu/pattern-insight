import numpy as np

from .pattern import Pattern


class PatternEvaluator:
    """
    Evaluates a user-supplied pattern predicate over a candle series.

    The evaluator is deliberately pattern-agnostic: all pattern logic is
    injected through callables (``is_pattern``, ``get_invalidation_value``,
    ``is_invalidated``, ``is_successful``). This makes the class reusable
    across different strategies without modification.

    The evaluation pipeline is split into three stages:

        1. ``find_pattern_indices``   -> locate candidate pattern occurrences.
        2. ``prepare_patterns_data``  -> precompute per-pattern metadata
                                          (prices, direction, regimes, ATR).
        3. ``evaluate_patterns``      -> simulate each pattern forward for a
                                          given holding horizon (``sessions_open``)
                                          and yield ``(index, Pattern, status)``.

    Stages 1 and 2 are computed once per interval, while stage 3 is executed
    once per horizon. This avoids redundant work when scanning many horizons.
    """

    def __init__(self, is_pattern, get_invalidation_value, is_invalidated, is_successful):
        """
        Args:
            is_pattern: Callable[[pd.DataFrame], pd.Series[bool]].
                Returns a boolean mask marking rows where the pattern occurs.
            get_invalidation_value: Callable[[pd.DataFrame, int], float].
                Returns the invalidation (stop) price for the pattern at ``index``.
            is_invalidated: Callable[[Pattern], bool].
                Legacy hook kept for API compatibility. Actual invalidation
                detection is vectorized in ``_calculate_metrics_vectorized``.
            is_successful: Callable[[Pattern, float], bool].
                Predicate deciding whether a non-invalidated pattern counts as
                a success, given the realized ``price_change``.
        """

        self.is_pattern = is_pattern
        self.get_invalidation_val = get_invalidation_value
        self.is_invalidated = is_invalidated
        self.is_successful = is_successful

        # Number of candles reserved after the evaluation window so that
        # forward simulation always has enough future data to look at.
        self.padding_right = 5

    # ------------------------------------------------------------------ #
    # Stage 1: pattern discovery
    # ------------------------------------------------------------------ #

    def find_pattern_indices(self, candles, end_epoch, look_back=1):
        """
        Return the row indices where the pattern occurs, restricted to
        candles with ``TIMESTAMP <= end_epoch``.

        ``look_back`` trailing candles at the start are excluded because
        pattern predicates typically rely on shifted values (e.g. engulfing
        patterns compare the current candle with the previous one), so the
        first few rows would be unreliable.
        """

        pattern_series = self.is_pattern(candles)
        is_pattern_mask = pattern_series.to_numpy()

        timestamps = candles['TIMESTAMP'].to_numpy()

        valid_mask = (timestamps <= end_epoch) & is_pattern_mask
        valid_mask[:look_back] = False
        indices = np.where(valid_mask)[0].tolist()

        return indices

    # ------------------------------------------------------------------ #
    # Stage 2: per-pattern metadata (vectorized)
    # ------------------------------------------------------------------ #

    def prepare_patterns_data(self, candles, pattern_indices):
        """
        Precompute all static, per-pattern information required by later
        evaluation stages. Everything here depends only on the pattern's
        origin candle and therefore can be computed once per interval.

        Returns a dict of numpy arrays aligned by pattern occurrence:

            indices             : row indices of each pattern in ``candles``
            open_prices         : close of the pattern candle (entry reference)
            invalidation_prices : stop level for each pattern
            is_long             : True if the invalidation is below entry
            atrs                : ATR at the pattern candle (for normalized stats)
            trend_regimes       : market trend regime at the pattern candle
            vol_regimes         : market volatility regime at the pattern candle
        """

        indices_arr = np.array(pattern_indices, dtype=np.int64)

        # We use the CLOSE of the pattern candle as the reference entry price.
        closes = candles['CLOSE'].to_numpy()
        open_prices = closes[indices_arr]

        inv_prices = np.array(
            [self.get_invalidation_val(candles, c) for c in pattern_indices],
            dtype=np.float64
        )


        # Direction is inferred from invalidation placement:
        # a stop below entry implies a long, above entry implies a short.
        is_long = inv_prices < open_prices

        # Fallbacks keep the pipeline robust when the caller does not
        # attach regime/ATR columns (e.g. running on raw candles).
        atrs = candles['ATR'].to_numpy()[indices_arr] if 'ATR' in candles.columns else np.ones(len(indices_arr))
        trend_regimes = candles['TREND_REGIME'].to_numpy()[indices_arr] if 'TREND_REGIME' in candles.columns else np.array(['ALL'] * len(indices_arr))
        vol_regimes = candles['VOL_REGIME'].to_numpy()[indices_arr] if 'VOL_REGIME' in candles.columns else np.array(['ALL'] * len(indices_arr))

        return {
            'indices': indices_arr,
            'open_prices': open_prices,
            'invalidation_prices': inv_prices,
            'is_long': is_long,
            'atrs': atrs,
            'trend_regimes': trend_regimes,
            'vol_regimes': vol_regimes
        }

    # ------------------------------------------------------------------ #
    # Stage 3a: filtering patterns for a specific horizon
    # ------------------------------------------------------------------ #

    def _filter_prepared_patterns(self, prepared_patterns, n_candles, sessions_open, target_trend_regime=None):
        """
        Drop patterns that cannot be fully simulated for the given horizon.

        A pattern at index ``i`` requires ``sessions_open + padding_right``
        future candles to be evaluated. Any pattern whose evaluation window
        would exceed the available data is removed to prevent lookahead bias
        and out-of-bounds slicing.

        Optionally restricts to a target trend regime (used by ``run_by_regime``).

        Returns ``None`` when no patterns survive filtering.
        """

        indices = prepared_patterns['indices']
        if len(indices) == 0:
            return None

        max_idx = n_candles - sessions_open - self.padding_right
        mask = indices < max_idx

        if target_trend_regime is not None:
            mask &= (prepared_patterns['trend_regimes'] == target_trend_regime)

        if not np.any(mask):
            return None

        return {
            'indices': indices[mask],
            'open_prices': prepared_patterns['open_prices'][mask],
            'invalidation_prices': prepared_patterns['invalidation_prices'][mask],
            'is_long': prepared_patterns['is_long'][mask],
            'atrs': prepared_patterns['atrs'][mask],
            'trend_regimes': prepared_patterns['trend_regimes'][mask],
            'vol_regimes': prepared_patterns['vol_regimes'][mask],
        }

    # ------------------------------------------------------------------ #
    # Stage 3b: vectorized forward simulation
    # ------------------------------------------------------------------ #

    def _calculate_metrics_vectorized(self, candles, filtered_patterns, sessions_open):
        """
        Simulate every filtered pattern forward for ``sessions_open`` candles
        and compute the raw excursion / invalidation metrics in a fully
        vectorized manner.

        Definitions:
            MFE (Max Favorable Excursion): best price move in the trade's favor.
            MAE (Max Adverse Excursion):   worst price move against the trade.
            Both are expressed as fractions of the entry price.

        A pattern is considered *invalidated* when price touches or crosses
        the invalidation level at any point inside the holding window.

        All calculations are done for both long and short candidates using
        numpy broadcasting; the correct side is selected via ``np.where``.

        Returns a dict of aligned arrays:
            close_prices, mfe_arr, mae_arr, is_inv_arr, price_changes
        """
        idx_filt = filtered_patterns['indices']
        open_filt = filtered_patterns['open_prices']
        inv_filt = filtered_patterns['invalidation_prices']
        is_long_filt = filtered_patterns['is_long']

        closes = candles['CLOSE'].to_numpy()
        highs = candles['HIGH'].to_numpy()
        lows = candles['LOW'].to_numpy()

        # Build a (n_patterns, sessions_open) matrix of future row indices
        # so that we can slice all forward windows in one shot.
        window_offsets = np.arange(1, sessions_open + 1)
        window_indices = idx_filt[:, None] + window_offsets

        slice_highs = highs[window_indices]
        slice_lows = lows[window_indices]

        max_highs = np.max(slice_highs, axis=1)
        min_lows = np.min(slice_lows, axis=1)
        close_prices = closes[idx_filt + sessions_open]

        # Long side: favorable = up, adverse = down, invalidated if price
        # touches the stop below entry at any point.
        mfe_long = (max_highs - open_filt) / open_filt
        mae_long = (min_lows - open_filt) / open_filt
        inv_long = np.any(slice_lows <= inv_filt[:, None], axis=1)

        # Short side: mirrored logic.
        mfe_short = (open_filt - min_lows) / open_filt
        mae_short = (open_filt - max_highs) / open_filt
        inv_short = np.any(slice_highs >= inv_filt[:, None], axis=1)

        # Select the correct side per pattern based on direction.
        mfe_arr = np.where(is_long_filt, mfe_long, mfe_short)
        mae_arr = np.where(is_long_filt, mae_long, mae_short)
        is_inv_arr = np.where(is_long_filt, inv_long, inv_short)

        # Signed return at the end of the holding window, direction-adjusted.
        price_changes_long = (close_prices - open_filt) / open_filt
        price_changes_short = (open_filt - close_prices) / open_filt
        price_changes = np.where(is_long_filt, price_changes_long, price_changes_short)

        return {
            'close_prices': close_prices,
            'mfe_arr': mfe_arr,
            'mae_arr': mae_arr,
            'is_inv_arr': is_inv_arr,
            'price_changes': price_changes
        }

    # ------------------------------------------------------------------ #
    # Public generator
    # ------------------------------------------------------------------ #

    def evaluate_patterns(self, candles, prepared_patterns, sessions_open, target_trend_regime=None):
        """
        Yield ``(index, Pattern, status)`` for every evaluable pattern, where
        ``status`` is one of ``'invalidated'``, ``'successful'`` or ``'failed'``.

        This is a generator so that callers can stream results into a stats
        collector or a plotter without materializing the whole list.

        Classification precedence:
            1. If the invalidation level was touched -> 'invalidated'.
            2. Else if ``is_successful(pattern, price_change)`` -> 'successful'.
            3. Otherwise -> 'failed'.
        """
        filtered = self._filter_prepared_patterns(
            prepared_patterns=prepared_patterns,
            n_candles=len(candles),
            sessions_open=sessions_open,
            target_trend_regime=target_trend_regime
        )

        if filtered is None:
            return

        metrics = self._calculate_metrics_vectorized(candles, filtered, sessions_open)

        idx_filt = filtered['indices']
        open_filt = filtered['open_prices']
        inv_filt = filtered['invalidation_prices']
        is_long_filt = filtered['is_long']
        atrs_filt = filtered['atrs']
        trend_filt = filtered['trend_regimes']
        vol_filt = filtered['vol_regimes']

        close_prices = metrics['close_prices']
        mfe_arr = metrics['mfe_arr']
        mae_arr = metrics['mae_arr']
        is_inv_arr = metrics['is_inv_arr']
        price_changes = metrics['price_changes']

        # Per-pattern loop is intentionally scalar: it materializes the
        # ``Pattern`` object expected by StatsCollector/Plotter. The heavy
        # lifting (excursions, invalidation) is already vectorized above.
        for i in range(len(idx_filt)):

            c = idx_filt[i]

            pattern = Pattern(
                open_price=open_filt[i],
                invalidation_price=inv_filt[i],
                close_price=close_prices[i],
                atr=atrs_filt[i],
                trend_regime=trend_filt[i],
                vol_regime=vol_filt[i],
                is_long=bool(is_long_filt[i])
            )

            pattern.mfe = float(mfe_arr[i])
            pattern.mae = float(mae_arr[i])

            if is_inv_arr[i]:
                status = 'invalidated'

            elif self.is_successful(pattern, float(price_changes[i])):
                status = 'successful'

            else:
                status = 'failed'

            yield c, pattern, status
