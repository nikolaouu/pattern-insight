import numpy as np

from .pattern import Pattern
from .plotter import Plotter


class PatternEvaluator:

    def __init__(self, is_pattern, get_invalidation_val, is_invalidated, is_successful):

        self.is_pattern = is_pattern
        self.get_invalidation_val = get_invalidation_val
        self.is_invalidated = is_invalidated
        self.is_successful = is_successful

        self.padding_right = 5

    def plot_patterns(self, candles, pattern_results, lookback, sessions_open, mode='all', charts_per_plot=8, title=""):

        if mode == 'all':
            list_to_use = [c for c, pattern, status in pattern_results]

        else:
            list_to_use = [c for c, pattern, status in pattern_results if status == mode]

        if not list_to_use:
            return

        plotter = Plotter()

        plots = [
            candles.iloc[p - lookback : p + sessions_open + self.padding_right]
            for p in list_to_use
        ]

        actual_plots = [
            plots[i: i + charts_per_plot]
            for i in range(0, len(plots), charts_per_plot)
        ]

        for p in actual_plots:
            plotter.plot_patterns(p, charts_per_plot=charts_per_plot, title=title)

    def find_pattern_indices(self, candles, end_epoch, look_back=1):

        pattern_series = self.is_pattern(candles)
        is_pattern_mask = pattern_series.to_numpy()

        timestamps = candles['TIMESTAMP'].to_numpy()

        valid_mask = (timestamps <= end_epoch) & is_pattern_mask
        valid_mask[:look_back] = False
        indices = np.where(valid_mask)[0].tolist()

        return indices

    def prepare_patterns_data(self, candles, pattern_indices):

        indices_arr = np.array(pattern_indices, dtype=np.int64)

        closes = candles['CLOSE'].to_numpy()
        open_prices = closes[indices_arr]

        inv_prices = np.array(
            [self.get_invalidation_val(candles, c) for c in pattern_indices],
            dtype=np.float64
        )

        is_long = inv_prices < open_prices

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

    def _filter_prepared_patterns(self, prepared_patterns, n_candles, sessions_open, target_trend_regime=None):

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

    def _calculate_metrics_vectorized(self, candles, filtered_patterns, sessions_open):

        idx_filt = filtered_patterns['indices']
        open_filt = filtered_patterns['open_prices']
        inv_filt = filtered_patterns['invalidation_prices']
        is_long_filt = filtered_patterns['is_long']

        closes = candles['CLOSE'].to_numpy()
        highs = candles['HIGH'].to_numpy()
        lows = candles['LOW'].to_numpy()

        window_offsets = np.arange(1, sessions_open + 1)
        window_indices = idx_filt[:, None] + window_offsets

        slice_highs = highs[window_indices]
        slice_lows = lows[window_indices]

        max_highs = np.max(slice_highs, axis=1)
        min_lows = np.min(slice_lows, axis=1)
        close_prices = closes[idx_filt + sessions_open]

        mfe_long = (max_highs - open_filt) / open_filt
        mae_long = (min_lows - open_filt) / open_filt
        inv_long = np.any(slice_lows <= inv_filt[:, None], axis=1)

        mfe_short = (open_filt - min_lows) / open_filt
        mae_short = (open_filt - max_highs) / open_filt
        inv_short = np.any(slice_highs >= inv_filt[:, None], axis=1)

        mfe_arr = np.where(is_long_filt, mfe_long, mfe_short)
        mae_arr = np.where(is_long_filt, mae_long, mae_short)
        is_inv_arr = np.where(is_long_filt, inv_long, inv_short)

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

    def evaluate_patterns(self, candles, prepared_patterns, sessions_open, target_trend_regime=None):

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
