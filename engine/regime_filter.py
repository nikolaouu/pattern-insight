import pandas as pd
import numpy as np


class MarketRegimeFilter:
    """
    Attaches market regime labels to a candle dataframe.

    Produces three derived columns consumed downstream by
    ``PatternEvaluator`` and ``PatternEngine``:

        * ``ATR``          : Average True Range (Wilder-smoothed via rolling mean).
        * ``TREND_REGIME`` : 'BULLISH' / 'BEARISH' / 'RANGING'.
        * ``VOL_REGIME``   : 'HIGH_VOL' / 'NORMAL_VOL' / 'LOW_VOL'.

    These labels are used in two distinct ways:

        1. As *filters*: ``PatternEngine.run_by_regime`` restricts evaluation
           to patterns whose origin candle carries a specific trend label.
        2. As *dimensions* of the results datacube: every cell aggregates
           statistics conditioned on the regime at the pattern candle.

    Causality
    ---------
    All rolling computations use only past data (``rolling`` with default
    right-aligned windows, ``shift(1)`` for the previous close). This is
    essential: a regime label must reflect the information available *at
    the pattern candle*, never future information. Any change to this class
    must preserve that invariant.

    In-sample assumption
    --------------------
    The volatility thresholds (``q_low`` / ``q_high``) are computed as
    quantiles over the *entire* input slice. When the slice is the full
    dataset, this leaks future distributional information into early
    labels. In practice this is acceptable because regime labels are used
    for *conditioning and reporting*, not for signal generation, and the
    quantile boundaries are stable across reasonable sample sizes. If the
    labels are ever promoted to a tradable signal, the quantiles must be
    computed on a trailing window instead.
    """

    # ------------------------------------------------------------------ #
    # ATR
    # ------------------------------------------------------------------ #

    @staticmethod
    def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Average True Range using the classic True Range definition:

            TR_t = max(high_t - low_t,
                       |high_t - close_{t-1}|,
                       |low_t  - close_{t-1}|)

        The TR series is smoothed with a simple rolling mean (not Wilder's
        EMA). This is a lighter approximation adequate for regime labeling
        and ATR-normalized excursions, but it is *not* the textbook Wilder
        ATR. Values on the first ``window`` rows are computed with
        ``min_periods=1`` so that no NaN is introduced at the start of the
        series (important because ``PatternEvaluator`` slices ATR by index).

        Args:
            df: Candle dataframe with HIGH, LOW, CLOSE columns.
            window: Rolling window length.

        Returns:
            pd.Series of ATR values aligned with ``df`` index.
        """
        high = df['HIGH']
        low = df['LOW']

        # Previous close; NaN on the first row, which is expected.
        close_prev = df['CLOSE'].shift(1)

        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()

        # Row-wise max of the three True Range candidates.
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=window, min_periods=1).mean()

    # ------------------------------------------------------------------ #
    # Entry point
    # ------------------------------------------------------------------ #

    @classmethod
    def apply_regimes(
        cls,
        df: pd.DataFrame,
        sma_fast_window: int = 20,
        sma_slow_window: int = 50,
        atr_window: int = 14
    ) -> pd.DataFrame:
        """
        Enrich ``df`` with ATR and regime labels.

        The input is copied, so the caller's dataframe is never mutated.
        Columns produced: ``ATR``, ``SMA_FAST``, ``SMA_SLOW``,
        ``TREND_REGIME``, ``VOL_REGIME``.

        Ordering matters: trend regimes are computed before volatility
        regimes because volatility regimes use ``ATR`` (already computed
        above) but not the trend label. Trend and volatility are treated
        as two independent dimensions; their cross-product is the regime
        space used for conditioning.

        Args:
            df: Candle dataframe with OPEN, HIGH, LOW, CLOSE.
            sma_fast_window: Fast SMA length for trend classification.
            sma_slow_window: Slow SMA length for trend classification.
            atr_window: ATR window length.

        Returns:
            A new dataframe with the additional columns.
        """
        df = df.copy()

        df['ATR'] = cls.calculate_atr(df, window=atr_window)
        df = cls._apply_trend_regimes(df, sma_fast_window, sma_slow_window)
        df = cls._apply_volatility_regimes(df)

        return df

    # ------------------------------------------------------------------ #
    # Trend dimension
    # ------------------------------------------------------------------ #

    @classmethod
    def _apply_trend_regimes(
        cls,
        df: pd.DataFrame,
        sma_fast_window: int,
        sma_slow_window: int
    ) -> pd.DataFrame:
        """
        Classify each candle as BULLISH, BEARISH or RANGING.

        Rules (strict, non-overlapping):
            * BULLISH : close > SMA_FAST and SMA_FAST > SMA_SLOW
            * BEARISH : close < SMA_FAST and SMA_FAST < SMA_SLOW
            * RANGING : everything else (including mixed alignment)

        The strictness means that transitional states (e.g. price above
        fast SMA but fast SMA still below slow SMA) are labeled RANGING,
        which is the intended conservative behavior: only fully aligned
        states receive a directional label.

        ``min_periods=1`` avoids NaNs at the start of the series; early
        candles will simply be classified on whatever partial window is
        available. This is consistent with the causality requirement: the
        label at candle t depends only on candles <= t.
        """
        df['SMA_FAST'] = df['CLOSE'].rolling(window=sma_fast_window, min_periods=1).mean()
        df['SMA_SLOW'] = df['CLOSE'].rolling(window=sma_slow_window, min_periods=1).mean()

        conditions_trend = [
            (df['CLOSE'] > df['SMA_FAST']) & (df['SMA_FAST'] > df['SMA_SLOW']),
            (df['CLOSE'] < df['SMA_FAST']) & (df['SMA_FAST'] < df['SMA_SLOW'])
        ]
        choices_trend = ['BULLISH', 'BEARISH']
        df['TREND_REGIME'] = np.select(conditions_trend, choices_trend, default='RANGING')

        return df

    # ------------------------------------------------------------------ #
    # Volatility dimension
    # ------------------------------------------------------------------ #

    @classmethod
    def _apply_volatility_regimes(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify each candle as HIGH_VOL, NORMAL_VOL or LOW_VOL based on
        the ATR-to-close ratio.

        Thresholds are the 33rd and 66th percentiles of ``ATR / CLOSE``
        over the input slice, so the three buckets are roughly equal-sized
        by construction (subject to ties and edge effects).

        Important caveat: the quantiles are computed *in-sample* (over the
        whole slice). For conditioning and reporting this is fine, but if
        volatility labels are ever used as a live signal, the quantiles
        must be replaced with trailing-window estimates to avoid lookahead.

        Boundary semantics:
            * ``>= q_high`` -> HIGH_VOL
            * ``<= q_low``  -> LOW_VOL
            * otherwise     -> NORMAL_VOL

        On extremely short inputs (fewer rows than needed for stable
        quantiles), all rows may collapse into a single bucket. That is
        acceptable for a diagnostic label but should be kept in mind when
        interpreting empty HIGH/LOW_VOL cells in the results.
        """
        # Normalize ATR by price so the classification is scale-invariant
        # across instruments and across time (avoids drift in raw ATR units).
        atr_pct = df['ATR'] / df['CLOSE']
        q_low = atr_pct.quantile(0.33)
        q_high = atr_pct.quantile(0.66)

        conditions_vol = [
            atr_pct >= q_high,
            atr_pct <= q_low
        ]
        choices_vol = ['HIGH_VOL', 'LOW_VOL']
        df['VOL_REGIME'] = np.select(conditions_vol, choices_vol, default='NORMAL_VOL')

        return df
