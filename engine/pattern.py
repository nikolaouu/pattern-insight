class Pattern:
    """
    Immutable value object representing a single pattern occurrence together
    with the outcome of its forward simulation.

    A ``Pattern`` is the atomic unit produced by ``PatternEvaluator`` and
    consumed by ``StatsCollector`` and ``Plotter``. It deliberately contains
    no evaluation logic: all derived values are exposed as read-only
    properties so that callers cannot accidentally mutate shared state.

    Conventions
    -----------
    * ``open_price`` is the *entry reference* (in practice the CLOSE of the
      pattern candle, as produced by ``PatternEvaluator.prepare_patterns_data``).
    * ``invalidation_price`` is the stop level. Its position relative to
      ``open_price`` is what defines the trade direction.
    * ``is_long`` disambiguate direction; it is expected to be consistent
      with ``invalidation_price < open_price`` (long) or ``>`` (short).
    * ``mfe`` and ``mae`` are *unsigned* magnitudes: MFE is the best move in
      the trade's favor, MAE is the worst move against it. Both are stored
      as fractions of the entry price (e.g. 0.02 == 2%).
    * ``price_change`` is *signed*: positive means the trade ended in profit,
      regardless of direction.
    """

    def __init__(self, open_price, invalidation_price, close_price, is_long, atr=None, trend_regime=None, vol_regime=None):
        """
        Args:
            open_price: Entry reference price for the pattern.
            invalidation_price: Stop level used to detect invalidation.
            close_price: Price at the end of the holding window.
            is_long: True for long trades, False for short trades.
            atr: Average True Range at the pattern candle. Used to normalize
                excursions into ATR units. Non-positive / missing values are
                coerced to 1.0 to avoid division-by-zero in ATR-based metrics.
            trend_regime: Trend regime label at the pattern candle
                (e.g. 'BULLISH', 'BEARISH', 'RANGING', or 'ALL').
            vol_regime: Volatility regime label at the pattern candle
                (e.g. 'HIGH_VOL', 'NORMAL_VOL', 'LOW_VOL', or 'ALL').
        """

        self.open_price = open_price
        self.get_invalidation_value = invalidation_price
        self.close_price = close_price
        self.is_long = is_long

        # ATR normalization factor. Guard against zero/negative/missing ATR
        # so that downstream ``*_atr`` properties never divide by zero.
        self.atr = atr if atr and atr > 0 else 1.0

        self.trend_regime = trend_regime
        self.vol_regime = vol_regime

        # Unsigned excursion magnitudes; populated by the evaluator after
        # construction because they depend on the forward window.
        self.mfe = 0.0  # Maximum Favorable Excursion % (fraction of entry price)
        self.mae = 0.0  # Maximum Adverse Excursion % (fraction of entry price)

    # ------------------------------------------------------------------ #
    # Realized return
    # ------------------------------------------------------------------ #

    @property
    def price_change(self):
        """
        Direction-adjusted return at the end of the holding window.

        Positive values always mean a profitable outcome, irrespective of
        whether the trade was long or short. This is the value used by
        ``is_successful`` and by all return-based statistics
        (Sharpe, Sortino, p-value, ...).
        """
        raw = (self.close_price - self.open_price) / self.open_price
        return raw if self.is_long else -raw

    @property
    def price_change_atr(self):
        """
        Direction-adjusted return expressed in ATR units.

        Note: this divides by ATR directly rather than by (ATR / price),
        so the result is in *price* ATR units, not in return-ATR units.
        Kept for historical compatibility; prefer ``mfe_atr`` / ``mae_atr``
        when normalizing excursions.
        """
        return (self.close_price - self.open_price) / self.atr

    # ------------------------------------------------------------------ #
    # ATR-normalized excursions
    # ------------------------------------------------------------------ #

    @property
    def mfe_atr(self):
        """
        MFE expressed in ATR units.

        Since ``mfe`` is a fraction of ``open_price``, we first convert it
        back to a price delta (``mfe * open_price``) and then divide by ATR.
        The result is dimensionless and comparable across instruments and
        volatility regimes.
        """
        return (self.mfe * self.open_price) / self.atr

    @property
    def mae_atr(self):
        """
        MAE expressed in ATR units. See ``mfe_atr`` for the conversion logic.
        """
        return (self.mae * self.open_price) / self.atr
