class Pattern:

    def __init__(self, open_price, invalidation_price, close_price, is_long, atr=None, trend_regime=None, vol_regime=None):

        self.open_price = open_price
        self.invalidation_price = invalidation_price
        self.close_price = close_price
        self.is_long = is_long

        self.atr = atr if atr and atr > 0 else 1.0

        self.trend_regime = trend_regime
        self.vol_regime = vol_regime

        self.mfe = 0.0  # Maximum Favorable Excursion %
        self.mae = 0.0  # Maximum Adverse Excursion %

    @property
    def price_change(self):
        raw = (self.close_price - self.open_price) / self.open_price
        return raw if self.is_long else -raw

    @property
    def price_change_atr(self):
        return (self.close_price - self.open_price) / self.atr

    @property
    def mfe_atr(self):
        return (self.mfe * self.open_price) / self.atr

    @property
    def mae_atr(self):
        return (self.mae * self.open_price) / self.atr
