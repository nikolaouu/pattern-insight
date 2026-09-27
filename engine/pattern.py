class Pattern:

    def __init__(self, open_price, invalidation_price, close_price):
        self.open_price = open_price
        self.invalidation_price = invalidation_price
        self.close_price = close_price

        self.mfe = 0  # Maximum Favorable Excursion %
        self.mae = 0  # Maximum Adverse Excursion %

    @property
    def price_change(self):
        return (self.close_price - self.open_price) / self.open_price
