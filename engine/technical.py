def is_green(candle):
    return candle['open'] < candle['close']


def is_red(candle):
    return candle['open'] > candle['close']


def bullish_engulf(c1, c2):
    return is_red(c1) and c1['open'] < c2['close']


def bearish_engulf(c1, c2):
    return is_green(c1) and c1['open'] > c2['close']
