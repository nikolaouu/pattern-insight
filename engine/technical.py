def is_green(candle):
    return candle['OPEN'] < candle['CLOSE']


def is_red(candle):
    return candle['OPEN'] > candle['CLOSE']


def bullish_engulf(c1, c2):
    return is_red(c1) and c1['OPEN'] < c2['CLOSE']


def bearish_engulf(c1, c2):
    return is_green(c1) and c1['OPEN'] > c2['CLOSE']
