import pandas as pd


def is_green(df: pd.DataFrame) -> pd.Series:
    return df['CLOSE'] > df['OPEN']


def is_red(df: pd.DataFrame) -> pd.Series:
    return df['OPEN'] > df['CLOSE']


def bullish_engulf(df: pd.DataFrame) -> pd.Series:

    c1_open = df['OPEN'].shift(1)
    c1_close = df['CLOSE'].shift(1)

    c2_open = df['OPEN']
    c2_close = df['CLOSE']

    is_red_c1 = c1_open > c1_close
    is_green_c2 = c2_close > c2_open

    engulf = is_red_c1 & is_green_c2 & (c2_open < c1_close) & (c2_close > c1_open)
    return engulf


def bearish_engulf(df: pd.DataFrame) -> pd.Series:

    c1_open = df['OPEN'].shift(1)
    c1_close = df['CLOSE'].shift(1)

    c2_open = df['OPEN']
    c2_close = df['CLOSE']

    is_green_c1 = c1_close > c1_open
    is_red_c2 = c2_open > c2_close

    engulf = is_green_c1 & is_red_c2 & (c2_open > c1_close) & (c2_close < c1_open)
    return engulf