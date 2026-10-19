import pandas as pd


# ---------------------------------------------------------------------- #
# Single-candle primitives
# ---------------------------------------------------------------------- #
#
# These are the building blocks for more complex patterns. They operate on
# whole dataframes and return boolean Series aligned with the input index,
# following the "vectorized predicate" convention used throughout the
# project (see PatternEvaluator.is_pattern).
#
# Convention:
#   * "green" == bullish candle  (close > open)
#   * "red"   == bearish candle  (open > close)
#   * Doji (open == close) is neither green nor red. Callers that need to
#     handle dojis explicitly must do so themselves.


def is_green(df: pd.DataFrame) -> pd.Series:
    """
    Return True on candles where the close is strictly above the open.

    Strict inequality means dojis (open == close) are classified as
    *not green*. This is intentional: engulfing and similar patterns that
    rely on a directional prior candle should not fire on dojis.
    """
    return df['CLOSE'] > df['OPEN']


def is_red(df: pd.DataFrame) -> pd.Series:
    """
    Return True on candles where the open is strictly above the close.

    Mirror of ``is_green``; dojis are *not red*. See ``is_green`` for the
    rationale behind strict inequality.
    """
    return df['OPEN'] > df['CLOSE']


# ---------------------------------------------------------------------- #
# Two-candle patterns
# ---------------------------------------------------------------------- #
#
# These predicates compare the current candle (c2) with the previous one
# (c1). The first row of the series is always False because c1 is undefined
# (NaN), which is the correct behavior: a two-candle pattern cannot exist
# at the very first candle of a dataset.
#
# "Engulfing" is defined in the classical sense:
#
#   * The two candles have opposite colors.
#   * The body of c2 fully contains the body of c1:
#         min(c2_open, c2_close) < min(c1_open, c1_close)
#         max(c2_open, c2_close) > max(c1_open, c1_close)
#
#   Implemented below via the equivalent pair of comparisons
#   (c2_open < c1_close AND c2_close > c1_open), which is algebraically
#   the same for opposite-colored candles and slightly cheaper to compute.


def bullish_engulf(df: pd.DataFrame) -> pd.Series:
    """
    Bullish engulfing: a red candle followed by a green candle whose body
    fully engulfs the red candle's body.

    Conditions:
        * c1 is red  (open_1 > close_1)
        * c2 is green (close_2 > open_2)
        * c2 opens below c1's close and closes above c1's open

    Returns a boolean Series aligned with ``df``. The first row is False
    because the previous candle is undefined.
    """
    c1_open = df['OPEN'].shift(1)
    c1_close = df['CLOSE'].shift(1)

    c2_open = df['OPEN']
    c2_close = df['CLOSE']

    is_red_c1 = c1_open > c1_close
    is_green_c2 = c2_close > c2_open

    # Engulf test: c2's body strictly contains c1's body.
    engulf = is_red_c1 & is_green_c2 & (c2_open < c1_close) & (c2_close > c1_open)
    return engulf


def bearish_engulf(df: pd.DataFrame) -> pd.Series:
    """
    Bearish engulfing: a green candle followed by a red candle whose body
    fully engulfs the green candle's body.

    Conditions:
        * c1 is green (close_1 > open_1)
        * c2 is red   (open_2 > close_2)
        * c2 opens above c1's close and closes below c1's open

    Returns a boolean Series aligned with ``df``. The first row is False
    because the previous candle is undefined.
    """
    c1_open = df['OPEN'].shift(1)
    c1_close = df['CLOSE'].shift(1)

    c2_open = df['OPEN']
    c2_close = df['CLOSE']

    is_green_c1 = c1_close > c1_open
    is_red_c2 = c2_open > c2_close

    # Engulf test: c2's body strictly contains c1's body.
    engulf = is_green_c1 & is_red_c2 & (c2_open > c1_close) & (c2_close < c1_open)
    return engulf
