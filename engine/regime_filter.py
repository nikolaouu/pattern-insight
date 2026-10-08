import pandas as pd
import numpy as np


class MarketRegimeFilter:

    @staticmethod
    def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:

        high = df['HIGH']
        low = df['LOW']

        close_prev = df['CLOSE'].shift(1)

        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=window, min_periods=1).mean()

    @classmethod
    def apply_regimes(
        cls,
        df: pd.DataFrame,
        sma_fast_window: int = 20,
        sma_slow_window: int = 50,
        atr_window: int = 14
    ) -> pd.DataFrame:
        df = df.copy()

        df['ATR'] = cls.calculate_atr(df, window=atr_window)
        df = cls._apply_trend_regimes(df, sma_fast_window, sma_slow_window)
        df = cls._apply_volatility_regimes(df)

        return df

    @classmethod
    def _apply_trend_regimes(
        cls,
        df: pd.DataFrame,
        sma_fast_window: int,
        sma_slow_window: int
    ) -> pd.DataFrame:
        df['SMA_FAST'] = df['CLOSE'].rolling(window=sma_fast_window, min_periods=1).mean()
        df['SMA_SLOW'] = df['CLOSE'].rolling(window=sma_slow_window, min_periods=1).mean()

        conditions_trend = [
            (df['CLOSE'] > df['SMA_FAST']) & (df['SMA_FAST'] > df['SMA_SLOW']),
            (df['CLOSE'] < df['SMA_FAST']) & (df['SMA_FAST'] < df['SMA_SLOW'])
        ]
        choices_trend = ['BULLISH', 'BEARISH']
        df['TREND_REGIME'] = np.select(conditions_trend, choices_trend, default='RANGING')

        return df

    @classmethod
    def _apply_volatility_regimes(cls, df: pd.DataFrame) -> pd.DataFrame:
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
