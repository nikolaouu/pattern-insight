import pandas as pd
import numpy as np


class MarketRegimeFilter:

    @staticmethod
    def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Υπολογίζει το Average True Range (ATR)."""
        high = df['HIGH']
        low = df['LOW']
        close_prev = df['CLOSE'].shift(1)

        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window, min_periods=1).mean()
        return atr

    @classmethod
    def apply_regimes(
        cls,
        df: pd.DataFrame,
        sma_fast_window: int = 20,
        sma_slow_window: int = 50,
        atr_window: int = 14
    ) -> pd.DataFrame:
        """
        Εμπλουτίζει το DataFrame με δείκτες και κατηγορίες Market Regime.
        """
        df = df.copy()

        # 1. Υπολογισμός ATR
        df['ATR'] = cls.calculate_atr(df, window=atr_window)

        # 2. Υπολογισμός SMA για Τάση
        df['SMA_FAST'] = df['CLOSE'].rolling(window=sma_fast_window, min_periods=1).mean()
        df['SMA_SLOW'] = df['CLOSE'].rolling(window=sma_slow_window, min_periods=1).mean()

        # 3. Trend Regime Classification
        conditions_trend = [
            (df['CLOSE'] > df['SMA_FAST']) & (df['SMA_FAST'] > df['SMA_SLOW']),
            (df['CLOSE'] < df['SMA_FAST']) & (df['SMA_FAST'] < df['SMA_SLOW'])
        ]
        choices_trend = ['BULLISH', 'BEARISH']
        df['TREND_REGIME'] = np.select(conditions_trend, choices_trend, default='RANGING')

        # 4. Volatility Regime Classification (βάσει ATR / CLOSE %)
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