from .pattern import Pattern
from .plotter import Plotter

import numpy as np


class PatternEvaluator:

    def __init__(self, is_pattern, get_invalidation_val, is_invalidated, is_successful):

        self.is_pattern = is_pattern
        self.get_invalidation_val = get_invalidation_val
        self.is_invalidated = is_invalidated
        self.is_successful = is_successful

        self.all_patterns_detected = []
        self.successful_patterns_detected = []
        self.failed_patterns_detected = []
        self.invalidated_patterns_detected = []

        self.padding_right = 5

    def plot_patterns(self, candles, lookback, sessions_open, mode=None, charts_per_plot=8, title=""):

        if mode is None:
            mode = 'all'

        list_to_use = []
        if mode == 'all':
            list_to_use = self.all_patterns_detected
        elif mode == 'successful':
            list_to_use = self.successful_patterns_detected
        elif mode == 'failed':
            list_to_use = self.failed_patterns_detected
        elif mode == 'invalidated':
            list_to_use = self.invalidated_patterns_detected

        plotter = Plotter()

        plots = [
            candles.iloc[p - lookback : p + sessions_open + self.padding_right]
            for p in list_to_use
        ]

        actual_plots = [
            plots[i: i + charts_per_plot]
            for i in range(0, len(plots), charts_per_plot)
        ]

        for p in actual_plots:
            plotter.plot_patterns(p, charts_per_plot=charts_per_plot, title=title)

    def find_pattern_indices(self, candles, end_epoch, look_back=1):

        pattern_series = self.is_pattern(candles)
        is_pattern_mask = pattern_series.to_numpy()

        timestamps = candles['TIMESTAMP'].to_numpy()

        valid_mask = (timestamps <= end_epoch) & is_pattern_mask
        valid_mask[:look_back] = False
        indices = np.where(valid_mask)[0].tolist()

        return indices

    def prepare_patterns_data(self, candles, pattern_indices):

        indices_arr = np.array(pattern_indices, dtype=np.int64)

        closes = candles['CLOSE'].to_numpy()
        open_prices = closes[indices_arr]

        inv_prices = np.array(
            [self.get_invalidation_val(candles, c) for c in pattern_indices],
            dtype=np.float64
        )

        is_long = inv_prices < open_prices

        atrs = candles['ATR'].to_numpy()[indices_arr] if 'ATR' in candles.columns else np.ones(len(indices_arr))
        trend_regimes = candles['TREND_REGIME'].to_numpy()[indices_arr] if 'TREND_REGIME' in candles.columns else np.array(['ALL'] * len(indices_arr))
        vol_regimes = candles['VOL_REGIME'].to_numpy()[indices_arr] if 'VOL_REGIME' in candles.columns else np.array(['ALL'] * len(indices_arr))

        return {
            'indices': indices_arr,
            'open_prices': open_prices,
            'invalidation_prices': inv_prices,
            'is_long': is_long,
            'atrs': atrs,
            'trend_regimes': trend_regimes,
            'vol_regimes': vol_regimes
        }

    def evaluate_patterns(self, candles, prepared_patterns, sessions_open, target_trend_regime=None):

        indices = prepared_patterns['indices']
        open_prices = prepared_patterns['open_prices']
        inv_prices = prepared_patterns['invalidation_prices']
        is_long_arr = prepared_patterns['is_long']
        atrs = prepared_patterns['atrs']
        trend_regimes = prepared_patterns['trend_regimes']
        vol_regimes = prepared_patterns['vol_regimes']

        closes = candles['CLOSE'].to_numpy()
        highs = candles['HIGH'].to_numpy()
        lows = candles['LOW'].to_numpy()

        n_candles = len(candles)
        max_idx = n_candles - sessions_open - self.padding_right

        for c, open_price, invalidation_price, is_long, atr, trend, vol in zip(
            indices, open_prices, inv_prices, is_long_arr, atrs, trend_regimes, vol_regimes
        ):

            if c >= max_idx:
                continue

            if target_trend_regime is not None and trend != target_trend_regime:
                continue

            close_price = closes[c + sessions_open]

            pattern = Pattern(
                open_price=open_price,
                invalidation_price=invalidation_price,
                close_price=close_price,
                atr=atr,
                trend_regime=trend,
                vol_regime=vol
            )

            status = self._evaluate_pattern(
                pattern=pattern,
                candles=candles,
                idx=c,
                sessions_open=sessions_open,
                highs=highs,
                lows=lows,
                is_long=is_long
            )

            yield pattern, status

    def _evaluate_pattern(self, pattern, candles, idx, sessions_open, highs=None, lows=None, is_long=None):

        start_idx = idx + 1
        end_idx = idx + sessions_open + 1

        if highs is not None and lows is not None:

            slice_lows = lows[start_idx:end_idx]
            slice_highs = highs[start_idx:end_idx]

            if is_long is None:
                is_long = pattern.invalidation_price < pattern.open_price

            if is_long:
                max_high = np.max(slice_highs)
                min_low = np.min(slice_lows)
                pattern.mfe = (max_high - pattern.open_price) / pattern.open_price
                pattern.mae = (min_low - pattern.open_price) / pattern.open_price
                is_inv = np.any(slice_lows <= pattern.invalidation_price)

            else:
                max_high = np.max(slice_highs)
                min_low = np.min(slice_lows)
                pattern.mfe = (pattern.open_price - min_low) / pattern.open_price
                pattern.mae = (pattern.open_price - max_high) / pattern.open_price
                is_inv = np.any(slice_highs >= pattern.invalidation_price)

        else:

            window = candles.iloc[start_idx:end_idx]

            if not window.empty:

                max_high = window['HIGH'].max()
                min_low = window['LOW'].min()

                if is_long is None:
                    is_long = pattern.invalidation_price < pattern.open_price

                if is_long:
                    pattern.mfe = (max_high - pattern.open_price) / pattern.open_price
                    pattern.mae = (min_low - pattern.open_price) / pattern.open_price

                else:
                    pattern.mfe = (pattern.open_price - min_low) / pattern.open_price
                    pattern.mae = (pattern.open_price - max_high) / pattern.open_price

            is_inv = any(self.is_invalidated(pattern, row) for row in window.itertuples())

        if is_inv:
            return 'invalidated'

        if self.is_successful(pattern, pattern.price_change):
            return 'successful'

        return 'failed'
