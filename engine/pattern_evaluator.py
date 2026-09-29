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

    def evaluate_patterns(self, candles, pattern_indices, sessions_open):

        self.all_patterns_detected = pattern_indices
        self.successful_patterns_detected = []
        self.failed_patterns_detected = []
        self.invalidated_patterns_detected = []

        closes = candles['CLOSE'].to_numpy()
        highs = candles['HIGH'].to_numpy()
        lows = candles['LOW'].to_numpy()

        n_candles = len(candles)
        max_idx = n_candles - sessions_open - self.padding_right

        for c in pattern_indices:

            if c >= max_idx:
                continue

            open_price = closes[c]
            invalidation_price = self.get_invalidation_val(candles, c)
            close_price = closes[c + sessions_open]

            pattern = Pattern(
                open_price=open_price,
                invalidation_price=invalidation_price,
                close_price=close_price
            )

            status = self._evaluate_pattern(
                pattern=pattern,
                candles=candles,
                idx=c,
                sessions_open=sessions_open,
                highs=highs,
                lows=lows
            )

            if status == 'successful':
                self.successful_patterns_detected.append(c)
            elif status == 'failed':
                self.failed_patterns_detected.append(c)
            elif status == 'invalidated':
                self.invalidated_patterns_detected.append(c)

            yield pattern, status

    def _build_pattern(self, candles, index, sessions_open):
        return Pattern(
            open_price=candles.at[index, 'CLOSE'],
            invalidation_price=self.get_invalidation_val(candles, index),
            close_price=candles.at[index + sessions_open, 'CLOSE']
        )

    def _evaluate_pattern(self, pattern, candles, idx, sessions_open, highs=None, lows=None):

        start_idx = idx + 1
        end_idx = idx + sessions_open + 1

        if highs is not None and lows is not None:

            slice_lows = lows[start_idx:end_idx]
            slice_highs = highs[start_idx:end_idx]

            is_long = pattern.invalidation_price < pattern.open_price

            if is_long:
                max_high = np.max(slice_highs)
                min_low = np.min(slice_lows)
                pattern.mfe = (max_high - pattern.open_price) / pattern.open_price
                pattern.mae = (min_low - pattern.open_price) / pattern.open_price # Negative percent
                is_inv = np.any(slice_lows <= pattern.invalidation_price)
            else:
                max_high = np.max(slice_highs)
                min_low = np.min(slice_lows)
                pattern.mfe = (pattern.open_price - min_low) / pattern.open_price
                pattern.mae = (pattern.open_price - max_high) / pattern.open_price # Negative percent
                is_inv = np.any(slice_highs >= pattern.invalidation_price)

        else:

            window = candles.iloc[start_idx:end_idx]

            if not window.empty:

                max_high = window['HIGH'].max()
                min_low = window['LOW'].min()

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