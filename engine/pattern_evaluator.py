from .pattern import Pattern
from .plotter import Plotter


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

    def plot_patterns(self, candles, lookback, sessions_open, mode=None, charts_per_plot=8, title=""):
        # mode can have one of those values ['all', 'successful', 'failed', 'invalidated']

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
        plots = []

        a = 5

        for p in list_to_use:
            plot_candles = candles[p-lookback : p + sessions_open+a]
            plots.append(plot_candles)

        actual_plots, i = [], 0

        while i // charts_per_plot < len(plots) // charts_per_plot:
            actual_plots.append(plots[i: i + charts_per_plot])
            i += charts_per_plot

        if len(plots) % charts_per_plot != 0:
            actual_plots.append(plots[i:])

        for p in actual_plots:
            plotter.plot_patterns(p, charts_per_plot=charts_per_plot, title=title)

    def detect_patterns(self, candles, look_back, sessions_open, end_epoch):

        self.all_patterns_detected = []
        self.successful_patterns_detected = []
        self.failed_patterns_detected = []
        self.invalidated_patterns_detected = []

        a = 5

        for c in range(look_back, len(candles) - sessions_open - a):

            if candles[c]['timestamp'] > end_epoch or not self.is_pattern(candles, c):
                continue

            pattern = self._build_pattern(candles, c, sessions_open)

            self.all_patterns_detected.append(c)

            status = self._evaluate_pattern(pattern, candles, c, sessions_open)
            if status == 'successful':
                self.successful_patterns_detected.append(c)
            elif status == 'failed':
                self.failed_patterns_detected.append(c)
            elif status == 'invalidated':
                self.invalidated_patterns_detected.append(c)

            yield pattern, status

    def _build_pattern(self, candles, index, sessions_open):
        return Pattern(
            open_price=candles[index]['close'],
            invalidation_price=self.get_invalidation_val(candles, index),
            close_price=candles[index + sessions_open]['close']
        )

    def _evaluate_pattern(self, pattern, candles, idx, sessions_open):

        for i in range(idx+1, idx + sessions_open + 1):
            if self.is_invalidated(pattern, candles, i):
                return 'invalidated'

        if self.is_successful(pattern, pattern.price_change):
            return 'successful'

        return 'failed'
