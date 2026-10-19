import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc


class Plotter:
    """
    Matplotlib-based plotting layer for per-pattern and per-session views.

    This class is the "micro" plotting counterpart to ``PatternVisualizer``
    (which handles "macro" plotly dashboards). It is responsible for:

        * Rendering grids of candlestick charts around individual patterns.
        * Rendering per-session bar charts (total / successful / failed /
          invalidated) across intervals.
        * Rendering session-open curves (absolute counts and/or percentages)
          for a single interval.

    Design notes
    ------------
    * All methods call ``plt.show()`` directly and return ``None``. Figures
      are not returned to the caller, so composition (e.g. saving to disk,
      embedding in a report) must be done by the caller via matplotlib's
      global state before the call returns.
    * Sizing constants (``figsize``, ``dpi``, bar ``width``) are tuned for
      dense multi-panel grids, not for presentation-quality single charts.
      Changing them changes the visual density of the entire report.
    * The dark background style is applied locally per method via
      ``plt.style.use('dark_background')``. This mutates global matplotlib
      state, so any subsequent plot inherits it until reset.
    """

    def __init__(self, width=0.175, dpi=500, figsize=(4.7, 2.5)):
        """
        Args:
            width: Candle body width in x-axis (date) units. Used by both
                ``candlestick_ohlc`` and the bar charts.
            dpi: Figure DPI. 500 is deliberately high so that the resulting
                PNGs remain readable when scaled down in a notebook.
            figsize: Figure size in inches. Small by design, since panels
                are meant to be inspected as grids, not individually.
        """
        self.width = width
        self.dpi = dpi
        self.figsize = figsize

        # Same padding convention as PatternEvaluator: reserve this many
        # candles after the pattern's evaluation window so the chart shows
        # a small amount of post-trade context.
        self.padding_right = 5

    # ------------------------------------------------------------------ #
    # Candlestick grids around individual patterns
    # ------------------------------------------------------------------ #

    def plot_pattern_results(self, candles, pattern_results, lookback, sessions_open, mode='all', charts_per_plot=8, title=""):
        """
        Render candlestick grids around each evaluated pattern.

        Args:
            candles: Full candle dataframe (must contain DATETIME, OPEN,
                HIGH, LOW, CLOSE).
            pattern_results: Iterable of ``(index, Pattern, status)`` tuples
                as produced by ``PatternEvaluator.evaluate_patterns``.
            lookback: Number of candles to show *before* the pattern.
            sessions_open: Holding horizon; controls how many candles are
                shown *after* the pattern.
            mode: ``'all'`` to plot every pattern, or a specific status
                (``'successful'``, ``'failed'``, ``'invalidated'``) to
                filter. Filtering is useful for visually diagnosing why
                trades fail.
            charts_per_plot: Number of candlestick panels per figure.
                Must be one of the layouts supported by ``plot_patterns``
                (4, 6, 8, or fallback 10).
            title: Suptitle for each figure.

        Notes:
            * When the filtered list is empty, the method returns silently
              (no empty figure is drawn).
            * Panels are chunked into figures of ``charts_per_plot``. The
              last chunk may be partially filled; unfilled axes are left
              blank.
        """
        if mode == 'all':
            list_to_use = [c for c, pattern, status in pattern_results]
        else:
            list_to_use = [c for c, pattern, status in pattern_results if status == mode]

        if not list_to_use:
            return

        # Slice a window around each pattern: [index - lookback,
        # index + sessions_open + padding_right).
        plots = [
            candles.iloc[p - lookback : p + sessions_open + self.padding_right]
            for p in list_to_use
        ]

        # Chunk windows into figures.
        actual_plots = [
            plots[i: i + charts_per_plot]
            for i in range(0, len(plots), charts_per_plot)
        ]

        for p in actual_plots:
            self.plot_patterns(p, charts_per_plot=charts_per_plot, title=title)

    # ------------------------------------------------------------------ #
    # Per-session bar chart
    # ------------------------------------------------------------------ #

    def plot_sessions_histogram(self, stats_list, labels, title):
        """
        Grouped bar chart of outcome counts per interval for a single
        session-open value.

        Args:
            stats_list: List of per-interval stats dicts (as produced by
                ``StatsCollector.finalize`` and reassembled by
                ``PatternEngine._rebuild_all_stats_from_df``). Each dict
                must contain ``total``, ``successful``, ``failed``,
                ``invalidated``.
            labels: Interval labels, aligned with ``stats_list``.
            title: Chart title.

        Notes:
            * Bars are offset by ``width`` fractions to avoid overlap.
            * The y-axis tick labels are intentionally tiny (size 3) to
              fit dense multi-interval reports. This is not suitable for
              presentation output.
        """
        total = [s['total'] for s in stats_list]
        succ = [s['successful'] for s in stats_list]
        fail = [s['failed'] for s in stats_list]
        inv = [s['invalidated'] for s in stats_list]

        x = np.arange(len(labels))

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Offsets: -1.5w, -0.5w, +0.5w, +1.5w -> four adjacent bars per group.
        ax.bar(x - 1.5 * self.width, total, self.width, label='Total', color=(0.0, 0.0, 1.0))
        ax.bar(x - 0.5 * self.width, succ,  self.width, label='Successful', color=(0.0, 1.0, 0.0))
        ax.bar(x + 0.5 * self.width, fail,  self.width, label='Failed', color=(1.0, 0.0, 0.0))
        ax.bar(x + 1.5 * self.width, inv,   self.width, label='Invalidated', color=(0.5, 0.0, 0.5))

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)

        ax.set_title(title, c=(1, 1, 1), y=1.05, fontsize=5)

        ax.tick_params(axis="y", which="major", labelsize=3)
        ax.tick_params(axis="x", which="major", labelsize=3)

        ax.legend(fontsize=3)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------ #
    # Session-open curves (counts and/or percentages)
    # ------------------------------------------------------------------ #

    def plot_sessions_analysis(self, summary, title, mode="both"):
        """
        Line chart of outcome counts and/or rates as a function of
        ``sessions_open``, for a single interval.

        Args:
            summary: Dict keyed by session-open value. Each value is a dict
                with keys ``successful``, ``failed``, ``invalidated``,
                ``success_rate``, ``fail_rate``, ``invalid_rate``.
            title: Chart title.
            mode: One of ``'absolute'`` (counts only), ``'percentage'``
                (rates only), or ``'both'`` (dual y-axis).

        Notes:
            * In ``'both'`` mode, counts are drawn on the left axis and
              percentages on a twin right axis.
            * The x-axis is treated as categorical: the ordering of
              ``summary`` keys defines the x order (Python 3.7+ dicts are
              insertion-ordered, so the caller controls the sequence).
        """
        sessions = list(summary.keys())

        successful, failed, invalidated = [], [], []
        success_rate, fail_rate, invalid_rate = [], [], []

        if mode in ("absolute", "both"):
            successful = [summary[s]["successful"] for s in sessions]
            failed = [summary[s]["failed"] for s in sessions]
            invalidated = [summary[s]["invalidated"] for s in sessions]

        if mode in ("percentage", "both"):
            success_rate = [summary[s]["success_rate"] for s in sessions]
            fail_rate = [summary[s]["fail_rate"] for s in sessions]
            invalid_rate = [summary[s]["invalid_rate"] for s in sessions]

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        if mode in ("absolute", "both"):
            ax.plot(sessions, successful, label="Successful", color=(0.0, 1.0, 0.0), linewidth=0.3, marker='o', markersize=0.5)
            ax.plot(sessions, failed, label="Failed", color=(1.0, 0.0, 0.0), linewidth=0.3, marker='o', markersize=0.5)
            ax.plot(sessions, invalidated, label="Invalidated", color=(0.8, 0.4, 1.0), linewidth=0.3, marker='o', markersize=0.5)
            ax.set_ylabel("Counts", fontsize=6)
            ax.legend(loc="upper left", fontsize=3)
        else:
            # Avoid an empty left axis label in percentage-only mode.
            ax.set_ylabel("")

        ax.set_xlabel("Sessions Open", fontsize=6)
        ax.tick_params(axis="both", labelsize=4)

        if mode in ("percentage", "both"):
            ax2 = ax.twinx()

            ax2.plot(sessions, success_rate, label="Success %", color=(1.0, 1.0, 0.0), linewidth=0.3, linestyle="--", marker='o', markersize=0.5)
            ax2.plot(sessions, fail_rate, label="Failed %", color=(0.83, 0.15, 0.15), linewidth=0.3, linestyle="--", marker='o', markersize=0.5)
            ax2.plot(sessions, invalid_rate, label="Invalid %", color=(0.52, 0.03, 0.61), linewidth=0.3, linestyle="--", marker='o', markersize=0.5)

            ax2.set_ylabel("Percent (%)", fontsize=6)
            ax2.tick_params(axis="y", labelsize=5)

            ax2.legend(loc="upper right", fontsize=3)

        ax.set_title(title, fontsize=5, pad=7)

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------ #
    # Candlestick grid renderer (internal)
    # ------------------------------------------------------------------ #

    def plot_patterns(self, plots, charts_per_plot, title=""):
        """
        Render a grid of candlestick charts, one per slice.

        Args:
            plots: List of candle dataframes, one per panel. Each slice
                must contain DATETIME, OPEN, HIGH, LOW, CLOSE.
            charts_per_plot: Number of panels; drives the grid shape:
                4 -> 2x2, 6 -> 2x3, 8 -> 2x4, otherwise -> 2x5.
            title: Suptitle for the figure.

        Notes:
            * ``mdates.date2num`` converts the pandas datetime column to
              matplotlib's float date representation required by
              ``candlestick_ohlc``.
            * Colors follow the convention: green = up, red = down.
            * If ``plots`` has fewer entries than ``charts_per_plot``, the
              remaining axes are left blank (no error).
        """
        final_plots = []

        for df_slice in plots:
            date_nums = mdates.date2num(df_slice['DATETIME'])

            # mplfinance's original_flavor expects (date, open, high, low, close)
            # tuples, not a dataframe.
            temp_p = list(zip(
                date_nums,
                df_slice['OPEN'],
                df_slice['HIGH'],
                df_slice['LOW'],
                df_slice['CLOSE']
            ))

            final_plots.append(temp_p)

        plt.style.use('dark_background')

        # Grid shape is hardcoded per supported panel count.
        if charts_per_plot == 4:
            m, n = 2, 2
        elif charts_per_plot == 6:
            m, n = 2, 3
        elif charts_per_plot == 8:
            m, n = 2, 4
        else:
            m, n = 2, 5

        fig, axes = plt.subplots(m, n, figsize=self.figsize, dpi=self.dpi)
        axes = axes.flatten()

        for i in range(len(final_plots)):
            ax = axes[i]
            candlestick_ohlc(ax, final_plots[i], width=self.width, colorup=(0, 1, 0), colordown=(1, 0, 0))

            ax.xaxis_date()

            ax.tick_params(axis="y", which="major", labelsize=4)
            ax.tick_params(axis="x", which="major", labelsize=3, rotation=35)

        fig.suptitle(title, fontsize=5)
        plt.tight_layout()
        plt.show()
