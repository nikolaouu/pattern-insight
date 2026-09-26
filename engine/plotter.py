import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc

class Plotter:

    def __init__(self, width=0.175, dpi=500, figsize=(4.7, 2.5)):

        self.width = width
        self.dpi = dpi
        self.figsize = figsize

    def plot_sessions_histogram(self, stats_list, labels, title):

        total = [s['total'] for s in stats_list]
        succ = [s['successful'] for s in stats_list]
        fail = [s['failed'] for s in stats_list]
        inv = [s['invalidated'] for s in stats_list]

        x = np.arange(len(labels))

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

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

    def plot_sessions_analysis(self, summary, title, mode="both"):

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

    def plot_patterns(self, plots, charts_per_plot, title=""):

        final_plots = []

        for df_slice in plots:

            temp_p = []

            for row in df_slice.itertuples():

                if hasattr(row, 'DATE') and hasattr(row, 'TIME'):
                    dt_val = pd.to_datetime(f"{row.DATE} {row.TIME}")
                else:
                    dt_val = pd.to_datetime(row.DATE)

                new_candle = (
                    mdates.date2num(dt_val),
                    row.OPEN,
                    row.HIGH,
                    row.LOW,
                    row.CLOSE
                )

                temp_p.append(new_candle)

            final_plots.append(temp_p)

        plt.style.use('dark_background')

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
