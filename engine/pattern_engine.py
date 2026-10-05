import os
import pickle

import pandas as pd

from rich.table import Table
from rich.console import Console

from .plotter import Plotter
from .visualizer import PatternVisualizer
from .stats_collector import StatsCollector
from .regime_filter import MarketRegimeFilter


class PatternEngine:

    def __init__(self, loader, evaluator, intervals, interval_labels, sessions_open_list):

        self.loader = loader
        self.evaluator = evaluator
        self.intervals = intervals
        self.interval_labels = interval_labels
        self.sessions_open_list = sessions_open_list

        self.results_df = None
        self.all_stats = None

        self.visualizer = PatternVisualizer()

    def save_results(self, filename="results_datacube.parquet"):

        if self.results_df is None:
            return

        if filename.endswith(".parquet"):
            self.results_df.to_parquet(filename, index=False)

        elif filename.endswith(".csv"):
            self.results_df.to_csv(filename, index=False)

        else:

            with open(filename, "wb") as f:
                pickle.dump(self.results_df, f)

    def load_results(self, path="results_datacube.parquet"):

        if not os.path.exists(path):
            return False

        if path.endswith(".parquet"):
            self.results_df = pd.read_parquet(path)

        elif path.endswith(".csv"):
            self.results_df = pd.read_csv(path)

        else:

            with open(path, "rb") as f:
                self.results_df = pickle.load(f)

        self._rebuild_all_stats_from_df()

        return True

    def run(self, show_results=True, plot_patterns=False):

        rows = []
        max_sessions = max(self.sessions_open_list)

        for label, (start, end) in zip(self.interval_labels, self.intervals):

            candles = self.loader.get_range(
                start,
                end,
                end_session_offset=max_sessions + self.evaluator.padding_right
            )

            candles = MarketRegimeFilter.apply_regimes(candles)

            pattern_indices = self.evaluator.find_pattern_indices(
                candles=candles,
                end_epoch=end,
                look_back=1
            )

            prepared_patterns = self.evaluator.prepare_patterns_data(
                candles=candles,
                pattern_indices=pattern_indices
            )

            for sessions_open in self.sessions_open_list:

                collector = StatsCollector()

                for pattern, status in self.evaluator.evaluate_patterns(
                        candles=candles,
                        prepared_patterns=prepared_patterns,
                        sessions_open=sessions_open
                ):

                    collector.update(pattern, status)

                if plot_patterns:
                    self.evaluator.plot_patterns(
                        candles=candles,
                        lookback=1,
                        sessions_open=sessions_open,
                        mode=plot_patterns,
                        title=f"{plot_patterns} patterns for {label}, {sessions_open} sessions"
                    )

                stats = collector.finalize()
                row = self._create_data_row(
                    interval=label,
                    regime='ALL',
                    sessions_open=sessions_open,
                    stats=stats
                )
                rows.append(row)

        self.results_df = pd.DataFrame(rows)
        self._rebuild_all_stats_from_df()

        if show_results:
            self.show_stats_per_session()
            self.show_stats_progress_in_intervals()

    def run_by_regime(self, target_regimes=None):

        if target_regimes is None:
            target_regimes = ['BULLISH', 'BEARISH', 'RANGING']

        rows = []
        max_sessions = max(self.sessions_open_list)

        for regime in target_regimes:

            for label, (start, end) in zip(self.interval_labels, self.intervals):

                candles = self.loader.get_range(start, end, end_session_offset=max_sessions + self.evaluator.padding_right)
                candles = MarketRegimeFilter.apply_regimes(candles)

                pattern_indices = self.evaluator.find_pattern_indices(candles=candles, end_epoch=end, look_back=1)
                prepared_patterns = self.evaluator.prepare_patterns_data(candles=candles, pattern_indices=pattern_indices)

                for sessions_open in self.sessions_open_list:

                    collector = StatsCollector()

                    for pattern, status in self.evaluator.evaluate_patterns(
                            candles=candles,
                            prepared_patterns=prepared_patterns,
                            sessions_open=sessions_open,
                            target_trend_regime=regime
                    ):

                        collector.update(pattern, status)

                    stats = collector.finalize()
                    row = self._create_data_row(
                        interval=label,
                        regime=regime,
                        sessions_open=sessions_open,
                        stats=stats
                    )
                    rows.append(row)

        self.results_df = pd.DataFrame(rows)
        self._rebuild_all_stats_from_df()

    def _create_data_row(self, interval, regime, sessions_open, stats):

        total = stats['total'] or 1

        return {
            'interval': interval,
            'regime': regime,
            'session_open': sessions_open,
            'total': stats['total'],
            'successful': stats['successful'],
            'failed': stats['failed'],
            'invalidated': stats['invalidated'],
            'win_rate': stats['successful'] / total,
            'fail_rate': stats['failed'] / total,
            'invalid_rate': stats['invalidated'] / total,
            'avg_return_succ': stats.get('avg_price_change_per_successful', 0.0),
            'avg_mfe': stats.get('avg_mfe', 0.0),
            'avg_mae': stats.get('avg_mae', 0.0),
            'avg_mfe_atr': stats.get('avg_mfe_atr', 0.0),
            'avg_mae_atr': stats.get('avg_mae_atr', 0.0),
            'sharpe_ratio': stats.get('sharpe_ratio', 0.0),
            'sortino_ratio': stats.get('sortino_ratio', 0.0),
            'max_drawdown': stats.get('max_drawdown', 0.0),
            'p_value': stats.get('p_value', 1.0),
            'skewness': stats.get('skewness', 0.0),
            'kurtosis': stats.get('kurtosis', 0.0),
            'price_changes': stats.get('price_changes', [])
        }

    def _rebuild_all_stats_from_df(self):

        if self.results_df is None or self.results_df.empty:
            return

        results_per_session = {s: [] for s in self.sessions_open_list}

        regimes = self.results_df['regime'].unique()
        target_regime = regimes[0]
        df_filtered = self.results_df[self.results_df['regime'] == target_regime]

        for s in self.sessions_open_list:

            session_rows = df_filtered[df_filtered['session_open'] == s]
            stats_list = []

            for label in self.interval_labels:

                match = session_rows[session_rows['interval'] == label]

                if not match.empty:
                    row = match.iloc[0].to_dict()
                    row['avg_price_change_per_successful'] = row['avg_return_succ']
                    stats_list.append(row)

            results_per_session[s] = stats_list

        self.all_stats = [(s, results_per_session[s]) for s in self.sessions_open_list]

    def show_stats_per_session(self, print_stats=True, plot_histogram=True):

        if self.all_stats is None:
            return

        plotter = Plotter()

        for session, session_stats in self.all_stats:

            if print_stats:
                self._print_session_stats(session_stats, title=f"Sessions {session}")

            if plot_histogram:
                plotter.plot_sessions_histogram(session_stats, title=f"Sessions {session}", labels=self.interval_labels)

    def show_stats_progress_in_intervals(self, mode="both"):

        plotter = Plotter()

        for i in range(len(self.intervals)):

            interval_label = self.interval_labels[i]
            summary = {}

            for sessions_open, stats_list in self.all_stats:

                if i < len(stats_list):

                    stats = stats_list[i]
                    total = stats["total"] or 1

                    summary[sessions_open] = {
                        "successful": stats["successful"],
                        "failed": stats["failed"],
                        "invalidated": stats["invalidated"],
                        "success_rate": stats["successful"] / total * 100,
                        "fail_rate": stats["failed"] / total * 100,
                        "invalid_rate": stats["invalidated"] / total * 100,
                    }

            plotter.plot_sessions_analysis(
                summary=summary,
                title=f"Interval: {interval_label} — Performance vs sessions_open ({mode})",
                mode=mode
            )

    def _print_session_stats(self, stats_list, title):

        console = Console(force_jupyter=True, soft_wrap=True, width=120)
        table = Table(title=title, title_style="bright_white", show_header=True, width=120)

        table.add_column("Interval", justify="center")
        table.add_column("Total", justify="center")
        table.add_column("Succ", header_style="bright_green", style="bright_green", justify="center")
        table.add_column("Fail", header_style="bright_red", style="bright_red", justify="center")
        table.add_column("Inv", header_style="bright_magenta", style="bright_magenta", justify="center")
        table.add_column("Succ%", header_style="bright_green", style="bright_green", justify="center")
        table.add_column("Fail%", header_style="bright_red", style="bright_red", justify="center")
        table.add_column("Inv%", header_style="bright_magenta", style="bright_magenta", justify="center")
        table.add_column("Chg%", header_style="bright_green", style="bright_green", justify="center")
        table.add_column("MFE%", header_style="bright_cyan", style="bright_cyan", justify="center")
        table.add_column("MAE%", header_style="bright_cyan", style="bright_cyan", justify="center")
        table.add_column("MFE(ATR)", header_style="bright_green", style="bright_green", justify="center")
        table.add_column("MAE(ATR)", header_style="bright_red", style="bright_red", justify="center")
        table.add_column("Sharpe", header_style="bright_yellow", style="bright_yellow", justify="center")
        table.add_column("Sortino", header_style="bright_yellow", style="bright_yellow", justify="center")
        table.add_column("MaxDD%", header_style="bright_red", style="bright_red", justify="center")
        table.add_column("p-val", header_style="bright_cyan", style="bright_cyan", justify="center")

        n = len(stats_list)

        for i in range(n):

            label = self.interval_labels[i] if i < len(self.interval_labels) else f"Int_{i}"

            stats = stats_list[i]
            total = stats["total"] or 1

            table.add_row(
                label,
                str(stats["total"]),
                str(stats["successful"]),
                str(stats["failed"]),
                str(stats["invalidated"]),
                f"{int(stats['successful'] / total * 100)}",
                f"{int(stats['failed'] / total * 100)}",
                f"{int(stats['invalidated'] / total * 100)}",
                f"{round(stats.get('avg_return_succ', 0.0) * 100, 2)}",
                f"{round(stats.get('avg_mfe', 0.0) * 100, 2)}",
                f"{round(stats.get('avg_mae', 0.0) * 100, 2)}",
                f"{round(stats.get('avg_mfe_atr', 0.0), 2)}x",
                f"{round(stats.get('avg_mae_atr', 0.0), 2)}x",
                f"{round(stats.get('sharpe_ratio', 0.0), 2)}",
                f"{round(stats.get('sortino_ratio', 0.0), 2)}",
                f"{round(stats.get('max_drawdown', 0.0) * 100, 2)}%",
                f"{round(stats.get('p_value', 1.0), 4)}",
                end_section=i == n - 1
            )

        console.print(table)

    def plot_heatmap(self, metric='win_rate', regime='ALL'):

        if self.results_df is None or self.results_df.empty:
            return

        fig = self.visualizer.plot_macro_heatmap(self.results_df, metric=metric, regime_filter=regime)
        fig.show()

    def plot_decay(self, metric: str='win_rate'):

        if self.results_df is None or self.results_df.empty:
            return

        fig = self.visualizer.plot_performance_decay(self.results_df, primary_metric=metric)
        fig.show()

    def plot_3d(self, metric='sharpe_ratio'):

        if self.results_df is None or self.results_df.empty:
            return

        fig = self.visualizer.plot_macro_3d_surface(self.results_df, metric=metric)
        fig.show()

    def plot_risk_reward(self, use_atr=True):

        if self.results_df is None or self.results_df.empty:
            return

        fig = self.visualizer.plot_mfe_vs_mae_scatter(self.results_df, use_atr=use_atr)
        fig.show()

    def show_tactical_table(self, session_open, regime='ALL'):

        if self.results_df is None or self.results_df.empty:
            return

        fig = self.visualizer.plot_tactical_table(self.results_df, session_open=session_open, regime=regime)
        fig.show()
