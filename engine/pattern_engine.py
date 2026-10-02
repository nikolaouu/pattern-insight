import os
import pickle

from rich.table import Table
from rich.console import Console

from .plotter import Plotter
from .metrics import PerformanceMetrics
from .stats_collector import StatsCollector
from .regime_filter import MarketRegimeFilter


class PatternEngine:

    def __init__(self, loader, evaluator, intervals, interval_labels, sessions_open_list):

        self.loader = loader
        self.evaluator = evaluator
        self.intervals = intervals
        self.interval_labels = interval_labels
        self.sessions_open_list = sessions_open_list

        self.all_stats = None

    def save_stats(self, filename="stats.pickle"):

        with open(filename, "wb") as f:
            pickle.dump(self.all_stats, f)

    def load_stats(self, path="stats.pickle"):

        if not os.path.exists(path):
            return False

        with open(path, "rb") as f:
            self.all_stats = pickle.load(f)

        return True

    def _print_session_stats(self, stats_list, title):

        console = Console(width=180)

        table = Table(title=title, title_style="bright_white", show_header=True)

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
            label, stats = self.interval_labels[i], stats_list[i]
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
                f"{round(stats['avg_price_change_per_successful'] * 100, 2)}",
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

        total_sum = sum(s["total"] for s in stats_list)
        successful_sum = sum(s["successful"] for s in stats_list)
        failed_sum = sum(s["failed"] for s in stats_list)
        invalidated_sum = sum(s["invalidated"] for s in stats_list)

        total_sum_safe = total_sum or 1

        success_pct = int(successful_sum / total_sum_safe * 100)
        failed_pct = int(failed_sum / total_sum_safe * 100)
        invalidated_pct = int(invalidated_sum / total_sum_safe * 100)

        weighted_sum_price_change = sum(s["avg_price_change_per_successful"] * s["successful"] for s in stats_list)
        avg_success_price_change_pct = ((weighted_sum_price_change / (successful_sum or 1)) * 100)

        weighted_sum_mfe = sum(s.get("avg_mfe", 0.0) * s["total"] for s in stats_list)
        avg_mfe_pct = (weighted_sum_mfe / total_sum_safe) * 100

        weighted_sum_mae = sum(s.get("avg_mae", 0.0) * s["total"] for s in stats_list)
        avg_mae_pct = (weighted_sum_mae / total_sum_safe) * 100

        all_price_changes = []
        for s in stats_list:
            all_price_changes.extend(s.get('price_changes', []))

        overall_metrics = PerformanceMetrics.calculate_all(all_price_changes)

        table.add_row(
            "[bright_yellow]SUM[/bright_yellow]",
            str(total_sum),
            str(successful_sum),
            str(failed_sum),
            str(invalidated_sum),
            f"{success_pct}",
            f"{failed_pct}",
            f"{invalidated_pct}",
            f"{round(avg_success_price_change_pct, 2)}",
            f"{round(avg_mfe_pct, 2)}",
            f"{round(avg_mae_pct, 2)}",
            f"{"--"}",
            f"{"--"}",
            f"{round(overall_metrics['sharpe_ratio'], 2)}",
            f"{round(overall_metrics['sortino_ratio'], 2)}",
            f"{round(overall_metrics['max_drawdown'] * 100, 2)}%",
            f"{round(overall_metrics['p_value'], 4)}",
            end_section=True
        )

        console.print(table)

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

            sessions = []

            successful = []
            failed = []
            invalidated = []

            success_rate = []
            fail_rate = []
            invalid_rate = []

            for sessions_open, stats_list in self.all_stats:

                stats = stats_list[i]
                total = stats["total"] or 1

                sessions.append(sessions_open)
                successful.append(stats["successful"])
                failed.append(stats["failed"])
                invalidated.append(stats["invalidated"])

                success_rate.append(stats["successful"] / total * 100)
                fail_rate.append(stats["failed"] / total * 100)
                invalid_rate.append(stats["invalidated"] / total * 100)

            summary = {}

            for j, s in enumerate(sessions):

                entry = {}

                if mode in ("absolute", "both"):
                    entry.update({
                        "successful": successful[j],
                        "failed": failed[j],
                        "invalidated": invalidated[j],
                    })

                if mode in ("percentage", "both"):
                    entry.update({
                        "success_rate": success_rate[j],
                        "fail_rate": fail_rate[j],
                        "invalid_rate": invalid_rate[j],
                    })

                summary[s] = entry

            plotter.plot_sessions_analysis(
                summary=summary,
                title=f"Interval: {interval_label} — Performance vs sessions_open ({mode})",
                mode=mode
            )

    def _summer(self):

        summary = {}

        for sessions_open, stats_list in self.all_stats:

            total = sum(s.get("total", 0) for s in stats_list)
            successful = sum(s.get("successful", 0) for s in stats_list)
            failed = sum(s.get("failed", 0) for s in stats_list)
            invalidated = sum(s.get("invalidated", 0) for s in stats_list)

            success_rate = (successful / total * 100) if total > 0 else 0.0

            summary[sessions_open] = {
                "total": total,
                "successful": successful,
                "failed": failed,
                "invalidated": invalidated,
                "success_rate": success_rate
            }

        return summary

    def run(self, show_results=True, plot_patterns=False):

        results_per_session = {s: [] for s in self.sessions_open_list}
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

                results_per_session[sessions_open].append(collector.finalize())

        self.all_stats = [
            (sessions_open, results_per_session[sessions_open])
            for sessions_open in self.sessions_open_list
        ]

        if show_results:
            self.show_stats_per_session()
            self.show_stats_progress_in_intervals()