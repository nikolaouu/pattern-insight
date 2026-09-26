import os
import pickle

from rich.table import Table
from rich import print

from .stats_collector import StatsCollector
from .plotter import Plotter


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

        table = Table(title=title)

        table.add_column("Interval", justify="center")
        table.add_column("Total", justify="center")
        table.add_column("Successful", style="bright_green", justify="center")
        table.add_column("Failed", style="bright_red", justify="center")
        table.add_column("Invalidated", style="bright_magenta", justify="center")
        table.add_column("Success %", style="bright_green", justify="center")
        table.add_column("Failed %", style="bright_red", justify="center")
        table.add_column("Invalidated %", style="bright_magenta", justify="center")
        table.add_column("Avg Success Price Change %", style="bright_green", justify="center")

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
                f"{round(stats['avg_price_change_per_successful'] * 100, 3)}",
                end_section = i == n-1
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

        table.add_row(
            "[bright_yellow]SUM[/bright_yellow]",
            str(total_sum),
            str(successful_sum),
            str(failed_sum),
            str(invalidated_sum),
            f"{success_pct}",
            f"{failed_pct}",
            f"{invalidated_pct}",
            f"{round(avg_success_price_change_pct, 3)}",
            end_section=True
        )

        print(table)

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

        self.all_stats = []

        for sessions_open in self.sessions_open_list:

            session_stats = []

            for label, (start, end) in zip(self.interval_labels, self.intervals):

                candles = self.loader.get_range(start, end, end_session_offset=sessions_open)

                collector = StatsCollector()

                for pattern, status in self.evaluator.detect_patterns(
                    candles=candles,
                    look_back=1,
                    sessions_open=sessions_open,
                    end_epoch=end
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

                session_stats.append(collector.finalize())

            self.all_stats.append((sessions_open, session_stats.copy()))

        if show_results:
            self.show_stats_per_session()
            self.show_stats_progress_in_intervals()
