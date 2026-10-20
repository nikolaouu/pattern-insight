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
    """
    Orchestrator for the pattern backtesting pipeline.

    The engine ties together every other component:

        CandlesLoader -> MarketRegimeFilter -> PatternEvaluator
                      -> StatsCollector      -> DataFrame
                      -> Plotter / PatternVisualizer

    Data model
    ----------
    The central artifact is ``self.results_df``: a "datacube" flattened
    into a dataframe where each row corresponds to one

        (interval, regime, session_open)

    cell, and columns hold the aggregated statistics for that cell
    (counts, rates, excursions, Sharpe/Sortino, p-value, ...). The
    companion attribute ``self.all_stats`` is a *view* of the same data
    grouped for per-session reporting, rebuilt from ``results_df`` via
    ``_rebuild_all_stats_from_df``.

    Regime convention
    -----------------
    Throughout the public API, ``regime='ALL'`` means "pooled across
    regimes" (the default for ``run``). Any other value is a concrete
    label ('BULLISH', 'BEARISH', 'RANGING') and selects a filtered view.
    When ``results_df`` contains both a pooled run and per-regime runs,
    the pooled rows carry ``regime='ALL'`` and coexist with the filtered
    ones; reporting methods pick 'ALL' by default if present.

    Lifecycle
    ---------
    1. Construct with loader, evaluator, intervals, labels, sessions list.
    2. Call ``run`` (pooled) and/or ``run_by_regime`` (per regime) to
       populate ``results_df``.
    3. Optionally ``save_results`` / ``load_results`` for persistence.
    4. Call reporting methods (tables, plots) -- they rebuild
       ``all_stats`` on the fly from ``results_df``, so they work after a
       reload without re-running the evaluation.

    Timeframe awareness
    -------------------
    The engine is timeframe-agnostic. Annualization of Sharpe/Sortino
    is derived from the candle duration exposed by the loader, so the
    same code produces correct results for 1m, 5m, 1h, 4h or daily
    data without configuration. If a different convention is needed
    (e.g. daily-equivalent Sharpe for intraday data), it can be
    overridden by passing a custom ``periods_per_year`` to the loader,
    the collector, or a future constructor parameter on the engine.
    """

    def __init__(self, loader, evaluator, intervals, interval_labels, sessions_open_list):
        """
        Args:
            loader: CandlesLoader instance.
            evaluator: PatternEvaluator instance (holds the pattern
                predicate and evaluation hooks).
            intervals: List of ``(start_epoch, end_epoch)`` tuples, one
                per evaluation interval.
            interval_labels: Human-readable labels, aligned with
                ``intervals``. Their *order* defines the row order of the
                datacube and is relied upon by reporting.
            sessions_open_list: List of holding horizons (in candles) to
                evaluate for each interval.

        Annualization
        -------------
        ``self.periods_per_year`` is derived automatically from the loader's
        inferred candle duration (see ``CandlesLoader.get_periods_per_year``)
        and is threaded down to ``StatsCollector`` -> ``PerformanceMetrics``
        so that Sharpe/Sortino are scaled for the dataset's actual timeframe.
        This is preferred over the historical hardcoded ``252``, which
        silently assumed daily data.
        """
        self.loader = loader
        self.evaluator = evaluator
        self.intervals = intervals
        self.interval_labels = interval_labels
        self.sessions_open_list = sessions_open_list

        self.results_df = None
        self.all_stats = None

        self.periods_per_year = loader.get_periods_per_year()

        self.visualizer = PatternVisualizer()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save_results(self, filename="results_datacube.parquet"):
        """
        Persist ``results_df`` to disk.

        Format is chosen by extension:
            * ``.parquet`` -> columnar, compact, preserves dtypes. Preferred.
            * ``.csv``     -> human-readable, loses dtypes (lists such as
                              ``price_changes`` are stringified).
            * anything else -> pickle (preserves everything, including
                              nested lists, but is not portable across
                              Python versions).

        No-op if ``results_df`` is None.
        """
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
        """
        Load a previously saved ``results_df`` and rebuild ``all_stats``.

        Returns True on success, False if the file does not exist.
        The format is inferred from the extension (same rules as
        ``save_results``).

        The rebuild step is essential: reporting methods rely on
        ``all_stats``, not on ``results_df`` directly, so a freshly
        loaded dataframe would otherwise be invisible to them.
        """
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

    # ------------------------------------------------------------------ #
    # Interval processing (single interval, one regime)
    # ------------------------------------------------------------------ #

    def _process_interval(self, start, end, label, regime='ALL', target_trend_regime=None, plot_patterns=False):
        """
        Evaluate all sessions_open values for a single interval.

        Pipeline per interval:
            1. Slice candles with a right-hand padding of
               ``max(sessions_open_list) + evaluator.padding_right``
               candles. This ensures the *last* patterns in the interval
               have enough forward data to be fully simulated for every
               horizon.
            2. Attach ATR and regime labels.
            3. Locate pattern indices once (shared across all horizons).
            4. Prepare per-pattern metadata once (shared across all
               horizons).
            5. For each ``sessions_open``: create a fresh StatsCollector,
               stream evaluated patterns into it, and append one row to
               the output.

        The single-vs-multi split (steps 3-4 once, step 5 per horizon)
        is the main performance optimization: pattern discovery and
        metadata preparation are the expensive parts, and they do not
        depend on the horizon.

        Args:
            start, end: Epoch bounds of the interval.
            label: Interval label used as a column value in results_df.
            regime: Regime label for this run. 'ALL' for pooled, or a
                concrete label when called from ``run_by_regime``.
            target_trend_regime: If not None, passed down to the
                evaluator to filter patterns by trend regime at origin.
                Matches ``regime`` in practice.
            plot_patterns: If truthy, trigger ``Plotter.plot_pattern_results``
                for the current interval/session. Useful for visual
                inspection of a specific run; slows down the pipeline
                dramatically because it renders inline.

        Returns:
            List of row dicts (one per sessions_open) ready to be appended
            to ``results_df``.
        """
        rows = []
        max_sessions = max(self.sessions_open_list)

        # Load candles with enough right-hand padding for the longest
        # horizon plus the evaluator's own safety margin.
        candles = self.loader.get_range(
            start,
            end,
            end_session_offset=max_sessions + self.evaluator.padding_right
        )

        candles = MarketRegimeFilter.apply_regimes(candles)

        # Discover patterns once for the whole interval.
        pattern_indices = self.evaluator.find_pattern_indices(
            candles=candles,
            end_epoch=end,
            look_back=1
        )

        # Precompute per-pattern metadata once for the whole interval.
        prepared_patterns = self.evaluator.prepare_patterns_data(
            candles=candles,
            pattern_indices=pattern_indices
        )

        for sessions_open in self.sessions_open_list:

            collector = StatsCollector(self.periods_per_year)
            pattern_results = []

            # Stream evaluated patterns into the collector. Optionally
            # collect the (index, pattern, status) tuples for plotting.
            for c, pattern, status in self.evaluator.evaluate_patterns(
                    candles=candles,
                    prepared_patterns=prepared_patterns,
                    sessions_open=sessions_open,
                    target_trend_regime=target_trend_regime
            ):

                collector.update(pattern, status)
                if plot_patterns:
                    pattern_results.append((c, pattern, status))

            if plot_patterns and pattern_results:

                plotter = Plotter()
                plotter.plot_pattern_results(
                    candles=candles,
                    pattern_results=pattern_results,
                    lookback=1,
                    sessions_open=sessions_open,
                    mode=plot_patterns,
                    title=f"{plot_patterns} patterns for {label}, {sessions_open} sessions"
                )

            stats = collector.finalize()

            row = self._create_data_row(
                interval=label,
                regime=regime,
                sessions_open=sessions_open,
                stats=stats
            )

            rows.append(row)

        return rows

    # ------------------------------------------------------------------ #
    # Public run entry points
    # ------------------------------------------------------------------ #

    def run(self, show_results=True, plot_patterns=False):
        """
        Run the pooled evaluation (regime='ALL') across all intervals and
        all sessions_open values.

        Populates ``self.results_df`` and ``self.all_stats``. Optionally
        prints per-session tables and session-progress plots.

        Args:
            show_results: If True, call ``show_stats_per_session`` and
                ``show_stats_progress_in_intervals`` after the run.
            plot_patterns: Passed through to ``_process_interval``; when
                truthy, inline candlestick grids are shown for each
                interval/horizon (use sparingly).
        """
        rows = []

        for label, (start, end) in zip(self.interval_labels, self.intervals):

            interval_rows = self._process_interval(
                start=start,
                end=end,
                label=label,
                regime='ALL',
                target_trend_regime=None,
                plot_patterns=plot_patterns
            )

            rows.extend(interval_rows)

        self.results_df = pd.DataFrame(rows)
        self._rebuild_all_stats_from_df()

        if show_results:
            self.show_stats_per_session()
            self.show_stats_progress_in_intervals()

    def run_by_regime(self, target_regimes=None):
        """
        Run the evaluation separately for each target trend regime.

        This produces a *longer* results_df than ``run``: rows are added
        with ``regime=<label>`` for each interval and each horizon, rather
        than the pooled ``regime='ALL'``. Reporting methods then let you
        view per-regime breakdowns by passing the corresponding filter.

        Note: per-pattern filtering is done via ``target_trend_regime``,
        which the evaluator applies against the trend regime *at the
        pattern candle* (not at some future candle). This is the correct
        no-lookahead behavior, and it means a "BULLISH run" contains
        patterns that occurred while the market was already labeled
        BULLISH at their origin.

        Args:
            target_regimes: Iterable of regime labels to run. Defaults to
                ``['BULLISH', 'BEARISH', 'RANGING']``.
        """
        if target_regimes is None:
            target_regimes = ['BULLISH', 'BEARISH', 'RANGING']

        rows = []
        for regime in target_regimes:
            for label, (start, end) in zip(self.interval_labels, self.intervals):
                interval_rows = self._process_interval(
                    start=start,
                    end=end,
                    label=label,
                    regime=regime,
                    target_trend_regime=regime,
                    plot_patterns=False
                )
                rows.extend(interval_rows)

        self.results_df = pd.DataFrame(rows)
        self._rebuild_all_stats_from_df()

    # ------------------------------------------------------------------ #
    # Row construction
    # ------------------------------------------------------------------ #

    def _create_data_row(self, interval, regime, sessions_open, stats):
        """
        Flatten a ``StatsCollector.finalize`` dict into a datacube row.

        Two things happen here that are worth noting:

        1. Rates are *derived* (successful / total, etc.) rather than
           stored directly by the collector, keeping the collector's
           output purely additive.

        2. The ``price_changes`` series is embedded verbatim as a column
           value (a Python list). This makes the dataframe heavier, but
           preserves the raw return series so that downstream analyses
           (e.g. custom metrics, alternative plots) can be computed
           without re-running the pipeline.

        The ``total or 1`` guard prevents division by zero on empty cells;
        such cells will appear with 0.0 rates, which is the intended
        neutral value.

        The ``periods_per_year`` column is stored for self-documentation:
        after ``load_results``, the annualization factor used for
        Sharpe/Sortino is recoverable from the dataframe, so the absolute
        scale of those ratios is unambiguous.

        Column schema is stable and is relied upon by the reporting
        methods; adding a metric requires updating this method *and* any
        reporting code that reads the new column.
        """
        total = stats['total'] or 1

        return {
            'interval': interval,
            'regime': regime,
            'session_open': sessions_open,
            'periods_per_year': self.periods_per_year,
            'total': stats['total'],
            'successful': stats['successful'],
            'failed': stats['failed'],
            'invalidated': stats['invalidated'],
            # Derived rates (fractions, not percents).
            'win_rate': stats['successful'] / total,
            'fail_rate': stats['failed'] / total,
            'invalid_rate': stats['invalidated'] / total,
            # Averages and risk-adjusted metrics.
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
            # Raw series, persisted for offline analysis.
            'price_changes': stats.get('price_changes', [])
        }

    # ------------------------------------------------------------------ #
    # all_stats rebuilding
    # ------------------------------------------------------------------ #

    def _rebuild_all_stats_from_df(self, target_regime=None):
        """
        Reconstruct ``self.all_stats`` from ``self.results_df``.

        ``all_stats`` is a list of ``(session_open, stats_list)`` tuples,
        where ``stats_list[i]`` corresponds to ``interval_labels[i]``. This
        is the structure consumed by the reporting methods, and it must be
        rebuilt after any operation that replaces ``results_df`` (run,
        run_by_regime, load_results).

        Regime selection
        ----------------
        If ``target_regime`` is None, 'ALL' is preferred when present
        (the output of ``run``), otherwise the first regime in the
        dataframe is used. This gives sensible behavior whether the
        datacube was produced by ``run`` or ``run_by_regime``.

        Ordering assumption
        -------------------
        ``stats_list`` is built by iterating ``self.interval_labels`` and
        matching rows by label. This means:

            * The position of a label in ``interval_labels`` defines its
              index in ``stats_list``.
            * Reporting methods that index into ``stats_list`` (e.g.
              ``show_stats_progress_in_intervals``) rely on this
              alignment. Changing ``interval_labels`` after a run will
              break the correspondence until ``_rebuild_all_stats_from_df``
              is called again.

        Missing cells
        -------------
        Labels absent from the filtered dataframe (e.g. an interval that
        produced no patterns) simply do not contribute a row to
        ``stats_list``. Reporting methods iterate over the shorter list,
        which is safe but means empty intervals are silently skipped.
        """
        if self.results_df is None or self.results_df.empty:
            return

        available_regimes = self.results_df['regime'].unique()

        if target_regime is None:

            if 'ALL' in available_regimes:
                target_regime = 'ALL'

            else:
                target_regime = available_regimes[0]

        if target_regime not in available_regimes:
            target_regime = available_regimes[0]

        df_filtered = self.results_df[self.results_df['regime'] == target_regime]

        results_per_session = {s: [] for s in self.sessions_open_list}

        for s in self.sessions_open_list:

            session_rows = df_filtered[df_filtered['session_open'] == s]
            stats_list = []

            for label in self.interval_labels:

                match = session_rows[session_rows['interval'] == label]

                if not match.empty:
                    row = match.iloc[0].to_dict()
                    # Normalize the collector-side name to the dataframe
                    # name so reporting code can use a single key.
                    row['avg_price_change_per_successful'] = row['avg_return_succ']
                    stats_list.append(row)

            results_per_session[s] = stats_list

        self.all_stats = [(s, results_per_session[s]) for s in self.sessions_open_list]

    # ------------------------------------------------------------------ #
    # Reporting: per-session tables and histograms
    # ------------------------------------------------------------------ #

    def show_stats_per_session(self, regime=None, print_stats=True, plot_histogram=True):
        """
        For each session_open value, print a rich table of per-interval
        stats and (optionally) a grouped bar chart of outcome counts.

        Args:
            regime: If None, uses the default regime resolution from
                ``_rebuild_all_stats_from_df`` ('ALL' if present).
            print_stats: If True, render a rich table per session.
            plot_histogram: If True, render a matplotlib histogram per
                session.

        The active regime is printed in titles so that the rendered output
        is self-documenting even when several regimes coexist in
        ``results_df``.
        """
        if self.results_df is None or self.results_df.empty:
            return

        self._rebuild_all_stats_from_df(target_regime=regime)

        if self.all_stats is None:
            return

        plotter = Plotter()
        active_regime = regime or ("ALL" if "ALL" in self.results_df['regime'].values else self.results_df['regime'].iloc[0])

        for session, session_stats in self.all_stats:

            if print_stats:
                self._print_session_stats(session_stats, title=f"Sessions {session} (Regime: {active_regime})")

            if plot_histogram:
                plotter.plot_sessions_histogram(
                    session_stats,
                    title=f"Sessions {session} (Regime: {active_regime})",
                    labels=self.interval_labels
                )

    def show_stats_progress_in_intervals(self, regime=None, mode="both"):
        """
        For each interval, render a line chart of outcome counts and/or
        rates as a function of ``sessions_open``.

        This is the horizontal read of the datacube: it answers "how does
        the edge evolve with holding time, within a single interval?".

        Args:
            regime: Same resolution as ``show_stats_per_session``.
            mode: Passed through to ``Plotter.plot_sessions_analysis``:
                'absolute' (counts only), 'percentage' (rates only), or
                'both' (dual axis).
        """
        if self.results_df is None or self.results_df.empty:
            return

        self._rebuild_all_stats_from_df(target_regime=regime)

        plotter = Plotter()
        active_regime = regime or ("ALL" if "ALL" in self.results_df['regime'].values else self.results_df['regime'].iloc[0])

        for i in range(len(self.intervals)):

            interval_label = self.interval_labels[i]
            summary = {}

            # Build a per-session summary dict for this interval. Note
            # that ``stats_list[i]`` is the row for this interval at the
            # i-th position; see _rebuild_all_stats_from_df for the
            # ordering contract.
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
                title=f"Interval: {interval_label} (Regime: {active_regime}) — Performance vs sessions_open ({mode})",
                mode=mode
            )

    # ------------------------------------------------------------------ #
    # Reporting: rich table
    # ------------------------------------------------------------------ #

    def _print_session_stats(self, stats_list, title):
        """
        Render a rich table summarizing per-interval stats for one session.

        Columns (in order):
            Interval, Total, Succ, Fail, Inv, Succ%, Fail%, Inv%,
            Chg%, MFE%, MAE%, MFE(ATR), MAE(ATR), Sharpe, Sortino,
            MaxDD%, p-val

        Formatting conventions:
            * Rates and returns are shown as integers or 2-decimal percents.
            * ATR-normalized excursions are shown with an 'x' suffix.
            * Ratios are shown to 2 decimals, p-values to 4.

        Notes:
            * ``Console(force_jupyter=True)`` makes output consistent
              between notebook and terminal contexts. ``soft_wrap`` and a
              fixed width of 120 avoid mid-cell wrapping in wide tables.
            * The last row gets ``end_section=True`` to visually close
              the table. If the number of rows is 0 this is a no-op.
            * ``stats_list`` is assumed to be aligned with
              ``self.interval_labels`` (see ``_rebuild_all_stats_from_df``).
        """
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

    # ------------------------------------------------------------------ #
    # Plotly delegates
    # ------------------------------------------------------------------ #
    #
    # Thin wrappers around PatternVisualizer. They check for a populated
    # results_df and then delegate. Kept as methods (not properties) so
    # callers can pass metric/regime at call time.

    def plot_heatmap(self, metric='win_rate', regime='ALL'):
        """
        Render a heatmap of ``metric`` across (interval, session_open) for
        the given regime. See ``PatternVisualizer.plot_macro_heatmap``.
        """
        if self.results_df is None or self.results_df.empty:
            return

        fig = self.visualizer.plot_macro_heatmap(self.results_df, metric=metric, regime_filter=regime)
        fig.show()

    def plot_decay(self, metric: str = 'win_rate'):
        """
        Render a performance decay curve for ``metric`` vs session_open.
        See ``PatternVisualizer.plot_performance_decay``.
        """
        if self.results_df is None or self.results_df.empty:
            return

        fig = self.visualizer.plot_performance_decay(self.results_df, primary_metric=metric)
        fig.show()

    def plot_3d(self, metric='sharpe_ratio'):
        """
        Render a 3D surface of ``metric``. See
        ``PatternVisualizer.plot_macro_3d_surface``.
        """
        if self.results_df is None or self.results_df.empty:
            return

        fig = self.visualizer.plot_macro_3d_surface(self.results_df, metric=metric)
        fig.show()

    def plot_risk_reward(self, use_atr=True):
        """
        Render the MFE vs MAE scatter. See
        ``PatternVisualizer.plot_mfe_vs_mae_scatter``.
        """
        if self.results_df is None or self.results_df.empty:
            return

        fig = self.visualizer.plot_mfe_vs_mae_scatter(self.results_df, use_atr=use_atr)
        fig.show()

    def show_tactical_table(self, session_open, regime='ALL'):
        """
        Render the tactical summary table for a specific session_open
        (and optionally a regime). See
        ``PatternVisualizer.plot_tactical_table``.
        """
        if self.results_df is None or self.results_df.empty:
            return

        fig = self.visualizer.plot_tactical_table(self.results_df, session_open=session_open, regime=regime)
        fig.show()
