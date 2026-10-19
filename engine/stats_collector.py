from collections import defaultdict
from .metrics import PerformanceMetrics


class StatsCollector:
    """
    Accumulates per-pattern outcomes into a flat dictionary of statistics.

    The collector is intentionally stateful and incremental: patterns are
    fed one at a time via ``update`` and the final metrics are computed once
    in ``finalize``. This keeps the hot loop in ``PatternEvaluator`` free of
    aggregation logic and allows the caller to stream results.

    Two categories of state are maintained:

    1. **Aggregates** (counts and running sums) stored in ``self.stats``.
       These are sufficient for counts, averages and per-status breakdowns.

    2. **Raw return series** stored in ``self.price_changes``.
       Required by ``PerformanceMetrics.calculate_all`` because Sharpe,
       Sortino, p-value, skewness and kurtosis operate on the full
       distribution, not on summary statistics.

    A single collector instance is expected to be used for exactly one
    (interval, sessions_open, regime) cell of the results datacube.
    """

    def __init__(self):
        """
        Initialize an empty collector.

        Note: ``self.stats`` is assigned here only to satisfy linters;
        the authoritative initialization happens in ``reset``, which is
        also used to recycle an instance between cells.
        """
        self.stats = None

        self.price_changes = []
        self.reset()

    def reset(self):
        """
        Reset all accumulated state.

        ``stats`` is a ``defaultdict(float)`` so that new keys can be added
        without an explicit ``KeyError`` guard. Explicit defaults are still
        seeded for the well-known keys to keep the schema stable and to make
        the resulting dict self-documenting when inspected.
        """
        self.stats = defaultdict(float)
        self.price_changes = []

        # Counts by outcome.
        self.stats.update({
            'total': 0,
            'successful': 0,
            'failed': 0,
            'invalidated': 0,
            # Running sums; converted to averages in ``finalize``.
            'sum_price_change_success': 0.0,
            'sum_mfe': 0.0,
            'sum_mae': 0.0,
            'sum_mfe_atr': 0.0,
            'sum_mae_atr': 0.0,
        })

    def update(self, pattern, status):
        """
        Register a single evaluated pattern.

        Args:
            pattern: A ``Pattern`` instance produced by ``PatternEvaluator``.
            status: One of ``'successful'``, ``'failed'`` or ``'invalidated'``.

        Excursion metrics (``mfe`` / ``mae`` and their ATR-normalized
        counterparts) are accumulated for *every* pattern regardless of
        outcome, so that ``avg_mfe`` / ``avg_mae`` reflect the whole
        population. In contrast, ``sum_price_change_success`` only
        aggregates realized returns of successful trades, matching the
        semantics of ``avg_return_succ`` downstream.
        """
        self.stats['total'] += 1
        self.stats['sum_mfe'] += pattern.mfe
        self.stats['sum_mae'] += pattern.mae
        self.stats['sum_mfe_atr'] += pattern.mfe_atr
        self.stats['sum_mae_atr'] += pattern.mae_atr

        # The raw (signed, direction-adjusted) return is stored for every
        # pattern, including invalidated ones, because distribution-based
        # metrics treat the full sample as the empirical return series.
        self.price_changes.append(pattern.price_change)

        if status == 'successful':
            self.stats['successful'] += 1
            self.stats['sum_price_change_success'] += pattern.price_change

        elif status == 'failed':
            self.stats['failed'] += 1

        elif status == 'invalidated':
            self.stats['invalidated'] += 1

    def finalize(self):
        """
        Compute the final statistics dictionary for the current cell.

        Performs three steps:

        1. Convert running sums into averages. Division by ``total`` is
           guarded with ``total or 1`` so that empty cells yield zeros
           instead of raising.

        2. Compute ``avg_price_change_per_successful`` from the success-only
           sum. This is *not* the mean of all returns -- it is the mean
           realized return conditioned on a successful outcome.

        3. Delegate distribution-based metrics (Sharpe, Sortino, max
           drawdown, p-value, skewness, kurtosis) to
           ``PerformanceMetrics.calculate_all``, which receives the full
           ``price_changes`` series.

        The returned dict is a plain ``dict`` (not the internal
        ``defaultdict``) and also embeds the raw ``price_changes`` list so
        that downstream consumers (e.g. ``PatternEngine._create_data_row``)
        can persist it into the results dataframe.

        Returns:
            dict: Aggregated statistics for this collector's cell.
        """
        s = dict(self.stats)
        total = s['total'] or 1

        # Excursion averages (population-wide).
        s['avg_mfe'] = s['sum_mfe'] / total
        s['avg_mae'] = s['sum_mae'] / total
        s['avg_mfe_atr'] = s['sum_mfe_atr'] / total
        s['avg_mae_atr'] = s['sum_mae_atr'] / total

        # Conditional mean return of successful trades only.
        if s['successful'] > 0:
            s['avg_price_change_per_successful'] = (s['sum_price_change_success'] / s['successful'])
        else:
            s['avg_price_change_per_successful'] = 0.0

        # Distribution-based metrics over the full return series.
        advanced_metrics = PerformanceMetrics.calculate_all(self.price_changes)
        s.update(advanced_metrics)

        # Persist the raw series so it can travel with the results dataframe.
        s['price_changes'] = self.price_changes

        return s
