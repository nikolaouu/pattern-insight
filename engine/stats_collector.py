from collections import defaultdict
from .metrics import PerformanceMetrics


class StatsCollector:

    def __init__(self):

        self.stats = None

        self.price_changes = []
        self.reset()

    def reset(self):

        self.stats = defaultdict(float)
        self.price_changes = []

        self.stats.update({
            'total': 0,
            'successful': 0,
            'failed': 0,
            'invalidated': 0,
            'sum_price_change_success': 0.0,
            'sum_mfe': 0.0,
            'sum_mae': 0.0,
            'sum_mfe_atr': 0.0,
            'sum_mae_atr': 0.0,
        })

    def update(self, pattern, status):

        self.stats['total'] += 1
        self.stats['sum_mfe'] += pattern.mfe
        self.stats['sum_mae'] += pattern.mae
        self.stats['sum_mfe_atr'] += pattern.mfe_atr
        self.stats['sum_mae_atr'] += pattern.mae_atr

        self.price_changes.append(pattern.price_change)

        if status == 'successful':
            self.stats['successful'] += 1
            self.stats['sum_price_change_success'] += pattern.price_change

        elif status == 'failed':
            self.stats['failed'] += 1

        elif status == 'invalidated':
            self.stats['invalidated'] += 1

    def finalize(self):

        s = dict(self.stats)
        total = s['total'] or 1

        s['avg_mfe'] = s['sum_mfe'] / total
        s['avg_mae'] = s['sum_mae'] / total
        s['avg_mfe_atr'] = s['sum_mfe_atr'] / total
        s['avg_mae_atr'] = s['sum_mae_atr'] / total

        if s['successful'] > 0:
            s['avg_price_change_per_successful'] = (s['sum_price_change_success'] / s['successful'])
        else:
            s['avg_price_change_per_successful'] = 0.0

        advanced_metrics = PerformanceMetrics.calculate_all(self.price_changes)
        s.update(advanced_metrics)

        s['price_changes'] = self.price_changes

        return s
