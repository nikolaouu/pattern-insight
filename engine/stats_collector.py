from collections import defaultdict

class StatsCollector:

    def __init__(self):

        self.stats = None
        self.reset()

    def reset(self):
        self.stats = defaultdict(float)
        self.stats.update({
            'total': 0,
            'successful': 0,
            'failed': 0,
            'invalidated': 0,
            'sum_price_change_success': 0.0
        })

    def update(self, pattern, status):

        self.stats['total'] += 1

        if status == 'successful':
            self.stats['successful'] += 1
            self.stats['sum_price_change_success'] += pattern.price_change

        elif status == 'failed':
            self.stats['failed'] += 1

        elif status == 'invalidated':
            self.stats['invalidated'] += 1

    def finalize(self):
        """Return a clean stats dict."""

        s = dict(self.stats)

        if s['successful'] > 0:
            s['avg_price_change_per_successful'] = (s['sum_price_change_success'] / s['successful'])

        else:
            s['avg_price_change_per_successful'] = 0.0

        return s
