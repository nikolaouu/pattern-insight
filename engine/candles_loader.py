import csv
from datetime import datetime


class CandlesLoader:

    def __init__(self, filename):

        self.filename = filename
        self._cache = None

    def load_all(self):

        if self._cache is not None:
            return self._cache

        candles = []

        with open(self.filename, 'r') as f:

            reader = csv.reader(f)
            next(reader)

            for row in reader:
                candles.append({
                    'timestamp': int(row[0].strip()),
                    'date': datetime.strptime(row[1].strip(), "%d/%m/%Y"),
                    'time': datetime.strptime(row[2].strip(), "%H:%M:%S").time(),
                    'open': float(row[3].strip()),
                    'high': float(row[4].strip()),
                    'low': float(row[5].strip()),
                    'close': float(row[6].strip()),
                    'volume': float(row[7].strip()),
                })

        self._cache = candles
        return candles

    def get_range(self, start_epoch, end_epoch, start_session_offset=0, end_session_offset=0):

        candles = self.load_all()

        if len(candles) > 2:
            epochs_per_session = candles[1]['timestamp'] - candles[0]['timestamp']
            end_epoch += end_session_offset * epochs_per_session
            start_epoch += start_session_offset * epochs_per_session

        return [c for c in candles if start_epoch <= c['timestamp'] <= end_epoch]
