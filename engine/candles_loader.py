import pandas as pd
import numpy as np


class CandlesLoader:

    def __init__(self, filename):
        self.filename = filename
        self._cache_df = None

    def load_all(self) -> pd.DataFrame:

        if self._cache_df is not None:
            return self._cache_df

        df = pd.read_csv(self.filename, skipinitialspace=True)

        df.columns = df.columns.str.strip()

        df['DATETIME'] = pd.to_datetime(df['TIMESTAMP'], unit='s')

        df['TIMESTAMP'] = df['TIMESTAMP'].astype(int)

        float_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
        df[float_cols] = df[float_cols].astype(float)

        self._cache_df = df
        return df

    def get_range(self, start_epoch, end_epoch, start_session_offset=0, end_session_offset=0) -> pd.DataFrame:

        df = self.load_all()

        if len(df) > 1:

            epochs_per_session = df['TIMESTAMP'].iloc[1] - df['TIMESTAMP'].iloc[0]
            end_epoch += end_session_offset * epochs_per_session
            start_epoch += start_session_offset * epochs_per_session

        timestamps = df['TIMESTAMP'].to_numpy()

        idx_start = np.searchsorted(timestamps, start_epoch, side='left')
        idx_end = np.searchsorted(timestamps, end_epoch, side='right')

        return df.iloc[idx_start:idx_end].reset_index(drop=True)
