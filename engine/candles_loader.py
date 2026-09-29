import pandas as pd


class CandlesLoader:

    def __init__(self, filename):
        self.filename = filename
        self._cache_df = None

    def load_all(self) -> pd.DataFrame:

        if self._cache_df is not None:
            return self._cache_df

        df = pd.read_csv(self.filename, skipinitialspace=True)

        df.columns = df.columns.str.strip()

        df['TIMESTAMP'] = df['TIMESTAMP'].astype(int)
        df['DATE'] = pd.to_datetime(df['DATE'], format="%d/%m/%Y").dt.date
        df['TIME'] = pd.to_datetime(df['TIME'], format="%H:%M:%S").dt.time

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

        mask = (df['TIMESTAMP'] >= start_epoch) & (df['TIMESTAMP'] <= end_epoch)
        filtered_df = df[mask]

        return filtered_df.reset_index(drop=True)