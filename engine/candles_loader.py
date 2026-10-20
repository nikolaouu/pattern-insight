import pandas as pd
import numpy as np


class CandlesLoader:
    """
    Loads and slices a single CSV candle file.

    Responsibilities:
        * Parse the CSV once and normalize dtypes (int timestamps,
          float OHLCV).
        * Cache the parsed dataframe so repeated ``get_range`` calls do
          not re-read the file.
        * Provide inclusive/exclusive range slicing by epoch, with an
          optional offset expressed in *sessions* (candles) rather than
          in raw seconds.
        * Infer the candle duration (and derived ``periods_per_year``)
          from the timestamp deltas, so that downstream annualization
          of Sharpe/Sortino is correct for any timeframe.

    Contract
    --------
    * The CSV must contain columns: ``TIMESTAMP``, ``OPEN``, ``HIGH``,
      ``LOW``, ``CLOSE``, ``VOLUME``. Extra columns are preserved.
    * ``TIMESTAMP`` is interpreted as Unix seconds and converted to a
      ``DATETIME`` column (added by the loader) for plotting.
    * Rows are assumed to be sorted by ``TIMESTAMP`` ascending. The
      slicing logic uses ``np.searchsorted``, which requires sorted input.
    * The returned slice is always re-indexed with ``reset_index(drop=True)``.
      This is critical: ``PatternEvaluator`` and ``Plotter`` use positional
      indexing on the returned dataframe, so a non-trivial index would
      silently misalign downstream operations.

    Mutability caveat
    -----------------
    ``load_all`` returns the *cached* dataframe directly, not a copy. Any
    in-place mutation by a caller will corrupt the cache for subsequent
    calls. Callers should either treat the result as read-only, or take
    an explicit ``.copy()``. ``get_range`` returns a fresh slice (via
    ``iloc`` + ``reset_index``), so it is safe to mutate its output.
    """

    def __init__(self, filename):
        """
        Args:
            filename: Path to the CSV file. The file is not read at
                construction time; parsing is deferred to the first
                ``load_all`` / ``get_range`` call.
        """
        self.filename = filename
        self._cache_df = None

    # ------------------------------------------------------------------ #
    # Full load (cached)
    # ------------------------------------------------------------------ #

    def load_all(self) -> pd.DataFrame:
        """
        Parse and cache the entire CSV.

        Steps:
            1. Read CSV, stripping whitespace from column names (some
               exporters pad headers with spaces).
            2. Convert ``TIMESTAMP`` (Unix seconds) to a ``DATETIME``
               column for plotting; keep the raw int ``TIMESTAMP`` for
               numeric comparisons.
            3. Cast OHLCV columns to float. Timestamp stays int.
            4. Cache and return.

        Returns the cached dataframe on subsequent calls without re-reading
        the file. See the class docstring for the mutability caveat.
        """
        if self._cache_df is not None:
            return self._cache_df

        df = pd.read_csv(self.filename, skipinitialspace=True)

        # Normalize header whitespace: ' OPEN' -> 'OPEN', etc.
        df.columns = df.columns.str.strip()

        # Add a human-readable datetime column derived from TIMESTAMP.
        df['DATETIME'] = pd.to_datetime(df['TIMESTAMP'], unit='s')

        # Ensure TIMESTAMP is a proper int (some CSVs emit it as string
        # or float if there are missing values).
        df['TIMESTAMP'] = df['TIMESTAMP'].astype(int)

        float_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
        df[float_cols] = df[float_cols].astype(float)

        self._cache_df = df
        return df

    # ------------------------------------------------------------------ #
    # Range slicing
    # ------------------------------------------------------------------ #

    def get_range(self, start_epoch, end_epoch, start_session_offset=0, end_session_offset=0) -> pd.DataFrame:
        """
        Return rows with ``start_epoch <= TIMESTAMP <= end_epoch``, after
        applying optional session offsets to the bounds.

        Session offsets
        ---------------
        An offset of ``k`` shifts the corresponding bound by ``k`` candles.
        This is a convenience for the engine, which needs to extend the
        right edge by ``max(sessions_open_list) + padding_right`` candles
        so that the last patterns in an interval have enough forward data
        to be evaluated.

        The candle duration is inferred from the *first two rows* of the
        dataset:

            epochs_per_session = TIMESTAMP[1] - TIMESTAMP[0]

        This assumes uniform candle spacing. It is correct for continuous
        crypto data but may be wrong across gaps (weekends, trading halts,
        data gaps). For the intended use -- sizing a right-hand padding --
        a small misestimate is harmless because the padding is a safety
        margin, not a hard boundary.

        Slice semantics
        ---------------
        * ``start`` is inclusive (``side='left'``).
        * ``end`` is inclusive (``side='right'``).

        This matches the typical expectation of "give me everything in
        [start, end]" and pairs naturally with the evaluator, which uses
        ``TIMESTAMP <= end_epoch`` internally.

        Args:
            start_epoch: Lower bound (Unix seconds).
            end_epoch: Upper bound (Unix seconds).
            start_session_offset: Number of candles to shift the start
                *later* (negative values shift earlier).
            end_session_offset: Number of candles to shift the end
                *later* (negative values shift earlier).

        Returns:
            A new dataframe sliced to the requested range, with a fresh
            RangeIndex. Empty if the range does not intersect the data.
        """
        df = self.load_all()

        # Infer candle duration from the first two rows. Guarded because
        # a single-row dataset would otherwise raise IndexError.
        if len(df) > 1:

            epochs_per_session = df['TIMESTAMP'].iloc[1] - df['TIMESTAMP'].iloc[0]
            end_epoch += end_session_offset * epochs_per_session
            start_epoch += start_session_offset * epochs_per_session

        timestamps = df['TIMESTAMP'].to_numpy()

        # searchsorted: O(log n) boundary location on a sorted array.
        # side='left'  -> start is inclusive.
        # side='right' -> end is inclusive.
        idx_start = np.searchsorted(timestamps, start_epoch, side='left')
        idx_end = np.searchsorted(timestamps, end_epoch, side='right')

        # iloc is positional, so it works regardless of index gaps from
        # prior slicing. reset_index(drop=True) restores a RangeIndex,
        # which the rest of the pipeline assumes.
        return df.iloc[idx_start:idx_end].reset_index(drop=True)

    def get_epochs_per_session(self) -> int:
        """Candle duration in seconds, inferred from the first two rows."""
        df = self.load_all()
        if len(df) < 2:
            raise ValueError("Need at least 2 candles to infer duration.")
        return int(df['TIMESTAMP'].iloc[1] - df['TIMESTAMP'].iloc[0])

    def get_periods_per_year(self) -> float:
        """Number of candles per calendar year (365.25 days)."""
        seconds_per_year = 365.25 * 24 * 3600
        return seconds_per_year / self.get_epochs_per_session()
