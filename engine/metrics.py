import numpy as np
from scipy import stats


class PerformanceMetrics:
    """
    Stateless collection of distribution-based performance metrics.

    All methods operate on a 1-D array of *signed* returns (fractions, e.g.
    0.02 == 2%) and are direction-agnostic: the caller is expected to have
    already converted trade P&L into a consistent return series
    (positive = profit), as done by ``Pattern.price_change``.

    Design notes
    ------------
    * Every method is defensive: short samples, zero variance and NaN
      inputs return neutral values (0.0 or 1.0) instead of raising.
    * Annualization assumes daily bars (``periods_per_year=252``). This is
      a nominal convention used for cross-cell comparability; it does not
      imply the underlying data is daily. Changing this constant rescales
      Sharpe/Sortino but leaves their ranking intact.
    * These metrics are computed *per cell* of the results datacube
      (one (interval, sessions_open, regime) combination), not on a
      portfolio equity curve. As a consequence:
        - Max drawdown is over the *ordered* sequence of trades within the
          cell, which is an approximation of a real equity curve.
        - p-value is a one-sample t-test against a zero mean, i.e. a test
          of "is the average return statistically distinguishable from 0?".
    """

    # ------------------------------------------------------------------ #
    # Risk-adjusted return
    # ------------------------------------------------------------------ #

    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: float = 252.0) -> float:
        """
        Annualized Sharpe ratio: mean excess return divided by its
        standard deviation, scaled by ``sqrt(periods_per_year)``.

        Guard conditions:
            * Fewer than 2 samples -> 0.0 (variance undefined).
            * Zero or NaN standard deviation -> 0.0 (no dispersion).

        Notes:
            * ``ddof=1`` (sample std) is used to match the usual convention.
            * ``risk_free_rate`` is expressed *per period*, not annualized.
              With the default 0.0 this is a plain mean/std ratio.
        """
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - risk_free_rate
        std = np.std(excess_returns, ddof=1)

        if std == 0 or np.isnan(std):
            return 0.0

        mean_return = np.mean(excess_returns)

        return float((mean_return / std) * np.sqrt(periods_per_year))

    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: float = 252.0) -> float:
        """
        Annualized Sortino ratio: mean excess return divided by the
        *downside* deviation (root-mean-square of negative excess returns).

        Unlike Sharpe, upside volatility is not penalized. This matters for
        skewed strategies (e.g. trend-following with occasional large wins),
        which are the typical use case for pattern-based systems.

        Guard conditions:
            * Fewer than 2 samples -> 0.0.
            * No negative returns -> 0.0 (downside deviation undefined).
              This is a deliberate convention: a strategy with zero losing
              trades gets 0.0 rather than ``inf`` so it remains comparable
              across cells and safe for tabular reporting.
            * Zero or NaN downside deviation -> 0.0.
        """
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return 0.0

        # Downside deviation uses the full sample size as denominator
        # (target semi-deviation convention), not just the count of
        # negative observations.
        downside_std = np.sqrt(np.mean(downside_returns ** 2))

        if downside_std == 0 or np.isnan(downside_std):
            return 0.0

        mean_return = np.mean(excess_returns)
        return float((mean_return / downside_std) * np.sqrt(periods_per_year))

    # ------------------------------------------------------------------ #
    # Drawdown
    # ------------------------------------------------------------------ #

    @staticmethod
    def calculate_max_drawdown(returns: np.ndarray) -> float:
        """
        Maximum drawdown of the compounded equity curve built from
        ``returns``, returned as a *negative* fraction (e.g. -0.15 == -15%).

        The curve is obtained via cumulative product ``cumprod(1 + r)``,
        which assumes full reinvestment and no costs. The drawdown at each
        step is ``(equity - running_peak) / running_peak``; the minimum of
        that series is the max drawdown.

        Important caveat: the ordering of trades inside a cell is the
        ordering in which patterns appear in the candle series, not a
        strategy execution order. This metric is therefore a rough
        diagnostic of tail risk per cell, not a portfolio-level number.

        Guard: an empty array yields 0.0 (no drawdown).
        """
        if len(returns) == 0:
            return 0.0

        cumulative = np.cumprod(1.0 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak

        return float(np.min(drawdown))

    # ------------------------------------------------------------------ #
    # Significance
    # ------------------------------------------------------------------ #

    @staticmethod
    def calculate_p_value(returns: np.ndarray, popmean: float = 0.0) -> float:
        """
        Two-sided one-sample t-test p-value for H0: mean(returns) == popmean.

        Used as a sanity check on whether a cell's average return is
        distinguishable from noise. Interpreted as: low p-value -> the
        observed mean is unlikely under the null of zero edge.

        Guard conditions:
            * Fewer than 2 samples -> 1.0 (test undefined; treat as
              "no evidence").
            * Zero or NaN standard deviation -> 1.0 (t-statistic undefined,
              typically because all returns are identical).
            * NaN from scipy -> 1.0.

        Note: no multiple-testing correction is applied. When scanning many
        (interval, session, regime) cells, expect a non-trivial number of
        false positives at the 0.05 threshold. Treat the p-value as a
        ranking signal, not as a hard filter.
        """
        if len(returns) < 2:
            return 1.0

        std = np.std(returns, ddof=1)

        if std == 0 or np.isnan(std):
            return 1.0

        _, p_val = stats.ttest_1samp(returns, popmean)

        if not np.isnan(p_val):
            return float(p_val)

        return 1.0

    # ------------------------------------------------------------------ #
    # Higher moments
    # ------------------------------------------------------------------ #

    @staticmethod
    def calculate_skewness(returns: np.ndarray) -> float:
        """
        Sample skewness (Fisher-Pearson, as computed by ``scipy.stats.skew``).

        Positive skew indicates a fat right tail (few large wins, many small
        losses) -- typical of trend/breakout patterns. Negative skew
        indicates the opposite, typical of mean-reversion.

        Guard: fewer than 3 samples -> 0.0 (skewness undefined); NaN -> 0.0.
        """
        if len(returns) < 3:
            return 0.0

        val = float(stats.skew(returns))
        return val if not np.isnan(val) else 0.0

    @staticmethod
    def calculate_kurtosis(returns: np.ndarray) -> float:
        """
        Excess kurtosis (Fisher, as computed by ``scipy.stats.kurtosis``).

        A value of 0 corresponds to a normal distribution; positive values
        indicate fatter tails and a more peaked center. High kurtosis warns
        that the Sharpe ratio (which assumes Gaussian-ish tails) may
        understate tail risk.

        Guard: fewer than 4 samples -> 0.0 (kurtosis undefined); NaN -> 0.0.
        """
        if len(returns) < 4:
            return 0.0

        val = float(stats.kurtosis(returns))
        return val if not np.isnan(val) else 0.0

    # ------------------------------------------------------------------ #
    # Aggregator
    # ------------------------------------------------------------------ #

    @classmethod
    def calculate_all(cls, returns: list or np.ndarray, risk_free_rate: float = 0.0) -> dict:
        """
        Compute the full metric bundle for a single cell.

        This is the only entry point used by ``StatsCollector.finalize``.
        The return schema is fixed -- even for empty input, all keys are
        present with neutral defaults -- so that downstream dataframe
        construction (``PatternEngine._create_data_row``) never has to
        special-case missing metrics.

        Args:
            returns: Sequence of signed per-trade returns (fractions).
            risk_free_rate: Per-period risk-free rate passed to
                Sharpe/Sortino. Defaults to 0.0.

        Returns:
            dict with keys: sharpe_ratio, sortino_ratio, max_drawdown,
            p_value, skewness, kurtosis.
        """
        ret_arr = np.array(returns, dtype=np.float64)

        if len(ret_arr) == 0:
            return {
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'p_value': 1.0,
                'skewness': 0.0,
                'kurtosis': 0.0,
            }

        return {
            'sharpe_ratio': cls.calculate_sharpe_ratio(ret_arr, risk_free_rate),
            'sortino_ratio': cls.calculate_sortino_ratio(ret_arr, risk_free_rate),
            'max_drawdown': cls.calculate_max_drawdown(ret_arr),
            'p_value': cls.calculate_p_value(ret_arr),
            'skewness': cls.calculate_skewness(ret_arr),
            'kurtosis': cls.calculate_kurtosis(ret_arr),
        }
