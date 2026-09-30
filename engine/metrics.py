import numpy as np
from scipy import stats


class PerformanceMetrics:

    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: float = 252.0) -> float:

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

        if len(returns) < 2:
            return 0.0

        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return 0.0

        downside_std = np.sqrt(np.mean(downside_returns ** 2))

        if downside_std == 0 or np.isnan(downside_std):
            return 0.0

        mean_return = np.mean(excess_returns)
        return float((mean_return / downside_std) * np.sqrt(periods_per_year))

    @staticmethod
    def calculate_max_drawdown(returns: np.ndarray) -> float:

        if len(returns) == 0:
            return 0.0

        cumulative = np.cumprod(1.0 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak

        return float(np.min(drawdown))

    @staticmethod
    def calculate_p_value(returns: np.ndarray, popmean: float = 0.0) -> float:

        if len(returns) < 2:
            return 1.0

        std = np.std(returns, ddof=1)

        if std == 0 or np.isnan(std):
            return 1.0

        _, p_val = stats.ttest_1samp(returns, popmean)

        if not np.isnan(p_val):
            return float(p_val)

        return 1.0

    @staticmethod
    def calculate_skewness(returns: np.ndarray) -> float:

        if len(returns) < 3:
            return 0.0

        val = float(stats.skew(returns))
        return val if not np.isnan(val) else 0.0

    @staticmethod
    def calculate_kurtosis(returns: np.ndarray) -> float:

        if len(returns) < 4:
            return 0.0

        val = float(stats.kurtosis(returns))
        return val if not np.isnan(val) else 0.0

    @classmethod
    def calculate_all(cls, returns: list or np.ndarray, risk_free_rate: float = 0.0) -> dict:

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