from typing import Dict, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen


class CointegrationScanner:
    """Performs Johansen VECM cointegration rank testing and ADF spread validation."""

    def __init__(self, confidence_level: str = "95%"):
        self.crit_map = {"90%": 0, "95%": 1, "99%": 2}
        self.crit_idx = self.crit_map.get(confidence_level, 1)

    def test_johansen(
        self, price_matrix: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1
    ) -> Dict[str, object]:
        """Executes Johansen multivariate cointegration test on aligned log-price panel.

        Parameters
        ----------
        price_matrix : pd.DataFrame
            T x N matrix of log prices.
        det_order : int
            Deterministic trend term (-1 = none, 0 = constant, 1 = linear trend).
        k_ar_diff : int
            Number of lagged differences in VECM specification.

        Returns
        -------
        Dict containing cointegration rank, critical values, and normalized cointegrating vector.
        """
        clean_matrix = price_matrix.dropna()
        if clean_matrix.shape[1] < 2:
            raise ValueError("At least 2 asset price series are required for cointegration testing.")

        # Run Johansen VECM canonical correlation estimation
        res = coint_johansen(clean_matrix.values, det_order, k_ar_diff)

        # Evaluate Trace and Max-Eigenvalue rank criteria
        trace_stats = res.lr1
        trace_crit = res.cvt[:, self.crit_idx]
        r_trace = int(np.sum(trace_stats > trace_crit))

        max_eig_stats = res.lr2
        max_eig_crit = res.cvm[:, self.crit_idx]
        r_max_eig = int(np.sum(max_eig_stats > max_eig_crit))

        # Primary normalized cointegrating eigenvector beta
        raw_beta = res.evec[:, 0]
        beta_normalized = raw_beta / raw_beta[0]

        is_cointegrated = (r_trace > 0) and (r_max_eig > 0)

        return {
            "is_cointegrated": is_cointegrated,
            "rank_trace": r_trace,
            "rank_max_eig": r_max_eig,
            "beta": beta_normalized,
            "trace_stats": trace_stats,
            "trace_crit": trace_crit,
        }

    def verify_spread_stationarity(
        self, spread: pd.Series, max_pvalue: float = 0.05
    ) -> Tuple[bool, float]:
        """Validates that implied error spread e_t is I(0) via Augmented Dickey-Fuller test."""
        clean_spread = spread.dropna()
        if len(clean_spread) < 30:
            return False, 1.0

        adf_res = adfuller(clean_spread, autolag="AIC")
        p_val = float(adf_res[1])
        return (p_val <= max_pvalue), p_val