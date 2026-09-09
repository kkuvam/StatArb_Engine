import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


class CovarianceRegularizer:
    """Applies Ledoit-Wolf analytical shrinkage to residual covariance matrices."""

    def __init__(self):
        self.lw = LedoitWolf()

    def shrink_covariance(self, residual_returns: pd.DataFrame) -> pd.DataFrame:
        """Computes regularized positive definite covariance matrix Sigma_shrunk.

        Parameters
        ----------
        residual_returns : pd.DataFrame
            Residual returns matrix (T x N).

        Returns
        -------
        pd.DataFrame
            Shrunk covariance matrix (N x N).
        """
        clean_residuals = residual_returns.dropna()
        self.lw.fit(clean_residuals.values)
        
        shrunk_cov = self.lw.covariance_
        
        return pd.DataFrame(
            shrunk_cov,
            index=clean_residuals.columns,
            columns=clean_residuals.columns,
        )