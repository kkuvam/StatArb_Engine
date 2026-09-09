from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class PCARiskEngine:
    def __init__(self, n_components: int = 2):
        self.requested_components = n_components
        self.pca = None
        self.n_components_used = n_components

    def extract_residuals_and_covariance(
        self, returns: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        clean_returns = returns.dropna()
        T, N = clean_returns.shape
        
        # Dynamic component scaling for small universes
        self.n_components_used = max(1, min(self.requested_components, N - 1))
        self.pca = PCA(n_components=self.n_components_used)

        mean = clean_returns.mean(axis=0)
        std = clean_returns.std(axis=0).replace(0, 1e-8)
        norm_returns = (clean_returns - mean) / std

        factor_returns = self.pca.fit_transform(norm_returns)  # Shape: (T, K)
        loadings = self.pca.components_                       # Shape: (K, N)
        systematic_norm = factor_returns @ loadings            # Shape: (T, N)

        residual_returns = (norm_returns.values - systematic_norm) * std.values
        residual_df = pd.DataFrame(
            residual_returns, index=clean_returns.index, columns=clean_returns.columns
        )

        factor_cov = np.cov(factor_returns, rowvar=False)
        if self.n_components_used == 1:
            factor_cov = np.array([[factor_cov]])

        # Factor loadings B shape: (N, K)
        B = loadings.T * std.values[:, np.newaxis]
        
        return residual_df, B, factor_cov

    def get_factor_loadings(self) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("PCA Engine must be fit prior to requesting loadings.")
        return self.pca.components_.T