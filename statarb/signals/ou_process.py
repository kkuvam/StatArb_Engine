import numpy as np
import pandas as pd
import statsmodels.api as sm


class OUProcessAnalyzer:
    """Estimates mean-reversion speed and half-life for cointegrated spreads."""

    @staticmethod
    def estimate_half_life(spread: pd.Series) -> float:
        """Estimates half-life T_1/2 of an Ornstein-Uhlenbeck spread process.

        Model:
            \Delta s_t = \alpha + \gamma * s_{t-1} + \epsilon_t
            \theta = -\ln(1 + \gamma)
            T_{1/2} = \ln(2) / \theta

        Returns
        -------
        float
            Half-life in trading days. Returns 999.0 if the process is non-stationary
            (\gamma >= 0) or over-correcting (\gamma <= -1).
        """
        clean_spread = spread.dropna()
        if len(clean_spread) < 30:
            return 999.0

        spread_lag = clean_spread.shift(1).iloc[1:]
        spread_diff = clean_spread.diff().iloc[1:]

        # OLS regression of \Delta s_t on s_{t-1}
        X = sm.add_constant(spread_lag.values)
        model = sm.OLS(spread_diff.values, X).fit()
        
        gamma_hat = float(model.params[1])

        # Domain Guardrail: Enforce -1.0 < gamma_hat < 0.0
        # If gamma_hat >= 0.0: Non-stationary / random walk
        # If gamma_hat <= -1.0: Over-correcting / non-continuous discrete artifact
        if gamma_hat >= 0.0 or gamma_hat <= -1.0:
            return 999.0

        theta = -np.log(1.0 + gamma_hat)
        half_life = np.log(2.0) / theta
        
        return float(half_life)