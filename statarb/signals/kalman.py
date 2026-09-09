import numpy as np
import pandas as pd


class KalmanHedgeEngine:
    """Online State-Space Kalman Filter for dynamic pair hedge ratios and signal generation.

    Maintains strict temporal causality to prevent lookahead bias.
    """

    def __init__(self, delta: float = 1e-4, obs_noise: float = 1e-3):
        """
        Parameters
        ----------
        delta : float
            State transition covariance factor Q = (delta / (1 - delta)) * I.
        obs_noise : float
            Measurement noise variance R.
        """
        self.delta = delta
        self.obs_noise = obs_noise

    def filter_pair(self, y1: pd.Series, y2: pd.Series) -> pd.DataFrame:
        """Processes pair model: y1_t = intercept_t + beta_t * y2_t + e_t.

        Returns DataFrame containing dynamic intercept, beta, spread, innovation variance, and Z-score.
        """
        assert len(y1) == len(y2), "Asset series lengths must be identical."

        n = len(y1)
        y1_vals = y1.values
        y2_vals = y2.values

        # Initialize state vector theta = [intercept, beta]^T
        theta = np.zeros(2)
        P = np.eye(2)
        Q = (self.delta / (1.0 - self.delta)) * np.eye(2)
        R = self.obs_noise

        intercepts = np.zeros(n)
        betas = np.zeros(n)
        spreads = np.zeros(n)
        variances = np.zeros(n)

        for t in range(n):
            # Regressor vector x_t = [1, y2_t]^T
            x_t = np.array([1.0, y2_vals[t]])

            # 1. State Prediction
            theta_pred = theta
            P_pred = P + Q

            # 2. Measurement Innovation
            y_hat = float(x_t @ theta_pred)
            e_t = y1_vals[t] - y_hat
            F_t = float(x_t @ P_pred @ x_t.T + R)

            # 3. Kalman Gain & State Correction
            K_t = (P_pred @ x_t.T) / F_t
            theta = theta_pred + K_t * e_t
            P = (np.eye(2) - np.outer(K_t, x_t)) @ P_pred

            intercepts[t] = theta[0]
            betas[t] = theta[1]
            spreads[t] = e_t
            variances[t] = F_t

        results = pd.DataFrame(
            {
                "intercept": intercepts,
                "beta": betas,
                "spread": spreads,
                "variance": variances,
            },
            index=y1.index,
        )

        # Dynamic normalized Z-score signal
        results["z_score"] = results["spread"] / np.sqrt(results["variance"])

        return results