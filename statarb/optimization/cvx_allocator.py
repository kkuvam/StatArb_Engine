from typing import Optional
import cvxpy as cp
import numpy as np
import pandas as pd


class ConvexPortfolioAllocator:
    def __init__(
        self,
        gamma: float = 0.5,
        max_gross_leverage: float = 2.0,
        max_position_weight: float = 0.10,
        transaction_cost_bps: float = 5.0,
        quadratic_turnover_penalty: float = 0.05,
        scale_factor: float = 10000.0,
        expected_pair_alpha_bps: float = 25.0,
    ):
        self.gamma = gamma
        self.max_gross_leverage = max_gross_leverage
        self.max_position_weight = max_position_weight
        self.tc_penalty_raw = transaction_cost_bps / 10000.0
        self.quad_turnover_penalty = quadratic_turnover_penalty * scale_factor
        self.scale_factor = scale_factor
        self.daily_alpha_scale = expected_pair_alpha_bps / 10000.0

    def optimize(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        factor_loadings: Optional[np.ndarray] = None,
        prev_weights: Optional[pd.Series] = None,
    ) -> pd.Series:
        assets = expected_returns.index
        N = len(assets)

        daily_expected_returns = expected_returns.values * self.daily_alpha_scale
        mu_scaled = daily_expected_returns * self.scale_factor
        Sigma_scaled = cov_matrix.values * self.scale_factor
        tc_penalty_scaled = self.tc_penalty_raw * self.scale_factor

        w = cp.Variable(N)
        portfolio_return = mu_scaled @ w
        portfolio_risk = cp.quad_form(w, Sigma_scaled)

        if prev_weights is not None:
            prev_w_vec = prev_weights.reindex(assets).fillna(0.0).values
            delta_w = w - prev_w_vec
            linear_cost = tc_penalty_scaled * cp.norm(delta_w, 1)
            quadratic_cost = (self.quad_turnover_penalty / 2.0) * cp.sum_squares(delta_w)
        else:
            linear_cost = tc_penalty_scaled * cp.norm(w, 1)
            quadratic_cost = 0.0

        objective = cp.Maximize(
            portfolio_return 
            - (self.gamma / 2.0) * portfolio_risk 
            - linear_cost 
            - quadratic_cost
        )

        constraints = [
            cp.sum(w) == 0.0,
            cp.norm(w, 1) <= self.max_gross_leverage,
            w >= -self.max_position_weight,
            w <= self.max_position_weight,
        ]

        if factor_loadings is not None and factor_loadings.shape[0] == N:
            K = factor_loadings.shape[1]
            if N > K:
                constraints.append(factor_loadings.T @ w == np.zeros(K))

        problem = cp.Problem(objective, constraints)

        for solver_choice in [cp.CLARABEL, cp.OSQP, cp.SCS]:
            try:
                if solver_choice == cp.CLARABEL:
                    problem.solve(solver=solver_choice, verbose=False)
                elif solver_choice == cp.OSQP:
                    problem.solve(
                        solver=solver_choice,
                        eps_abs=1e-4,
                        eps_rel=1e-4,
                        max_iter=30000,
                    )
                else:
                    problem.solve(solver=solver_choice, verbose=False)

                if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    return pd.Series(np.round(w.value, 6), index=assets)
            except cp.SolverError:
                continue

        return pd.Series(0.0, index=assets)