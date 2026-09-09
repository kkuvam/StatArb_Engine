import itertools
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from statarb.optimization.cvx_allocator import ConvexPortfolioAllocator
from statarb.risk.covariance import CovarianceRegularizer
from statarb.risk.pca import PCARiskEngine
from statarb.signals.kalman import KalmanHedgeEngine
from statarb.signals.ou_process import OUProcessAnalyzer


class StatefulPairSignalTracker:
    """Tracks state hysteresis S_{k,t} in {-1, 0, 1} across backtest iterations

    with hard Z_stop risk limits and inverse-volatility scaling.
    """

    def __init__(
        self,
        z_entry: float = 2.0,
        z_exit: float = 0.5,
        z_stop: float = 3.5,
        tau_decay: float = 15.0,
    ):
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.z_stop = z_stop
        self.tau_decay = tau_decay
        self.pair_states: Dict[Tuple[str, str], int] = {}

    def compute_pair_signal(
        self,
        asset_A: str,
        asset_B: str,
        z_score: float,
        beta: float,
        half_life: float,
        spread_std: float,
    ) -> Tuple[float, float]:
        pair_key = (asset_A, asset_B)
        current_state = self.pair_states.get(pair_key, 0)

        # 1. State Hysteresis & Hard Z-Stop Evaluation
        abs_z = abs(z_score)

        if abs_z >= self.z_stop:
            new_state = 0  # Hard risk exit on spread divergence / structural break
        elif z_score >= self.z_entry:
            new_state = -1  # Short spread (Short Asset A, Long Asset B)
        elif z_score <= -self.z_entry:
            new_state = 1  # Long spread (Long Asset A, Short Asset B)
        elif abs_z <= self.z_exit:
            new_state = 0  # Target exit reached
        else:
            new_state = current_state  # Hold current state

        self.pair_states[pair_key] = new_state

        if new_state == 0 or not (0.5 <= half_life <= 60.0):
            return 0.0, 0.0

        # 2. Continuous Signal Intensity Calculation
        decay_weight = np.exp(-half_life / self.tau_decay)
        effective_z = max(0.0, abs_z - self.z_exit)
        intensity = new_state * np.arctan(effective_z) * decay_weight

        # 3. Inverse Spread Volatility Scaling
        vol_scalar = 1.0 / max(spread_std, 1e-4)
        scaled_intensity = intensity * vol_scalar

        # 4. Individual Pair Dollar-Neutral Vector
        denom = 1.0 + abs(beta)
        weight_A = (1.0 / denom) * scaled_intensity
        weight_B = (-beta / denom) * scaled_intensity

        return weight_A, weight_B


class RefactoredWalkForwardEngine:
    """Walk-forward backtest engine with point-in-time ADTV filtering,

    rolling ADF stationarity verification, factor risk decomposition,
    and CVXPY portfolio optimization.
    """

    def __init__(
        self,
        config: Dict,
        lookback_window: int = 126,
        transaction_cost_bps: float = 5.0,
        z_entry: float = 2.0,
        z_exit: float = 0.5,
        z_stop: float = 3.5,
        tau_decay: float = 15.0,
        min_trade_threshold: float = 0.01,
    ):
        self.config = config
        self.lookback_window = lookback_window
        self.tc_penalty = transaction_cost_bps / 10000.0
        self.min_trade_threshold = min_trade_threshold

        # Core System Engines
        self.kalman = KalmanHedgeEngine(
            delta=float(config["signals"]["kalman_delta"]),
            obs_noise=float(config["signals"]["kalman_obs_noise"]),
        )
        self.ou_analyzer = OUProcessAnalyzer()
        self.pca_engine = PCARiskEngine(
            n_components=int(config["risk"]["pca_components"])
        )
        self.cov_reg = CovarianceRegularizer()

        # Stateful Signal Tracker Integration
        self.signal_tracker = StatefulPairSignalTracker(
            z_entry=z_entry,
            z_exit=z_exit,
            z_stop=z_stop,
            tau_decay=tau_decay,
        )

        # Convex Mean-Variance Portfolio Allocator
        self.allocator = ConvexPortfolioAllocator(
            gamma=float(config["optimization"]["target_gamma"]),
            max_gross_leverage=float(config["optimization"]["max_gross_leverage"]),
            max_position_weight=float(config["optimization"]["max_position_weight"]),
            transaction_cost_bps=transaction_cost_bps,
            quadratic_turnover_penalty=0.05,
            expected_pair_alpha_bps=25.0,
        )

    def _verify_rolling_stationarity(
        self, spread_window: pd.Series, max_pvalue: float = 0.05
    ) -> bool:
        """Executes Augmented Dickey-Fuller unit-root test on rolling spread window."""
        clean_spread = spread_window.dropna()
        if len(clean_spread) < 30:
            return False
        try:
            p_val = float(adfuller(clean_spread, autolag="AIC")[1])
            return p_val <= max_pvalue
        except Exception:
            return False

    def run_backtest(
        self, log_prices: pd.DataFrame, liquidity_mask: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        tickers = log_prices.columns.tolist()
        dates = log_prices.index[self.lookback_window :]
        T_sim = len(dates)

        all_pairs = list(itertools.combinations(tickers, 2))
        prices = np.exp(log_prices)
        simple_returns = prices.pct_change().fillna(0.0)

        # 1. Clear stateful pair tracker memory to ensure backtest run isolation
        self.signal_tracker.pair_states.clear()

        # 2. Pre-filter Kalman states across full timeline: O(T * M) complexity
        kalman_cache = {}
        for asset_A, asset_B in all_pairs:
            kalman_cache[(asset_A, asset_B)] = self.kalman.filter_pair(
                log_prices[asset_A], log_prices[asset_B]
            )

        portfolio_returns = np.zeros(T_sim)
        turnovers = np.zeros(T_sim)
        active_pairs_count = np.zeros(T_sim)
        prev_w = pd.Series(0.0, index=tickers)

        # 3. Main Walk-Forward Simulation Loop
        for i, t in enumerate(dates):
            t_idx = log_prices.index.get_loc(t)
            window_returns = simple_returns.iloc[
                t_idx - self.lookback_window : t_idx + 1
            ]

            # Apply Point-in-Time ADTV Liquidity Mask
            if liquidity_mask is not None:
                active_assets = (
                    liquidity_mask.loc[t][liquidity_mask.loc[t]].index.tolist()
                )
            else:
                active_assets = tickers

            raw_asset_weights = pd.Series(0.0, index=tickers)
            active_pairs = 0

            # Evaluate candidate pairs
            for asset_A, asset_B in all_pairs:
                if asset_A not in active_assets or asset_B not in active_assets:
                    continue

                kalman_df = kalman_cache[(asset_A, asset_B)].iloc[: t_idx + 1]
                latest_z = float(kalman_df["z_score"].iloc[-1])
                latest_beta = float(kalman_df["beta"].iloc[-1])
                spread_window = kalman_df["spread"].iloc[-self.lookback_window :]

                # 3A. Rolling ADF Stationarity Verification
                is_stationary = self._verify_rolling_stationarity(
                    spread_window, max_pvalue=0.05
                )
                if not is_stationary:
                    # Force exit in tracker if spread loses stationarity
                    self.signal_tracker.pair_states[(asset_A, asset_B)] = 0
                    continue

                half_life = self.ou_analyzer.estimate_half_life(spread_window)
                spread_std = float(spread_window.std())

                # 3B. Compute Pair Allocation Vector
                w_A, w_B = self.signal_tracker.compute_pair_signal(
                    asset_A=asset_A,
                    asset_B=asset_B,
                    z_score=latest_z,
                    beta=latest_beta,
                    half_life=half_life,
                    spread_std=spread_std,
                )

                if w_A != 0.0 or w_B != 0.0:
                    raw_asset_weights[asset_A] += w_A
                    raw_asset_weights[asset_B] += w_B
                    active_pairs += 1

            active_pairs_count[i] = active_pairs

            # 4. Formulate Conviction Vector & Portfolio Optimization
            if active_pairs > 0 and np.abs(raw_asset_weights).sum() > 0:
                # L-infinity Signal Scaling into [-1.0, 1.0] conviction space
                max_signal_intensity = np.abs(raw_asset_weights).max()
                expected_alpha = raw_asset_weights / max_signal_intensity

                # Covariance Matrix Reconstruction: \Sigma_full = B * \Sigma_F * B^T + D_\epsilon
                try:
                    (
                        res_df,
                        B,
                        F_cov,
                    ) = self.pca_engine.extract_residuals_and_covariance(
                        window_returns[active_assets]
                    )
                    D_eps = self.cov_reg.shrink_covariance(res_df).values

                    Sigma_total = B @ F_cov @ B.T + D_eps
                    cov_matrix = pd.DataFrame(
                        Sigma_total, index=active_assets, columns=active_assets
                    )
                    factor_loadings = B
                except Exception:
                    cov_matrix = window_returns[active_assets].cov()
                    factor_loadings = None

                # Solve Convex Optimization Problem
                try:
                    curr_w_active = self.allocator.optimize(
                        expected_returns=expected_alpha[active_assets],
                        cov_matrix=cov_matrix,
                        factor_loadings=factor_loadings,
                        prev_weights=prev_w[active_assets],
                    )
                    curr_w = pd.Series(0.0, index=tickers)
                    curr_w[active_assets] = curr_w_active
                except Exception:
                    curr_w = prev_w
            else:
                curr_w = pd.Series(0.0, index=tickers)

            # 5. Position Trade Buffer Filter (Eliminate micro-allocations < 1%)
            weight_diff = np.abs(curr_w - prev_w)
            curr_w = pd.Series(
                np.where(
                    weight_diff >= self.min_trade_threshold, curr_w, prev_w
                ),
                index=tickers,
            )

            # 6. Performance & Execution Cost Accounting
            if t_idx < len(log_prices) - 1:
                next_day_return = simple_returns.iloc[t_idx + 1]
                gross_return = float(curr_w @ next_day_return)
                turnover = float(np.sum(np.abs(curr_w - prev_w)))
                cost = turnover * self.tc_penalty

                portfolio_returns[i] = gross_return - cost
                turnovers[i] = turnover

            prev_w = curr_w

        # 7. Generate NAV Curve & Calculate Strategy Summary Statistics
        nav = (1.0 + pd.Series(portfolio_returns, index=dates)).cumprod()
        stats = self._calculate_metrics(
            nav, portfolio_returns, turnovers, active_pairs_count
        )

        results_df = pd.DataFrame(
            {
                "NAV": nav,
                "Daily_Return": portfolio_returns,
                "Turnover": turnovers,
                "Active_Pairs": active_pairs_count,
            },
            index=dates,
        )
        return results_df, stats

    def _calculate_metrics(
        self,
        nav: pd.Series,
        returns: np.ndarray,
        turnovers: np.ndarray,
        active_pairs: np.ndarray,
    ) -> Dict[str, float]:
        total_return = float(nav.iloc[-1] - 1.0)
        ann_return = float(np.mean(returns) * 252)
        ann_vol = float(np.std(returns) * np.sqrt(252))
        sharpe = ann_return / (ann_vol + 1e-8)
        peak = nav.cummax()
        drawdown = (nav - peak) / peak

        return {
            "Total Return": total_return,
            "Annualized Return": ann_return,
            "Annualized Volatility": ann_vol,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": float(drawdown.min()),
            "Average Daily Turnover": float(np.mean(turnovers)),
            "Average Active Pairs": float(np.mean(active_pairs)),
        }


# Backwards-compatibility class alias
RobustWalkForwardEngine = RefactoredWalkForwardEngine