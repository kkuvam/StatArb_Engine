# StatArb_Engine: Multi-Factor Statistical Arbitrage Engine

`StatArb_Engine` is an end-to-end, production-grade walk-forward systematic multi-factor statistical arbitrage and pairs/basket trading strategy pipeline developed in Python. The system decouples signal extraction, econometric verification, factor risk decomposition, and convex portfolio optimization into a modular architecture.

---

## 1. System Architecture & Directory Structure

StatArb_Engine/
├── config/
│   ├── params.yaml              # Strategy hyperparameters & risk parameters
│   └── universe.py              # Flat ticker utilities & sector pair mappings
├── data/
│   └── cache/                   # Parquet disk cache for market data (Git-ignored)
├── statarb/
│   ├── __init__.py
│   ├── data/
│   │   ├── ingestion.py         # CachedYFinanceDataHandler (PIT total return retrieval)
│   │   └── sanitization.py      # DataSanitizer (ADTV liquidity mask & log-price panel)
│   ├── signals/
│   │   ├── cointegration.py     # CointegrationScanner (Johansen VECM & rolling ADF)
│   │   ├── kalman.py            # KalmanHedgeEngine (Dynamic state-space hedge ratios)
│   │   ├── ou_process.py        # OUProcessAnalyzer (Continuous OU half-life estimation)
│   │   └── stateful_tracker.py  # StatefulPairSignalTracker (Hysteresis state memory)
│   ├── risk/
│   │   ├── pca.py               # PCARiskEngine (Factor extraction & residual returns)
│   │   └── covariance.py        # CovarianceRegularizer (Ledoit-Wolf shrinkage)
│   ├── optimization/
│   │   ├── cvx_allocator.py     # ConvexPortfolioAllocator (Scaled CVXPY solver)
│   │   ├── impact_allocator.py  # Almgren-Chriss non-linear market impact solver
│   │   └── black_litterman.py   # StatArbBlackLittermanEngine (View blending)
│   └── backtest/
│       ├── engine.py            # RefactoredWalkForwardEngine (Simulation harness)
│       └── tearsheet.py         # BacktestTearSheet (Performance diagnostics)
├── tests/
│   └── test_system_suite.py     # System test harness
├── .gitignore
├── README.md
└── requirements.txt

---

## 2. Core Modules & Mathematical Specification

### Module 1: Data Sanitization & Signal Generation
- Liquidity Masking & Sanitization:
  Computes a 20-day trailing Average Daily Dollar Volume (ADTV) panel. Assets falling below $10M ADTV at bar t are dynamically masked out of signal generation and portfolio construction.
  ADTV_{i,t} = (1/20) * sum_{tau=0}^{19} (P_{i,t-tau} * V_{i,t-tau})
  Log-price panel: y_i(t) = ln(P_i(t)) with strict forward-fill limits (<= 3 bars).

- Online State-Space Kalman Filtering:
  State transition equation:
  \theta_t = [\alpha_t, \beta_t]^T = \theta_{t-1} + w_t,  w_t ~ N(0, Q_t)
  Measurement equation:
  y_{1,t} = x_t^T \theta_t + v_t,  v_t ~ N(0, R_t),  x_t = [1, y_{2,t}]^T
  Innovation Z-score:
  e_t = y_{1,t} - x_t^T \hat{\theta}_{t|t-1},  F_t = x_t^T P_{t|t-1} x_t + R_t,  Z_t = e_t / sqrt(F_t)

- Ornstein-Uhlenbeck (OU) Spread Dynamics:
  Discrete AR(1) spread process: \Delta s_t = \alpha + \gamma s_{t-1} + \epsilon_t. Domain restriction enforces -1.0 < \gamma < 0.0 => \theta = -\ln(1 + \gamma) > 0.
  Continuous half-life: T_{1/2} = ln(2) / \theta. Pairs are restricted to 0.5 <= T_{1/2} <= 60.0 days.

- Stateful Signal Hysteresis Matrix (S_{k,t} in {-1, 0, 1}):
  - Entry Trigger: |Z_t| >= Z_{entry} = 2.0
  - Target Exit: |Z_t| <= Z_{exit} = 0.5
  - Hard Risk Cutoff: |Z_t| >= Z_{stop} = 3.5 (Immediate exit on non-stationary cointegration break)
  - Rolling Dickey-Fuller (ADF) Check: Requires p <= 0.05 on a rolling 126-bar window.

- Signal Intensity & Weight Vector Allocation:
  Inverse-spread-volatility scaled signal intensity:
  q_k = -S_{k,t} * arctan(max(0, |Z_t| - Z_{exit})) * exp(-T_{1/2}/\tau) * (1 / \sigma_{spread})
  Per-pair dollar-neutral weight allocation:
  p_{vec, A} = (1 / (1 + |\beta|)) * q_k,  p_{vec, B} = (-\beta / (1 + |\beta|)) * q_k

---

### Module 2: Factor Risk Decomposition & Covariance Shrinkage
- PCA Factor Extraction:
  Decomposes return matrix R_{norm} in R^{T x N} into K = min(K_{config}, N-1) components. Factor loading matrix B in R^{N x K} and idiosyncratic residual return panel:
  \tilde{R} = (R_{norm} - F_K B^T) \odot \sigma

- Covariance Matrix Reconstruction:
  Shrinks residual covariance via analytical Ledoit-Wolf shrinkage: D_{\epsilon} = Cov_{LW}(\tilde{R}).
  Reconstructed portfolio covariance matrix:
  \Sigma_{full} = B \Sigma_F B^T + D_{\epsilon}

---

### Module 3: Convex Portfolio Optimization (Markowitz Rescaled)
- L_\infty Signal Conviction Normalization:
  \alpha_{conviction} = \alpha_{raw} / max_i |\alpha_{raw, i}| in [-1.0, 1.0]
  \mu_{daily} = \alpha_{conviction} * (expected_pair_alpha_bps / 10000)

- Optimization Formulation:
  max_w  w^T \mu_{scaled} - (\gamma/2) w^T \Sigma_{scaled} w - \lambda_{tc, scaled} ||w - w_{prev}||_1 - (\phi_{quad}/2) ||w - w_{prev}||_2^2
  s.t.   1^T w = 0                          (Dollar Neutrality)
         ||w||_1 <= L_{max} = 2.0           (Gross Leverage Cap)
         B^T w = 0                          (Factor Neutrality Constraint)
         -w_{max} <= w_i <= w_{max} = 0.10  (Single-Name Position Limit)

- Solver Cascade: Priority fallback order CLARABEL -> OSQP (10^-4 tolerance relaxation) -> SCS.

---

## 3. Installation & Quickstart

### Environment Setup
git clone https://github.com/your-org/StatArb_Engine.git
cd StatArb_Engine

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt

### Running System Verification Tests
pytest tests/test_system_suite.py -v

### Executing Backtest Pipeline
from statarb.backtest.engine import RefactoredWalkForwardEngine
from statarb.backtest.tearsheet import BacktestTearSheet

engine = RefactoredWalkForwardEngine(config_path="config/params.yaml")
results = engine.run()

tearsheet = BacktestTearSheet(results)
tearsheet.generate_summary()

---

## 4. License
Distributed under the MIT License. See LICENSE for details.
