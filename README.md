# StatArb_Engine: Multi-Factor Statistical Arbitrage Engine

`StatArb_Engine` is an end-to-end, production-grade walk-forward systematic multi-factor statistical arbitrage and pairs/basket trading strategy pipeline developed in Python. The system decouples signal extraction, econometric verification, factor risk decomposition, and convex portfolio optimization into a modular architecture.

---

## 1. System Architecture & Directory Structure

```
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
```
---

## 2. Core Modules & Mathematical Specification

### Module 1: Data Sanitization & Signal Generation
- **Liquidity Masking & Sanitization:**
  Computes a 20-day trailing Average Daily Dollar Volume ($\text{ADTV}$) panel. Assets falling below \$10M ADTV at bar $t$ are dynamically masked out of signal generation and portfolio construction.

  $$\text{ADTV}_{i,t} = \frac{1}{20} \sum_{\tau=0}^{19} (P_{i,t-\tau} \cdot V_{i,t-\tau})$$

  Log-price panel: $y_i(t) = \ln(P_i(t))$ with strict forward-fill limits ($\le 3$ bars).

- **Online State-Space Kalman Filtering:**
  State transition equation:

  $$\mathbf{\theta}_t = [\alpha_t, \beta_t]^T = \mathbf{\theta}_{t-1} + \mathbf{w}_t, \quad \mathbf{w}_t \sim \mathcal{N}(0, \mathbf{Q}_t)$$

  Measurement equation:

  $$y_{1,t} = \mathbf{x}_t^T \mathbf{\theta}_t + v_t, \quad v_t \sim \mathcal{N}(0, R_t), \quad \mathbf{x}_t = [1, y_{2,t}]^T$$

  Innovation Z-score:

  $$e_t = y_{1,t} - \mathbf{x}_t^T \hat{\mathbf{\theta}}_{t|t-1}, \quad F_t = \mathbf{x}_t^T \mathbf{P}_{t|t-1} \mathbf{x}_t + R_t, \quad Z_t = \frac{e_t}{\sqrt{F_t}}$$

- **Ornstein-Uhlenbeck (OU) Spread Dynamics:**
  Discrete AR(1) spread process: $\Delta s_t = \alpha + \gamma s_{t-1} + \epsilon_t$. Domain restriction enforces $-1.0 < \gamma < 0.0 \implies \theta = -\ln(1 + \gamma) > 0$.
  Continuous half-life: $T_{1/2} = \frac{\ln(2)}{\theta}$. Pairs are restricted to $0.5 \le T_{1/2} \le 60.0\text{ days}$.

- **Stateful Signal Hysteresis Matrix** ($S_{k,t} \in \{-1, 0, 1\}$):
  - Entry Trigger: $|Z_t| \ge Z_{\text{entry}} = 2.0$
  - Target Exit: $|Z_t| \le Z_{\text{exit}} = 0.5$
  - Hard Risk Cutoff: $|Z_t| \ge Z_{\text{stop}} = 3.5$ (Immediate exit on non-stationary cointegration break)
  - Rolling Dickey-Fuller (ADF) Check: Requires $p \le 0.05$ on a rolling 126-bar window.

- **Signal Intensity & Weight Vector Allocation:**
  Inverse-spread-volatility scaled signal intensity:

  $$q_k = -S_{k,t} \cdot \arctan(\max(0, |Z_t| - Z_{\text{exit}})) \cdot \exp\left(-\frac{T_{1/2}}{\tau}\right) \cdot \frac{1}{\sigma_{\text{spread}}}$$

  Per-pair dollar-neutral weight allocation:

  $$p_{\text{vec}, A} = \frac{1}{1 + |\beta|} \cdot q_k, \quad p_{\text{vec}, B} = \frac{-\beta}{1 + |\beta|} \cdot q_k$$

---

### Module 2: Factor Risk Decomposition & Covariance Shrinkage
- **PCA Factor Extraction:**
  Decomposes return matrix $R_{\text{norm}} \in \mathbb{R}^{T \times N}$ into $K = \min(K_{\text{config}}, N-1)$ components. Factor loading matrix $B \in \mathbb{R}^{N \times K}$ and idiosyncratic residual return panel:

  $$\tilde{R} = (R_{\text{norm}} - F_K B^T) \odot \sigma$$

- **Covariance Matrix Reconstruction:**
  Shrinks residual covariance via analytical Ledoit-Wolf shrinkage: $D_{\epsilon} = \text{Cov}_{\text{LW}}(\tilde{R})$.
  Reconstructed portfolio covariance matrix:

  $$\mathbf{\Sigma}_{\text{full}} = B \mathbf{\Sigma}_F B^T + D_{\epsilon}$$

---

### Module 3: Convex Portfolio Optimization (Markowitz Rescaled)
- **$L_\infty$ Signal Conviction Normalization:**

  $$\alpha_{\text{conviction}} = \frac{\alpha_{\text{raw}}}{\max_i |\alpha_{\text{raw}, i}|} \in [-1.0, 1.0]$$

  $$\mu_{\text{daily}} = \alpha_{\text{conviction}} \cdot \left(\frac{\text{expected\_pair\_alpha\_bps}}{10000}\right)$$

- **Optimization Formulation:**

  $$\max_{\mathbf{w}} \quad \mathbf{w}^T \mathbf{\mu}_{\text{scaled}} - \frac{\gamma}{2} \mathbf{w}^T \mathbf{\Sigma}_{\text{scaled}} \mathbf{w} - \lambda_{\text{tc, scaled}} \|\mathbf{w} - \mathbf{w}_{\text{prev}}\|_1 - \frac{\phi_{\text{quad}}}{2} \|\mathbf{w} - \mathbf{w}_{\text{prev}}\|_2^2$$

  $$\begin{aligned}
  \text{s.t.} \quad & \mathbf{1}^T \mathbf{w} = 0 && \text{(Dollar Neutrality)} \\
  & \|\mathbf{w}\|_1 \le L_{\text{max}} = 2.0 && \text{(Gross Leverage Cap)} \\
  & B^T \mathbf{w} = 0 && \text{(Factor Neutrality Constraint)} \\
  & -w_{\text{max}} \le w_i \le w_{\text{max}} = 0.10 && \text{(Single-Name Position Limit)}
  \end{aligned}$$

- **Solver Cascade:** Priority fallback order CLARABEL $\rightarrow$ OSQP ($10^{-4}$ tolerance relaxation) $\rightarrow$ SCS.---

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
