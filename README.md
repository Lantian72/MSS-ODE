# MSS-ODE: Physics-Constrained Online State-Space Modeling

- **Manuscript Title:** Tracking Multi-Regime Manufacturing Dynamics Using Physics-Constrained Online State-Space Modeling
- **Authors:** Tian Lan, Xing Yang, Haijie Xu, and Chen Zhang (Corresponding Author: `zhangchen01@tsinghua.edu.cn`)
- **Journal:** *Journal of Intelligent Manufacturing*
- **Repository:** [https://github.com/Lantian72/MSS-ODE](https://github.com/Lantian72/MSS-ODE)

This repository provides the self-contained Python/Numba implementation of the **Multi-Steady-State Ordinary Differential Equation (MSS-ODE)** framework together with the synthetic multi-regime simulation benchmark (Section 4 and Appendix C).

---

## Directory Structure

- `mss_ode/`: Core implementation of the MSS-ODE framework
  - `model.py`: Exact zero-order-hold (ZOH) continuous-to-discrete coefficient mapping (`zoh_jac`), analytical Jacobians, $2d$-dimensional Kalman filter, exact one-step fixed-lag RTS smoothing moments (`step_filter`), bound-constrained L-BFGS-B component M-step updates (`optimize_components`), and multi-horizon state rollout functions (`forecast_held`, `forecast_conditional`).
  - `detection.py`: Causal rolling-window PELT segmentation (`pelt`, `scan_chunk`), finite-sample AR(1) weak-dependence confirmation (`ar_pvalue`), linear-trend confirmation (`trend_pvalue`), and multi-channel reference consolidation (`confirm_references`).
  - `order_delay.py`: Exact ZOH discretization and sensitivity Jacobians (`map_jac`) for first-order ($q=1$), second-order ($q=2$), and third-order cascade ($q=3$) transfer functions (Appendix G).
- `synthetic_benchmark/`: Synthetic multi-regime simulation benchmark (Section 4 & Appendix C)
  - `simulate.py`: Continuous-state ZOH multi-regime process simulator (`generate_trajectory`) supporting all eight scenario families in Table C1 (synchronized steps, asynchronous shifts, irrelevant inputs, CV-only shifts, gradual ramp drifts, correlated MV jumps, AR(1) measurement noise, overlapping multi-input steps, and time-varying noise bursts).
  - `run_synthetic_demo.py`: Runnable benchmark script that trains and evaluates MSS-ODE, Trend-confirm, Truth-timed, and fixed-reference SSM on synthetic multi-regime trajectories and reports both multi-horizon RMSE (10 to 60 min) and ground-truth reference-localization metrics (TP, FP, FN, recall, precision, and confirmation delay).
- `requirements.txt`: Python package dependencies (`numpy>=1.24.0`, `scipy>=1.10.0`, `numba>=0.58.0`).

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the synthetic multi-regime benchmark (Section 4 & Appendix C)
```bash
# Default synchronous multi-regime baseline (seed=0, 600 warm-up passes)
python synthetic_benchmark/run_synthetic_demo.py

# Example: evaluate under correlated MV jumps (rho=0.9) or gradual drift (D=300 min)
python synthetic_benchmark/run_synthetic_demo.py --family correlated_mv --level 0.9 --seed 0
python synthetic_benchmark/run_synthetic_demo.py --family gradual --level 300 --seed 0
```
