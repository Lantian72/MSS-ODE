"""Runnable demonstration of MSS-ODE on the synthetic multi-regime benchmark (Section 4 & Appendix C)."""
import argparse
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mss_ode import (
    coefficients,
    step_filter,
    optimize_components,
    forecast_conditional,
    reference_arrays,
    fixed_pass,
    confirm_references,
)
from synthetic_benchmark.simulate import (
    N,
    TRAIN,
    HORIZONS,
    Y_NOISE,
    DEFAULT_CONFIG,
    generate_trajectory,
)

MATCH_TOLERANCE = 200


def localization_metrics(refs, oracle_refs):
    """Compute TP, FP, FN, precision, recall, and median confirmation delay against ground-truth transitions."""
    used = set()
    delays = []
    unmatched = []
    for t in refs:
        choices = [
            (j, int(e))
            for j, e in enumerate(oracle_refs)
            if j not in used and int(e) <= int(t) <= int(e) + MATCH_TOLERANCE
        ]
        if choices:
            j, e = choices[0]
            used.add(j)
            delays.append(int(t) - e)
        else:
            unmatched.append(int(t))
    tp = len(delays)
    fp = len(unmatched)
    fn = len(oracle_refs) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / len(oracle_refs) if len(oracle_refs) > 0 else 1.0
    med_delay = float(np.median(delays)) if delays else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, median_extra_delay=med_delay)


def train_warmup(data, cfg, passes=600):
    x, y = data["x"][:TRAIN], data["y"][:TRAIN]
    u, up, z, _, reset = reference_arrays(x, y, np.array([0]))
    theta = np.tile([0.0, 5.0, 1.0, 0.0], (4, 1))
    for _ in range(passes):
        states, P, M, _ = fixed_pass(
            u, up, z, reset, theta, cfg["omega2"], cfg["sigma2"], cfg["initial_cov"], cfg["reset_cov"]
        )
        optimize_components(theta, M, cfg, profile=True)
    states, P, M, _ = fixed_pass(
        u, up, z, reset, theta, cfg["omega2"], cfg["sigma2"], cfg["initial_cov"], cfg["reset_cov"]
    )
    return dict(theta=theta, mu=states[-1], P=P, M=M)


def evaluate_replay(data, refs, state, cfg):
    x, y = data["x"], data["y"]
    theta, mu, P, M = [state[k].copy() for k in ("theta", "mu", "P", "M")]
    co = coefficients(theta)
    refset = set(int(r) for r in refs)
    anchor_x = x[0].copy()
    anchor_y = y[0]
    predictions = np.full((N - TRAIN, len(HORIZONS)), np.nan)
    for t in range(TRAIN, N):
        reset = t in refset
        if reset:
            anchor_x = x[t].copy()
            anchor_y = y[t]
        u = x[t] - anchor_x
        up = x[t - 1] - anchor_x
        z = y[t] - anchor_y
        mu, P, m = step_filter(mu, P, co, theta, u, up, z, cfg, reset)
        if m is not None:
            gamma = (t + 1 + cfg["gamma_offset"]) ** (-cfg["gamma_power"])
            M = (1.0 - gamma) * M + gamma * m
            optimize_components(theta, M, cfg, profile=False)
            co = coefficients(theta)
        predictions[t - TRAIN] = forecast_conditional(mu, co, theta, x, anchor_x, t, HORIZONS) + anchor_y
    predictions = predictions * data["y_scale"] + data["y_mean"]
    origins = np.arange(TRAIN, N - HORIZONS[-1])
    rows = origins - TRAIN
    rmse = []
    for j, h in enumerate(HORIZONS):
        targets = origins + h
        error = predictions[rows, j] - data["raw_y"][targets]
        rmse.append(float(np.sqrt(np.mean(error**2))))
    return rmse


def main():
    parser = argparse.ArgumentParser(description="Run synthetic multi-regime MSS-ODE benchmark.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (0..29 in Appendix C)")
    parser.add_argument("--family", type=str, default="baseline", help="Scenario family from Table C1")
    parser.add_argument("--level", type=float, default=0.0, help="Scenario severity level")
    parser.add_argument("--passes", type=int, default=600, help="Warm-up training passes (default: 600)")
    args = parser.parse_args()

    print(f"Generating synthetic multi-regime trajectory (seed={args.seed}, family={args.family}, level={args.level})...")
    data = generate_trajectory(seed=args.seed, family=args.family, level=args.level)
    cfg = DEFAULT_CONFIG.copy()
    cfg["sigma2"] = (Y_NOISE / data["y_scale"]) ** 2

    print(f"Running warm-up parameter estimation ({args.passes} passes) on training prefix...")
    state = train_warmup(data, cfg, passes=args.passes)

    refs_ar, _, _ = confirm_references(data["x"], cfg, rule="ar", min_activation=TRAIN)
    refs_trend, _, _ = confirm_references(data["x"], cfg, rule="trend", min_activation=TRAIN)
    schedules = {
        "MSS-ODE (PELT + AR confirm)": refs_ar,
        "Trend-confirm (PELT + slope)": refs_trend,
        "Truth-timed (Oracle timing)": data["oracle_refs"],
        "SSM (Fixed initial reference)": np.array([], dtype=np.int64),
    }

    loc_ar = localization_metrics(refs_ar, data["oracle_refs"])
    print("\nConfirmed reference indices (evaluation period):", refs_ar.tolist())
    print("Oracle transition indices (evaluation period):  ", data["oracle_refs"].tolist())
    print(
        f"Reference localization (MSS-ODE): TP={loc_ar['tp']}, FP={loc_ar['fp']}, FN={loc_ar['fn']}, "
        f"Recall={loc_ar['recall']:.1%}, Precision={loc_ar['precision']:.1%}, "
        f"Median extra delay={loc_ar['median_extra_delay']:.1f} min"
    )
    print("\nEvaluating walk-forward multi-step forecasting RMSE:")
    header = f"{'Method':30s} | " + " | ".join(f"{h:2d} min" for h in HORIZONS)
    print(header)
    print("-" * len(header))
    for name, refs in schedules.items():
        rmse = evaluate_replay(data, refs, state, cfg)
        print(f"{name:30s} | " + " | ".join(f"{v:6.3f}" for v in rmse))


if __name__ == "__main__":
    main()
