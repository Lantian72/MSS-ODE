"""Synthetic multi-regime continuous-time ZOH process simulator (Section 4 & Appendix C)."""
import numpy as np
from scipy.linalg import expm

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

N = 8000
TRAIN = 2400
EVENT_TIMES = [3200, 4400, 5600, 6800]
HORIZONS = np.array([10, 20, 30, 40, 50, 60], dtype=np.int64)
TRUE_THETA = np.array([
    [0.8, 8.0, 0.7, 0.0],
    [-0.6, 12.0, 1.0, 0.0],
    [0.4, 20.0, 1.2, 0.0],
    [0.0, 10.0, 0.9, 0.0],
])
X_NOISE = 0.08
Y_NOISE = 0.05

DEFAULT_CONFIG = dict(
    W=1440,
    penalty=100.0,
    min_size=30,
    L=50,
    alpha=0.05,
    merge=10,
    sigma2=1e-10,
    omega2=1e-4,
    reset_cov=1e-3,
    K_bounds=[-5.0, 5.0],
    tau_bounds=[1.0, 200.0],
    zeta_bounds=[0.1, 5.0],
    c_bounds=[-1.0, 1.0],
    initial_cov=1e-4,
    gamma_offset=10.0,
    gamma_power=0.6,
    maxiter=200,
    ftol=1e-9,
    gtol=1e-6,
)


def _rng(seed, tag):
    return np.random.default_rng(np.random.SeedSequence([20261022, int(seed), tag]))


@njit(cache=True)
def _plant(inputs, A, B):
    n, d = inputs.shape
    state = np.zeros((d, 2))
    components = np.empty((n, d))
    for t in range(n):
        for i in range(d):
            a = A[i, 0, 0] * state[i, 0] + A[i, 0, 1] * state[i, 1] + B[i, 0] * inputs[t, i]
            b = A[i, 1, 0] * state[i, 0] + A[i, 1, 1] * state[i, 1] + B[i, 1] * inputs[t, i]
            state[i, 0] = a
            state[i, 1] = b
            components[t, i] = a
    return components


def _physical_matrices():
    aa, bb = [], []
    for K, tau, zeta, _ in TRUE_THETA:
        aug = np.array([[0.0, 1.0, 0.0], [-1.0 / tau**2, -2.0 * zeta / tau, K / tau**2], [0.0, 0.0, 0.0]])
        e = expm(aug)
        aa.append(e[:2, :2])
        bb.append(e[:2, 2])
    return np.array(aa), np.array(bb)


def _add_transition(values, start, duration, delta):
    duration = int(duration)
    if duration == 0:
        fraction = (np.arange(len(values)) >= start).astype(float)
    else:
        fraction = np.clip((np.arange(len(values)) - start + 1) / duration, 0.0, 1.0)
    values += fraction * delta
    return start if duration == 0 else start + duration - 1


def generate_trajectory(seed=0, family="baseline", level=0):
    """Generate a synthetic 8,000-step multi-regime trajectory under the specified scenario family."""
    x = np.zeros((N, 4))
    levels_rng = _rng(seed, 1)
    for start in range(80, 2000, 80):
        x[start:] = levels_rng.normal(0.0, 1.0, 4)
    offset = np.zeros(N)
    rr = _rng(seed, 2)
    jumps = np.sign(rr.normal(size=(4, 4))) * rr.uniform(1.2, 1.8, (4, 4))
    if family == "correlated_mv":
        rho = float(level)
        C = np.linalg.cholesky((1.0 - rho) * np.eye(3) + rho * np.ones((3, 3)))
        jumps[:, :3] = jumps[:, :3] @ C.T
    affected = set()
    if family in ["irrelevant_mv", "cv_only"]:
        affected = set(_rng(seed, 3).permutation(4)[: int(round(float(level) * 4))].tolist())
    events = []
    for e, start in enumerate(EVENT_TIMES):
        jump = jumps[e].copy()
        output_jump = 0.8 * (1 if e % 2 == 0 else -1)
        duration = int(level) if family == "gradual" else 0
        gap = int(level) if family == "overlapping" else 0
        shift = int(level) if family == "asynchronous" else 0
        if family == "irrelevant_mv" and e in affected:
            jump[:3] = 0.0
            output_jump = 0.0
        if family == "cv_only" and e in affected:
            jump[:] = 0.0
        relevant_starts, relevant_ends = [], []
        for i in range(4):
            when = start + (i * gap if i < 3 else 0)
            if abs(jump[i]) > 1e-12:
                end = _add_transition(x[:, i], when, duration, jump[i])
                if TRUE_THETA[i, 0] != 0:
                    relevant_starts.append(when)
                    relevant_ends.append(end)
        if output_jump != 0:
            when = start + shift + 2 * gap
            end = _add_transition(offset, when, duration, output_jump)
            relevant_starts.append(when)
            relevant_ends.append(end)
        relevant = bool(relevant_starts)
        events.append(
            dict(
                event=e,
                start=min(relevant_starts) if relevant else start,
                end=max(relevant_ends) if relevant else start,
                relevant=relevant,
            )
        )
    A, B = _physical_matrices()
    components = _plant(x, A, B)
    clean_y = components.sum(axis=1) + offset
    ex = _rng(seed, 4).normal(size=(N, 4))
    ey = _rng(seed, 5).normal(size=N)
    if family == "correlated_noise":
        phi = float(level)
        previous = ex[TRAIN - 1].copy()
        for t in range(TRAIN, N):
            previous = phi * previous + np.sqrt(1.0 - phi * phi) * ex[t]
            ex[t] = previous
    noise_scale = np.ones(N)
    if family == "timevarying_noise":
        for start in EVENT_TIMES:
            noise_scale[start + 400 : min(N, start + 800)] = float(level)
    raw_x = x + X_NOISE * ex * noise_scale[:, None]
    raw_y = clean_y + Y_NOISE * ey * noise_scale
    xm = raw_x[:TRAIN].mean(0)
    xs = raw_x[:TRAIN].std(0)
    ym = raw_y[:TRAIN].mean()
    ys = raw_y[:TRAIN].std()
    equilibrium = x @ TRUE_THETA[:, 0] + offset
    oracle = []
    for event in events:
        eligible = event["end"] + DEFAULT_CONFIG["L"] - 1
        valid = event["relevant"] and eligible < N and np.ptp(equilibrium[event["end"] : eligible + 1]) < 1e-10
        if valid:
            oracle.append(eligible)
    return dict(
        x=np.ascontiguousarray((raw_x - xm) / xs),
        y=np.ascontiguousarray((raw_y - ym) / ys),
        raw_x=raw_x,
        raw_y=raw_y,
        clean_y=clean_y,
        x_mean=xm,
        x_scale=xs,
        y_mean=float(ym),
        y_scale=float(ys),
        oracle_refs=np.array(oracle, dtype=np.int64),
    )
