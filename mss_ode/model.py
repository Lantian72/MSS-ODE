"""Physics-constrained continuous-time ZOH state-space model and recursive EM.

Implements:
- Exact zero-order-hold (ZOH) mapping from continuous-time physical parameters
  (K_i, tau_i, zeta_i, c_i) to discrete-time companion coefficients
  (alpha_{i,1}, alpha_{i,2}, beta_{i,0}, beta_{i,1}) and analytical Jacobians (Eqs. 5-8).
- 2d-dimensional Kalman filter and exact one-step fixed-lag RTS smoothing moments (Section 3.2).
- Bound-constrained L-BFGS-B recursive M-step updates (Eq. 24).
- Multi-horizon state rollout under held inputs or exogenous input trajectories.
"""
import numpy as np
from scipy.optimize import minimize

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator


@njit(cache=True)
def zoh_jac(theta):
    """Compute exact ZOH discrete coefficients and Jacobian with respect to (K, tau, zeta, c)."""
    K, tau, zeta, c = theta
    d = 1.0 / tau
    x = d * d * (zeta * zeta - 1.0)
    if abs(x) < 1e-5:
        C = 1.0 + x / 2.0 + x * x / 24.0 + x**3 / 720.0 + x**4 / 40320.0
        S = 1.0 + x / 6.0 + x * x / 120.0 + x**3 / 5040.0 + x**4 / 362880.0
        Sp = 1.0 / 6.0 + x / 60.0 + x * x / 1680.0 + x**3 / 90720.0
    elif x > 0.0:
        r = np.sqrt(x)
        C = np.cosh(r)
        S = np.sinh(r) / r
        Sp = (C - S) / (2.0 * x)
    else:
        r = np.sqrt(-x)
        C = np.cos(r)
        S = np.sin(r) / r
        Sp = (C - S) / (2.0 * x)
    a = np.exp(-zeta * d)
    D = C + zeta * d * S
    coeff = np.empty(4)
    coeff[0] = 2.0 * a * C
    coeff[1] = -a * a
    step = 1.0 - a * D
    coeff[2] = K * step
    coeff[3] = K * (1.0 - coeff[0] - coeff[1]) - coeff[2]
    J = np.zeros((4, 4))
    J[2, 0] = step
    J[3, 0] = 1.0 - coeff[0] - coeff[1] - step
    for j in range(1, 3):
        if j == 1:
            dx = -2.0 * x / tau
            dd = -d / tau
            dz = 0.0
        else:
            dx = 2.0 * d * d * zeta
            dd = 0.0
            dz = 1.0
        dC = 0.5 * S * dx
        dS = Sp * dx
        da = a * (-dz * d - zeta * dd)
        dD = dC + (dz * d + zeta * dd) * S + zeta * d * dS
        J[0, j] = 2.0 * (da * C + a * dC)
        J[1, j] = -2.0 * a * da
        J[2, j] = -K * (da * D + a * dD)
        J[3, j] = -K * (J[0, j] + J[1, j]) - J[2, j]
    return coeff, J


@njit(cache=True)
def objective(theta, moments):
    """Evaluate the component-wise expected quadratic transition loss and gradient."""
    co, J = zoh_jac(theta)
    v = np.array([1.0, -co[0], -co[1], -co[2], -co[3], -theta[3]])
    q = moments @ v
    grad = np.zeros(4)
    for j in range(4):
        for k in range(4):
            grad[j] -= 2.0 * J[k, j] * q[k + 1]
    grad[3] -= 2.0 * q[5]
    return np.dot(v, q), grad


@njit(cache=True)
def coefficients(theta):
    """Map an array of component parameter vectors (d, 4) to ZOH coefficients (d, 4)."""
    co = np.empty((len(theta), 4))
    for i in range(len(theta)):
        co[i], _ = zoh_jac(theta[i])
    return co


def step_filter(mu, P, co, theta, u, uprev, y, config, reset):
    """Execute one step of Kalman filtering and one-lag RTS smoothing."""
    nv = len(theta)
    dim = 2 * nv
    if reset:
        pred = np.zeros(dim)
        Ppred = np.eye(dim) * config["reset_cov"]
        cross_prior = None
    else:
        A = np.zeros((dim, dim))
        ix = np.arange(nv) * 2
        A[ix, ix] = co[:, 0]
        A[ix, ix + 1] = co[:, 1]
        A[ix + 1, ix] = 1.0
        pred = A @ mu
        pred[::2] += co[:, 2] * u + co[:, 3] * uprev + theta[:, 3]
        cross_prior = P @ A.T
        Ppred = A @ cross_prior
        Ppred[ix, ix] += config["omega2"]
    pc = Ppred[:, ::2].sum(axis=1)
    innovation_var = pc[::2].sum() + config["sigma2"]
    if not np.isfinite(innovation_var) or innovation_var <= 0.0:
        raise FloatingPointError("Invalid innovation covariance")
    error = y - pred[::2].sum()
    updated = pred + pc * (error / innovation_var)
    Pnew = Ppred - np.outer(pc, pc) / innovation_var
    Pnew = 0.5 * (Pnew + Pnew.T)
    if reset:
        return updated, Pnew, None
    old_c = cross_prior[:, ::2].sum(axis=1)
    old_mu = mu + old_c * (error / innovation_var)
    old_P = P - np.outer(old_c, old_c) / innovation_var
    cross = cross_prior.T - np.outer(pc, old_c) / innovation_var
    moments = np.empty((nv, 6, 6))
    for i in range(nv):
        a = 2 * i
        b = a + 1
        v = np.array([updated[a], updated[b], old_mu[b], u[i], uprev[i], 1.0])
        M = np.outer(v, v)
        M[0, 0] += Pnew[a, a]
        M[0, 1] += Pnew[a, b]
        M[1, 0] += Pnew[b, a]
        M[1, 1] += Pnew[b, b]
        M[2, 2] += old_P[b, b]
        M[0, 2] += cross[a, b]
        M[2, 0] += cross[a, b]
        M[1, 2] += cross[b, b]
        M[2, 1] += cross[b, b]
        moments[i] = M
    return updated, Pnew, moments


@njit(cache=True)
def _residual_vector(theta):
    co, _ = zoh_jac(theta)
    return np.array([1.0, -co[0], -co[1], -co[2], -co[3], -theta[3]])


@njit(cache=True)
def _centered_objective(theta, M, v0, q0):
    co, J = zoh_jac(theta)
    v = np.array([1.0, -co[0], -co[1], -co[2], -co[3], -theta[3]])
    delta = v - v0
    md = M @ delta
    q = q0 + md
    value = 2.0 * np.dot(delta, q0) + np.dot(delta, md)
    gradient = np.zeros(4)
    for j in range(4):
        for k in range(4):
            gradient[j] -= 2.0 * J[k, j] * q[k + 1]
    gradient[3] -= 2.0 * q[5]
    return value, gradient


@njit(cache=True)
def _profile_step(shape, M, bounds, previous, v0, q0):
    """Profile out bounded linear parameters (K_i, c_i) using an algebraically centered objective."""
    tau, zeta = np.exp(shape[0]), np.exp(shape[1])
    unit = np.array([1.0, tau, zeta, 0.0])
    co, _ = zoh_jac(unit)
    e = np.array([0.0, 0.0, 0.0, co[2], co[3], 0.0])
    f = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    base = np.array([previous[0], tau, zeta, previous[3]])
    delta = _residual_vector(base) - v0
    qb = q0 + M @ delta
    H00 = e @ M @ e
    H01 = e @ M @ f
    H11 = M[5, 5]
    r0 = e @ qb
    r1 = f @ qb
    kl, ku = bounds[0] - previous[0]
    cl, cu = bounds[3] - previous[3]
    candidates = np.empty((6, 2))
    candidates[0] = np.array([0.0, min(cu, max(cl, r1 / max(H11, 1e-30)))])
    det = H00 * H11 - H01 * H01
    if det > 1e-18 * max(H00 * H11, 1e-30):
        candidates[1] = np.array([
            min(ku, max(kl, (r0 * H11 - r1 * H01) / det)),
            min(cu, max(cl, (r1 * H00 - r0 * H01) / det)),
        ])
    else:
        candidates[1] = candidates[0]
    candidates[2] = np.array([kl, min(cu, max(cl, (r1 - H01 * kl) / max(H11, 1e-30)))])
    candidates[3] = np.array([ku, min(cu, max(cl, (r1 - H01 * ku) / max(H11, 1e-30)))])
    candidates[4] = np.array([min(ku, max(kl, (r0 - H01 * cl) / max(H00, 1e-30))), cl])
    candidates[5] = np.array([min(ku, max(kl, (r0 - H01 * cu) / max(H00, 1e-30))), cu])
    best = np.inf
    chosen = previous.copy()
    gradient = np.zeros(4)
    for i in range(6):
        theta = np.array([previous[0] + candidates[i, 0], tau, zeta, previous[3] + candidates[i, 1]])
        val, g = _centered_objective(theta, M, v0, q0)
        if val < best:
            best = val
            chosen = theta
            gradient = g
    return best, np.array([gradient[1] * tau, gradient[2] * zeta]), chosen


def optimize_components(theta, moments, config, profile=False):
    """Solve the bound-constrained component-wise M-step via L-BFGS-B."""
    bounds = [
        config["K_bounds"],
        config["tau_bounds"],
        config["zeta_bounds"],
        config["c_bounds"],
    ]
    failures = 0
    iterations = 0
    if not profile:
        for i in range(len(theta)):
            before, _ = objective(theta[i], moments[i])
            result = minimize(
                objective,
                theta[i],
                args=(moments[i],),
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={k: config[k] for k in ("maxiter", "ftol", "gtol")},
            )
            failures += not result.success
            iterations += int(result.nit)
            if result.fun <= before + 1e-12:
                theta[i] = result.x
        return int(failures), int(iterations)

    b_arr = np.array(bounds, dtype=float)
    shape_bounds = np.log(b_arr[1:3])
    factor = 1.0 / config["omega2"]
    for i in range(len(theta)):
        prev = theta[i].copy()
        v0 = _residual_vector(prev)
        q0 = np.asarray(moments[i].astype(np.longdouble) @ v0.astype(np.longdouble), dtype=float)

        def fun(v):
            val, grad, _ = _profile_step(v, moments[i], b_arr, prev, v0, q0)
            return val * factor, grad * factor

        res = minimize(
            fun,
            np.log(prev[1:3]),
            jac=True,
            bounds=shape_bounds,
            method="L-BFGS-B",
            options={k: config[k] for k in ("maxiter", "ftol", "gtol")},
        )
        failures += not res.success
        iterations += int(res.nit)
        val, _, chosen = _profile_step(res.x, moments[i], b_arr, prev, v0, q0)
        if not np.isfinite(val) or not np.isfinite(chosen).all():
            raise FloatingPointError("Nonfinite centered profiled objective")
        if val <= 0.0:
            theta[i] = chosen
    return int(failures), int(iterations)


@njit(cache=True)
def forecast_held(mu, co, theta, u_origin, horizons):
    """Multi-step forecast holding the input fixed at its prediction-origin value."""
    out = np.empty(len(horizons))
    a = mu[::2].copy()
    b = mu[1::2].copy()
    j = 0
    force = (co[:, 2] + co[:, 3]) * u_origin + theta[:, 3]
    for h in range(1, horizons[-1] + 1):
        nxt = co[:, 0] * a + co[:, 1] * b + force
        b = a
        a = nxt
        if h == horizons[j]:
            out[j] = a.sum()
            j += 1
            if j == len(horizons):
                break
    return out


@njit(cache=True)
def forecast_conditional(mu, co, theta, x, anchor_x, origin, horizons):
    """Multi-step conditional response forecast driven by exogenous input trajectory x."""
    out = np.full(len(horizons), np.nan)
    a = mu[::2].copy()
    b = mu[1::2].copy()
    j = 0
    for h in range(1, min(horizons[-1], len(x) - origin - 1) + 1):
        u = x[origin + h] - anchor_x
        up = x[origin + h - 1] - anchor_x
        nxt = co[:, 0] * a + co[:, 1] * b + co[:, 2] * u + co[:, 3] * up + theta[:, 3]
        b = a
        a = nxt
        if h == horizons[j]:
            out[j] = a.sum()
            j += 1
            if j == len(horizons):
                break
    return out


def reference_arrays(x, y, refs, width=1):
    """Construct reference-centered input and output arrays from confirmed reference indices."""
    n = len(y)
    xb = np.zeros_like(x)
    yb = np.zeros_like(y)
    reset = np.zeros(n, np.bool_)
    starts = sorted(set([0] + [int(t) for t in refs if t < n]))
    ends = starts[1:] + [n]
    for s, e in zip(starts, ends):
        left = max(0, s - width + 1)
        xb[s:e] = x[left : s + 1].mean(0)
        yb[s:e] = y[left : s + 1].mean()
        reset[s] = True
    u = x - xb
    up = np.vstack([x[0], x[:-1]]) - xb
    return (
        np.ascontiguousarray(u),
        np.ascontiguousarray(up),
        np.ascontiguousarray(y - yb),
        yb,
        reset,
    )


@njit(cache=True)
def fixed_pass(u, up, y, reset, theta, omega, sigma, initial_cov, reset_cov):
    """Run one full pass of Kalman filtering and one-step fixed-lag smoothing over a training sequence."""
    n, d = u.shape
    dim = 2 * d
    co = coefficients(theta)
    A = np.zeros((dim, dim))
    Q = np.zeros((dim, dim))
    for i in range(d):
        A[2 * i, 2 * i] = co[i, 0]
        A[2 * i, 2 * i + 1] = co[i, 1]
        A[2 * i + 1, 2 * i] = 1.0
        Q[2 * i, 2 * i] = omega
    mu = np.zeros(dim)
    P = np.eye(dim) * initial_cov
    M = np.zeros((d, 6, 6))
    count = 0
    xs = np.empty((n, dim))
    nll = 0.0
    for t in range(n):
        oldmu = mu.copy()
        oldP = P.copy()
        if reset[t]:
            prior = np.zeros(dim)
            cov = np.eye(dim) * (initial_cov if t == 0 else reset_cov)
            cross_prior = np.zeros((dim, dim))
        else:
            prior = A @ mu
            prior[::2] += co[:, 2] * u[t] + co[:, 3] * up[t] + theta[:, 3]
            cross_prior = P @ A.T
            cov = A @ cross_prior + Q
        pc = np.zeros(dim)
        for i in range(dim):
            for j in range(d):
                pc[i] += cov[i, 2 * j]
        variance = pc[::2].sum() + sigma
        error = y[t] - prior[::2].sum()
        mu = prior + pc * (error / variance)
        P = cov - np.outer(pc, pc) / variance
        P = (P + P.T) * 0.5
        xs[t] = mu
        nll += np.log(variance) + error * error / variance
        if reset[t]:
            continue
        oc = np.zeros(dim)
        for i in range(dim):
            for j in range(d):
                oc[i] += cross_prior[i, 2 * j]
        smu = oldmu + oc * (error / variance)
        sP = oldP - np.outer(oc, oc) / variance
        cross = cross_prior.T - np.outer(pc, oc) / variance
        for i in range(d):
            a = 2 * i
            b = a + 1
            v = np.array([mu[a], mu[b], smu[b], u[t, i], up[t, i], 1.0])
            moment = np.outer(v, v)
            moment[0, 0] += P[a, a]
            moment[0, 1] += P[a, b]
            moment[1, 0] += P[b, a]
            moment[1, 1] += P[b, b]
            moment[2, 2] += sP[b, b]
            moment[0, 2] += cross[a, b]
            moment[2, 0] += cross[a, b]
            moment[1, 2] += cross[b, b]
            moment[2, 1] += cross[b, b]
            M[i] += moment
        count += 1
    return xs, P, M / max(1, count), nll / n
