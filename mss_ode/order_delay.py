"""Exact ZOH discretization for dynamic orders q in {1, 2, 3} and explicit input delays (Appendix G)."""
import numpy as np
from .model import zoh_jac

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator


@njit(cache=True)
def exp_jac(G, dG):
    """Matrix exponential and directional derivatives via scaling and squaring."""
    size = G.shape[0]
    norm = np.max(np.sum(np.abs(G), axis=1))
    squarings = max(0, int(np.ceil(np.log2(max(1.0, norm / 0.5)))))
    A = G / (2.0**squarings)
    dA = dG / (2.0**squarings)
    F = np.eye(size)
    term = F.copy()
    dF = np.zeros_like(dG)
    dt = np.zeros_like(dG)
    for k in range(1, 27):
        previous = term.copy()
        term = previous @ A / k
        for j in range(len(dG)):
            dt[j] = (dt[j] @ A + previous @ dA[j]) / k
            dF[j] += dt[j]
        F += term
    for _ in range(squarings):
        previous = F.copy()
        for j in range(len(dG)):
            dF[j] = dF[j] @ previous + previous @ dF[j]
        F = previous @ previous
    return F, dF


@njit(cache=True)
def map_jac(theta, q):
    """Map continuous-time parameters to exact ZOH coefficients and Jacobian for order q in {1, 2, 3}."""
    if q == 2:
        return zoh_jac(theta)
    if q == 1:
        K, tau, c = theta
        a = np.exp(-1.0 / tau)
        co = np.array([a, K * (-np.expm1(-1.0 / tau))])
        J = np.zeros((2, 3))
        J[0, 1] = a / tau**2
        J[1, 0] = -np.expm1(-1.0 / tau)
        J[1, 1] = -K * J[0, 1]
        return co, J
    K, tau, zeta, lag, c = theta
    f = 1.0 / (tau * tau * lag)
    G = np.zeros((4, 4))
    G[0, 1] = 1.0
    G[1, 2] = 1.0
    G[2, 0] = -f
    G[2, 1] = -(2.0 * zeta / (tau * lag) + 1.0 / tau**2)
    G[2, 2] = -(1.0 / lag + 2.0 * zeta / tau)
    G[2, 3] = f
    dG = np.zeros((3, 4, 4))
    dG[0, 2, 0] = 2.0 * f / tau
    dG[0, 2, 1] = 2.0 * zeta / (tau * tau * lag) + 2.0 / tau**3
    dG[0, 2, 2] = 2.0 * zeta / tau**2
    dG[0, 2, 3] = -2.0 * f / tau
    dG[1, 2, 1] = -2.0 / (tau * lag)
    dG[1, 2, 2] = -2.0 / tau
    dG[2, 2, 0] = f / lag
    dG[2, 2, 1] = 2.0 * zeta / (tau * lag**2)
    dG[2, 2, 2] = 1.0 / lag**2
    dG[2, 2, 3] = -f / lag
    Z, dZ = exp_jac(G, dG)
    T = Z[:3, :3].copy()
    b = Z[:3, 3].copy()
    trace = np.trace(T)
    a = np.array([trace, -0.5 * (trace * trace - np.trace(T @ T)), np.exp(G[2, 2])])
    unit = np.empty(3)
    unit[0] = b[0]
    unit[1] = (T @ b)[0] - a[0] * unit[0]
    unit[2] = 1.0 - a.sum() - unit[0] - unit[1]
    co = np.concatenate((a, K * unit))
    J = np.zeros((6, 5))
    J[3:, 0] = unit
    for j in range(3):
        dT = dZ[j, :3, :3].copy()
        db = dZ[j, :3, 3].copy()
        da = np.array([np.trace(dT), -trace * np.trace(dT) + np.trace(T @ dT), a[2] * dG[j, 2, 2]])
        du = np.empty(3)
        du[0] = db[0]
        du[1] = (dT @ b + T @ db)[0] - da[0] * unit[0] - a[0] * du[0]
        du[2] = -da.sum() - du[0] - du[1]
        J[:3, j + 1] = da
        J[3:, j + 1] = K * du
    return co, J
