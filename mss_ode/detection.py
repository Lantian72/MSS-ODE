"""Causal rolling-window PELT segmentation and AR(1) / linear-trend confirmation (Algorithm 1)."""
import numpy as np
from scipy.stats import t as student_t

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator


@njit(cache=True)
def pelt(y, penalty, m):
    """Exact Pruned Exact Linear Time (PELT) segmentation under piecewise-constant SSE cost."""
    n = len(y)
    back = np.full(n + 1, -1, np.int64)
    if n < 2 * m:
        return back
    p = np.zeros(n + 1)
    p2 = np.zeros(n + 1)
    for i in range(n):
        v = y[i] - y[0]
        p[i + 1] = p[i] + v
        p2[i + 1] = p2[i] + v * v
    if p2[n] - p[n] * p[n] / n <= penalty:
        back[n] = 0
        return back
    F = np.full(n + 1, np.inf)
    F[0] = -penalty
    active = np.empty(n + 1, np.int64)
    expiry = np.full(n + 1, n + m + 1, np.int64)
    active[0] = 0
    count = 1
    for end in range(m, n + 1):
        add = end - m
        if add >= m:
            active[count] = add
            count += 1
        size = 0
        best = np.inf
        bestj = -1
        for k in range(count):
            j = active[k]
            if expiry[j] <= end:
                continue
            active[size] = j
            size += 1
            cost = p2[end] - p2[j] - (p[end] - p[j]) ** 2 / (end - j)
            value = F[j] + max(0.0, cost) + penalty
            if value < best:
                best = value
                bestj = j
        count = size
        F[end] = best
        back[end] = bestj
        for k in range(count):
            j = active[k]
            cost = p2[end] - p2[j] - (p[end] - p[j]) ** 2 / (end - j)
            if F[j] + max(0.0, cost) > best + 1e-10:
                expiry[j] = min(expiry[j], end + m)
    return back


@njit(cache=True)
def scan_chunk(y, W, penalty, m, begin, end, first_seen):
    """Run causal rolling-window PELT over [begin, end) with window length W."""
    for t in range(begin, end):
        start = max(0, t - W + 1)
        back = pelt(y[start : t + 1], penalty, m)
        k = back[len(back) - 1]
        while k > 0:
            g = start + k
            if first_seen[g] < 0:
                first_seen[g] = t
            k = back[k]
    return first_seen


def ar_pvalue(values):
    """Two-sided OLS t-test p-value for H_0: phi = 0 in an AR(1) model with intercept."""
    x = values[:-1]
    y = values[1:]
    xx = x - x.mean()
    yy = y - y.mean()
    denom = xx @ xx
    if len(x) <= 2 or denom <= 1e-20:
        return None
    phi = (xx @ yy) / denom
    residual = yy - phi * xx
    variance = (residual @ residual) / (len(x) - 2)
    if variance <= 1e-20:
        return None
    return float(2.0 * student_t.sf(abs(phi) / np.sqrt(variance / denom), len(x) - 2))


def trend_pvalue(values):
    """Two-sided OLS t-test p-value for H_0: zero linear slope over the confirmation window."""
    t = np.arange(len(values), dtype=float)
    t -= t.mean()
    centered = values - values.mean()
    slope = (t @ centered) / (t @ t)
    residual = centered - slope * t
    variance = (residual @ residual) / (len(values) - 2)
    if variance <= 1e-20:
        return None
    return float(2.0 * student_t.sf(abs(slope) / np.sqrt(variance / (t @ t)), len(values) - 2))


def confirm_references(x, cfg, rule="ar", min_activation=0, refresh_late_window=False):
    """Detect PELT candidates across all input channels and confirm steady-state references."""
    n, d = x.shape
    seen = np.full((d, n), -1, dtype=np.int64)
    for i in range(d):
        scan_chunk(np.ascontiguousarray(x[:, i]), cfg["W"], cfg["penalty"], cfg["min_size"], 0, n, seen[i])
    test_fn = ar_pvalue if rule == "ar" else trend_pvalue
    events = []
    counts = dict(candidates=0, accepted=0, rejected=0, invalid=0, pending=0)
    for i in range(d):
        for k in np.flatnonzero(seen[i] >= 0):
            first = int(seen[i, k])
            activation = max(first, int(k) + cfg["L"] - 1)
            if activation < min_activation:
                continue
            counts["candidates"] += 1
            if activation >= n:
                counts["pending"] += 1
                continue
            begin = (activation - cfg["L"] + 1) if refresh_late_window else int(k)
            p = test_fn(x[begin : begin + cfg["L"], i])
            if p is None:
                counts["invalid"] += 1
                continue
            if p <= cfg["alpha"]:
                counts["rejected"] += 1
                continue
            counts["accepted"] += 1
            events.append((activation, i, int(k), first, p))
    events.sort()
    refs = []
    last = -cfg["merge"] - 1
    for t, *_ in events:
        if t > last + cfg["merge"]:
            refs.append(t)
            last = t
    counts["updates"] = len(refs)
    counts["merged"] = len(events) - len(refs)
    return np.asarray(refs, dtype=np.int64), events, counts
