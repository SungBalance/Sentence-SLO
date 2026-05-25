"""Micro-benchmark: numba JIT vs pure-Python for _compute_serve_defer_pair arithmetic.

Reproduces the exact math from:
  - RequestSLOState.multi_level_pressure_components (slo_state.py)
  - Scheduler._compute_serve_defer_pair (scheduler.py)

Input arrays (one row per admitted MEASURED request):
  remaining     : float64[N]  — expected_remaining_len() output
  ttd           : float64[N]  — time_to_deadline(now)  (may be NaN → None)
  tpot_s        : float64     — scalar TPOT estimate
  epoch_s       : float64     — scalar epoch time (NaN → None)
  n_admitted    : int64       — len(admitted)
  denom_cap     : int64       — max_num_running_reqs (or cap_n)

Output arrays:
  serve_pressure: float64[N]  — scaled serve pressure (inf → inf, NaN → None)
  defer_pressure: float64[N]  — scaled defer pressure

_PRESSURE_DENOM_EPSILON = 1e-9  (matches slo_state.py)
"""

import math
import time

import numpy as np
import numba  # noqa: F401 (checked for availability)
from numba import njit

_EPS = 1e-9

# ---------------------------------------------------------------------------
# Pure-Python reference — mirrors multi_level_pressure_components exactly
# ---------------------------------------------------------------------------

def py_pressure_pair(
    remaining: np.ndarray,   # float64[N]
    ttd: np.ndarray,         # float64[N]  (NaN = deadline unknown)
    tpot_s: float,
    epoch_s: float,          # NaN = None (serve-only mode)
    n_admitted: int,
    denom_cap: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Python reference. Returns (serve_out, defer_out) float64[N].

    NaN encodes "None" (unmeasurable). inf encodes past-deadline.
    The scale factor n_admitted / max(1, denom_cap) is applied after
    per-request pressure, matching _compute_serve_defer_pair.
    """
    N = len(remaining)
    serve_out = np.empty(N, dtype=np.float64)
    defer_out = np.empty(N, dtype=np.float64)
    scale = n_admitted / max(1, denom_cap)
    epoch_none = math.isnan(epoch_s)

    for i in range(N):
        rem = remaining[i]
        t = ttd[i]

        # NaN in ttd or remaining → unmeasurable
        if math.isnan(rem) or math.isnan(t):
            serve_out[i] = math.nan
            defer_out[i] = math.nan
            continue

        refill = rem * tpot_s

        if t <= 0.0:
            # Deadline already passed
            raw_serve = math.inf
            raw_defer = math.inf
        else:
            raw_serve = refill / max(t, _EPS)
            if epoch_none:
                raw_defer = raw_serve
            else:
                defer_buf = t - epoch_s
                if defer_buf <= 0.0:
                    raw_defer = math.inf
                else:
                    raw_defer = refill / max(defer_buf, _EPS)

        # Apply system-scale factor
        if math.isinf(raw_serve):
            serve_out[i] = math.inf
            defer_out[i] = math.inf
        else:
            serve_out[i] = raw_serve * scale
            defer_out[i] = raw_defer * scale if not math.isinf(raw_defer) else math.inf

    return serve_out, defer_out


# ---------------------------------------------------------------------------
# Numba JIT kernel — same arithmetic, numba-friendly
# ---------------------------------------------------------------------------

@njit(cache=True)
def nb_pressure_pair(
    remaining: np.ndarray,
    ttd: np.ndarray,
    tpot_s: float,
    epoch_s: float,
    n_admitted: np.int64,
    denom_cap: np.int64,
    serve_out: np.ndarray,
    defer_out: np.ndarray,
) -> None:
    """In-place numba kernel — writes into pre-allocated serve_out / defer_out."""
    N = remaining.shape[0]
    scale = n_admitted / (denom_cap if denom_cap > 0 else 1)
    epoch_none = math.isnan(epoch_s)

    for i in range(N):
        rem = remaining[i]
        t = ttd[i]

        if math.isnan(rem) or math.isnan(t):
            serve_out[i] = math.nan
            defer_out[i] = math.nan
            continue

        refill = rem * tpot_s

        if t <= 0.0:
            raw_serve = math.inf
            raw_defer = math.inf
        else:
            raw_serve = refill / (t if t > _EPS else _EPS)
            if epoch_none:
                raw_defer = raw_serve
            else:
                defer_buf = t - epoch_s
                if defer_buf <= 0.0:
                    raw_defer = math.inf
                else:
                    raw_defer = refill / (defer_buf if defer_buf > _EPS else _EPS)

        if math.isinf(raw_serve):
            serve_out[i] = math.inf
            defer_out[i] = math.inf
        else:
            serve_out[i] = raw_serve * scale
            if math.isinf(raw_defer):
                defer_out[i] = math.inf
            else:
                defer_out[i] = raw_defer * scale


def nb_pressure_pair_wrapper(
    remaining, ttd, tpot_s, epoch_s, n_admitted, denom_cap
):
    N = len(remaining)
    serve_out = np.empty(N, dtype=np.float64)
    defer_out = np.empty(N, dtype=np.float64)
    nb_pressure_pair(
        remaining, ttd,
        np.float64(tpot_s), np.float64(epoch_s),
        np.int64(n_admitted), np.int64(denom_cap),
        serve_out, defer_out,
    )
    return serve_out, defer_out


# ---------------------------------------------------------------------------
# Equality check
# ---------------------------------------------------------------------------

def _make_inputs(N, rng):
    remaining = rng.uniform(10.0, 200.0, size=N).astype(np.float64)
    ttd = rng.uniform(-0.05, 2.0, size=N).astype(np.float64)
    # 10% unmeasurable (NaN)
    mask = rng.random(N) < 0.1
    remaining[mask] = math.nan
    ttd[rng.random(N) < 0.05] = math.nan
    tpot_s = 0.025
    epoch_s = 0.030
    n_admitted = N
    denom_cap = 48
    return remaining, ttd, tpot_s, epoch_s, n_admitted, denom_cap


def assert_equal(N=32, seed=42):
    rng = np.random.default_rng(seed)
    remaining, ttd, tpot_s, epoch_s, n_admitted, denom_cap = _make_inputs(N, rng)
    s_py, d_py = py_pressure_pair(remaining, ttd, tpot_s, epoch_s, n_admitted, denom_cap)
    s_nb, d_nb = nb_pressure_pair_wrapper(remaining, ttd, tpot_s, epoch_s, n_admitted, denom_cap)

    # Compare element-by-element, handling NaN and inf
    for i in range(N):
        sp, sn = s_py[i], s_nb[i]
        dp, dn = d_py[i], d_nb[i]
        if math.isnan(sp):
            assert math.isnan(sn), f"serve[{i}]: py=NaN nb={sn}"
        elif math.isinf(sp):
            assert math.isinf(sn) and sp == sn, f"serve[{i}]: py={sp} nb={sn}"
        else:
            assert abs(sp - sn) < 1e-9, f"serve[{i}]: py={sp:.12g} nb={sn:.12g}"
        if math.isnan(dp):
            assert math.isnan(dn), f"defer[{i}]: py=NaN nb={dn}"
        elif math.isinf(dp):
            assert math.isinf(dn) and dp == dn, f"defer[{i}]: py={dp} nb={dn}"
        else:
            assert abs(dp - dn) < 1e-9, f"defer[{i}]: py={dp:.12g} nb={dn:.12g}"
    print(f"  Equality check PASSED for N={N}")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench(N_list=(8, 32, 64, 128, 256), n_iters=10_000, seed=7):
    rng = np.random.default_rng(seed)

    # Pre-allocate output arrays for numba (avoids alloc inside loop)
    # We'll vary N, so allocate per-N inside the loop.

    rows = []
    for N in N_list:
        remaining, ttd, tpot_s, epoch_s, n_admitted, denom_cap = _make_inputs(N, rng)
        serve_out = np.empty(N, dtype=np.float64)
        defer_out = np.empty(N, dtype=np.float64)

        # Numba warm-up (2 calls to trigger JIT compile)
        nb_pressure_pair(
            remaining, ttd,
            np.float64(tpot_s), np.float64(epoch_s),
            np.int64(n_admitted), np.int64(denom_cap),
            serve_out, defer_out,
        )
        nb_pressure_pair(
            remaining, ttd,
            np.float64(tpot_s), np.float64(epoch_s),
            np.int64(n_admitted), np.int64(denom_cap),
            serve_out, defer_out,
        )

        # Time Python reference
        t0 = time.perf_counter()
        for _ in range(n_iters):
            py_pressure_pair(remaining, ttd, tpot_s, epoch_s, n_admitted, denom_cap)
        py_us = (time.perf_counter() - t0) / n_iters * 1e6

        # Time numba JIT (in-place, pre-allocated)
        t0 = time.perf_counter()
        for _ in range(n_iters):
            nb_pressure_pair(
                remaining, ttd,
                np.float64(tpot_s), np.float64(epoch_s),
                np.int64(n_admitted), np.int64(denom_cap),
                serve_out, defer_out,
            )
        nb_us = (time.perf_counter() - t0) / n_iters * 1e6

        speedup = py_us / nb_us if nb_us > 0 else float("inf")
        rows.append((N, py_us, nb_us, speedup))

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Equality checks ===")
    for N in [8, 32, 64, 128]:
        assert_equal(N)

    print("\n=== Benchmark (10k iters each) ===")
    print(f"{'N':>6}  {'Python (µs)':>12}  {'Numba (µs)':>11}  {'Speedup':>8}")
    print("-" * 46)
    rows = bench()
    for N, py_us, nb_us, speedup in rows:
        print(f"{N:>6}  {py_us:>12.2f}  {nb_us:>11.2f}  {speedup:>7.1f}x")
