"""S2, the ceiling as tested: TOST on the per-window flag rate (spec Phase 6;
pre-registration §7).

The statistic is the difference in per-window flag rate between two runs, one
run per arm. Its interval comes from independent circular moving-block
bootstraps of each run (`declaration.block_bootstrap_rates`), each run with its
own seed, so it covers within-run variability only. Two one-sided tests at 5%
are equivalent to the 90% percentile interval lying inside (-delta, delta).
"""

from __future__ import annotations

import numpy as np

import declaration as d

#: Equivalence margin on the flag rate, the same for every detector (§7).
DELTA = 0.05
#: Two one-sided tests at this level: a (1 - 2 * TOST_ALPHA) interval.
TOST_ALPHA = 0.05
N_BOOT = 10_000
#: Seeds per run (§7): arm 1, discarded, copying, the positive-control run.
SEEDS = {"panel-training-arm1": 0, "panel-training-arm2": 1,
         "panel-training-arm3": 2, "panel-mixture-timeshared-0p5": 3}
#: Genuine training and the positive-control run (§7).
GENUINE = "panel-training-arm1"
CONTROL = "panel-mixture-timeshared-0p5"


def tost(flags_a: np.ndarray, flags_b: np.ndarray, seed_a: int, seed_b: int,
         delta: float = DELTA, block: int = d.BLOCK_WINDOWS,
         n_boot: int = N_BOOT) -> dict:
    """Rate(a) - rate(b), its 90% percentile interval, and the verdict:
    equivalent iff the interval lies strictly inside (-delta, delta)."""
    a, b = np.asarray(flags_a, float), np.asarray(flags_b, float)
    diff = d.block_bootstrap_rates(a, block, n_boot, np.random.default_rng(seed_a)) \
        - d.block_bootstrap_rates(b, block, n_boot, np.random.default_rng(seed_b))
    lo, hi = (float(np.quantile(diff, q)) for q in (TOST_ALPHA, 1 - TOST_ALPHA))
    return {"rate_a": float(a.mean()), "rate_b": float(b.mean()),
            "n_a": len(a), "n_b": len(b), "difference": float(a.mean() - b.mean()),
            "ci90": [lo, hi], "delta": delta,
            "equivalent": bool(-delta < lo and hi < delta)}
