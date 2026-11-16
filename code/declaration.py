"""R1, the declaration test (spec D3): a one-class model of declared windows.

Fitted on development declared windows (`groups.DECLARED`), thresholded at
the (1 - alpha) quantile of calibration declared windows, a window is flagged
when its anomaly score exceeds the threshold. Two families, chosen between
in Phase 2a on declared data alone:

* ``"mahalanobis"``: features standardised on the fit data, then squared
  Mahalanobis distance under a Ledoit-Wolf shrunk covariance. Shrinkage is
  needed, not tuned: the five `env_frac_*` and six `wav_frac_*` features each
  sum to ~1, so the sample covariance is near-singular (smallest eigenvalue
  ~2e-7 of the largest on dev declared windows).
* ``"iforest"``: an isolation forest on the raw features; the score is
  sklearn's anomaly score, ``-score_samples``, in (0, 1].

Both scores are positive and larger means less like the declaration.

R1a (spec D3) appends one amplitude feature: log of the window's AC RMS over
the median AC RMS of its own capture. Gain is not comparable across clamp
sessions (spec §3), so amplitude enters only as within-capture relative
change. The median is over the rows passed, so callers pass whole captures.

The firewall: Phase 2a (tasks 2.1-2.4) must score no hidden-training window.
`assert_not_hidden` is called on every row set 2a scores.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest

import groups
import prereg

REPO = Path(__file__).resolve().parents[1]
WINDOWS_DIR = REPO / "results" / "windows"

FAMILIES = ("mahalanobis", "iforest")
#: Fixed a priori, not tuned (spec §7: no detector optimisation).
IFOREST_KWARGS = {"n_estimators": 500, "random_state": 0, "n_jobs": -1}
#: The alpha grid for Phase 2. The R1 kill check (§8) is stated at alpha <= 0.05;
#: 0.10 is carried for the T4 alpha sweep.
ALPHAS = (0.01, 0.02, 0.05, 0.10)
#: Moving-block bootstrap block: 12 windows = 60 s at the 5 s hop, so a block
#: spans many decode requests and several optimizer steps.
BLOCK_WINDOWS = 12
N_BOOT = 2000
SEED = 0


def load_windows(split: str, hop_s: int = 5) -> tuple[dict, list[dict]]:
    """(header, rows) of a `scripts/make_windows.py` window file.

    The test split is locked behind the pre-registration (spec task 3.2).
    """
    if split == "test":
        prereg.require()
    path = WINDOWS_DIR / f"windows-hop{hop_s}s-{split}.jsonl"
    with path.open() as f:
        header = json.loads(next(f))
        rows = [json.loads(line) for line in f]
    assert header.get("header") and header["split"] == split, path
    return header, rows


def select(rows: list[dict], conditions=None, sessions=None) -> list[dict]:
    return [r for r in rows
            if (conditions is None or r["condition"] in conditions)
            and (sessions is None or r["session"] in sessions)]


def assert_not_hidden(rows: list[dict]) -> None:
    """Raise if any row is a hidden-training configuration (Phase 2a firewall)."""
    bad = sorted({r["run_id"] for r in rows if groups.is_hidden(r["condition"])})
    if bad:
        raise RuntimeError(f"Phase 2a may not score hidden-training runs: {bad}")


def matrix(rows: list[dict], names: list[str], amplitude: bool = False) -> np.ndarray:
    x = np.array([[r["features"][n] for n in names] for r in rows])
    if not amplitude:
        return x
    by_cap = defaultdict(list)
    for r in rows:
        by_cap[r["capture_id"]].append(r["ac_rms_a"])
    med = {c: float(np.median(v)) for c, v in by_cap.items()}
    amp = np.log([r["ac_rms_a"] / med[r["capture_id"]] for r in rows])
    return np.column_stack([x, amp])


class OneClass:
    def __init__(self, family: str):
        if family not in FAMILIES:
            raise ValueError(family)
        self.family = family

    def fit(self, x: np.ndarray) -> "OneClass":
        if self.family == "mahalanobis":
            self.mu, self.sd = x.mean(0), x.std(0)
            self.precision = LedoitWolf().fit((x - self.mu) / self.sd).precision_
        else:
            self.forest = IsolationForest(**IFOREST_KWARGS).fit(x)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        if self.family == "mahalanobis":
            z = (x - self.mu) / self.sd
            return np.einsum("ij,jk,ik->i", z, self.precision, z)
        return -self.forest.score_samples(x)


def threshold(cal_scores: np.ndarray, alpha: float) -> float:
    """The (1 - alpha) quantile of calibration declared scores (spec D3)."""
    return float(np.quantile(cal_scores, 1.0 - alpha))


def block_bootstrap_ci(flags: np.ndarray, block: int = BLOCK_WINDOWS,
                       n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    """95% circular block bootstrap interval of a flag rate within one run.

    Blocks of `block` consecutive windows, wrapping from the run's end to its
    start, are drawn from all n start positions until n windows are covered;
    the last block is truncated so each resample has exactly n windows. Every
    window is equally likely to be drawn, so the bootstrap is centred on the
    sample rate. (A non-circular version under-samples the first and last
    windows; PR #2 review.)
    """
    rates = block_bootstrap_rates(flags, block, n_boot, np.random.default_rng(seed))
    return float(np.quantile(rates, 0.025)), float(np.quantile(rates, 0.975))


def block_bootstrap_rates(flags: np.ndarray, block: int, n_boot: int,
                          rng: np.random.Generator) -> np.ndarray:
    """`n_boot` circular block bootstrap resamples of a within-run flag rate
    (the scheme of `block_bootstrap_ci`)."""
    flags = np.asarray(flags, float)
    n = len(flags)
    b = min(block, n)
    k = int(np.ceil(n / b))
    starts = rng.integers(0, n, size=(n_boot, k))
    idx = (starts[:, :, None] + np.arange(b)).reshape(n_boot, k * b)[:, :n] % n
    return flags[idx].mean(1)


def run_bootstrap_ci(flags_by_run: dict[str, np.ndarray], n_boot: int = N_BOOT,
                     seed: int = SEED) -> tuple[float, float]:
    """95% run-level bootstrap interval of a pooled flag rate: runs drawn with
    replacement, the rate pooled over the windows of the drawn runs."""
    runs = list(flags_by_run)
    hits = np.array([np.sum(flags_by_run[r]) for r in runs], float)
    n = np.array([len(flags_by_run[r]) for r in runs], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(runs), size=(n_boot, len(runs)))
    rates = hits[idx].sum(1) / n[idx].sum(1)
    return float(np.quantile(rates, 0.025)), float(np.quantile(rates, 0.975))


def by_run(rows: list[dict], values: np.ndarray) -> dict[str, np.ndarray]:
    """Values grouped by run, each in window time order."""
    out = defaultdict(list)
    for r, v in sorted(zip(rows, values), key=lambda p: (p[0]["run_id"], p[0]["t0_s"])):
        out[r["run_id"]].append(v)
    return {k: np.array(v) for k, v in out.items()}
