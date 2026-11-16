"""H1: leave-one-strategy-out coverage of the two-stage forest (spec task 4.1).

Level 1 is the published-recipe replica (`detectors.fit_replica`). Level 2 is
the same recipe fitted on corpus windows under the pre-registration §2.1 class
map: honest training plus every strategy group except the held-out one, from
all splits, plus negatives from development and calibration only. The flag is
`predict == "training"` (G&K format, no threshold).

Intervals follow pre-registration §6 and §8: the primary interval of a pooled
rate is fixed by its run count, never chosen.
"""

from __future__ import annotations

import numpy as np

import declaration as d
import groups
from detectors import NON_AI, TwoStageForest

#: Splits whose negatives may train a level-2 forest (spec D2).
NEGATIVE_FIT_SPLITS = ("development", "calibration")
#: Conditions whose `none` windows are inference (pre-registration §2.1).
INFERENCE_CONDITIONS = ("serving-decode", "serving-prefill")
NON_AI_CONDITIONS = ("idle", "steady-gemm", "memory-bound")
NON_AI_PREFIXES = ("periodic-gemm-",)
#: The groups H1 holds out in turn (spec D2).
STRATEGY_GROUPS = tuple(groups.HIDDEN_GROUPS)


def window_class(row: dict) -> str | None:
    """The fitting class of a window under pre-registration §2.1, or None if
    the window is not used in fitting."""
    label, cond = row["label"], row["condition"]
    if label == "training":
        return "training"
    if label != "none":
        return None
    if cond in INFERENCE_CONDITIONS or cond.startswith("mixture-"):
        return "inference"
    if cond in NON_AI_CONDITIONS or cond.startswith(NON_AI_PREFIXES):
        return NON_AI
    return None


def fit_rows(rows: list[dict], held_group: str | None,
             held_runs: frozenset[str] = frozenset()) -> list[dict]:
    """Level-2 training rows: everything outside `held_group` and `held_runs`,
    with negatives restricted to development and calibration."""
    out = []
    for r in rows:
        g = groups.group(r["condition"])
        if g == held_group or r["run_id"] in held_runs:
            continue
        if g == "negatives" and r["split"] not in NEGATIVE_FIT_SPLITS:
            continue
        if window_class(r) is not None:
            out.append(r)
    return out


def fit_level2(rows: list[dict], names: list[str]) -> TwoStageForest:
    y = np.array([window_class(r) for r in rows])
    return TwoStageForest().fit(d.matrix(rows, names), y)


def stratified_block_ci(flags_by_run: dict[str, np.ndarray],
                        block: int = d.BLOCK_WINDOWS, n_boot: int = d.N_BOOT,
                        seed: int = d.SEED) -> tuple[float, float]:
    """95% interval of a rate pooled over runs, by circular block bootstrap
    within each run (each run keeps its window count). Within-run variation
    only (pre-registration §8, the 2-run case)."""
    rng = np.random.default_rng(seed)
    hits = np.zeros(n_boot)
    total = 0
    for run in sorted(flags_by_run):
        f = np.asarray(flags_by_run[run], float)
        n = len(f)
        b = min(block, n)
        k = int(np.ceil(n / b))
        starts = rng.integers(0, n, size=(n_boot, k))
        idx = (starts[:, :, None] + np.arange(b)).reshape(n_boot, k * b)[:, :n] % n
        hits += f[idx].sum(1)
        total += n
    rates = hits / total
    return float(np.quantile(rates, 0.025)), float(np.quantile(rates, 0.975))


def pooled_interval(flags_by_run: dict[str, np.ndarray]) -> tuple[str, tuple[float, float]]:
    """(method, 95% interval) of a pooled rate, by run count (pre-registration §8)."""
    flags_by_run = {k: v for k, v in flags_by_run.items() if len(v)}
    n = len(flags_by_run)
    if n == 0:
        raise ValueError("no runs with windows")
    if n == 1:
        (f,) = flags_by_run.values()
        return "within-run block", d.block_bootstrap_ci(f)
    if n == 2:
        return "stratified block (within-run variation only)", stratified_block_ci(flags_by_run)
    return "run-level", d.run_bootstrap_ci(flags_by_run)


def summarise(rows: list[dict], pred: np.ndarray) -> dict:
    """Per-run and pooled detection on `training` windows of one scored set.

    `partial` windows are excluded from the rate and counted; the flag rate
    over all windows is given beside."""
    flags = pred == "training"
    per_run, tr_by_run = {}, {}
    for run, idx in sorted(_index_by_run(rows).items()):
        lab = np.array([rows[i]["label"] for i in idx])
        f = flags[idx]
        tr = f[lab == "training"]
        tr_by_run[run] = tr
        per_run[run] = {
            "condition": rows[idx[0]]["condition"], "split": rows[idx[0]]["split"],
            "session": rows[idx[0]]["session"], "n_windows": len(idx),
            "n_training": int(len(tr)), "n_partial": int((lab == "partial").sum()),
            "detection": float(tr.mean()) if len(tr) else None,
            "detection_ci95": list(d.block_bootstrap_ci(tr)) if len(tr) else None,
            "flag_rate_all": float(f.mean()),
            "predicted": {c: int((pred[idx] == c).sum()) for c in sorted(set(pred[idx]))},
        }
    tr_by_run = {k: v for k, v in tr_by_run.items() if len(v)}
    n_tr = sum(len(v) for v in tr_by_run.values())
    pooled = {"n_runs": len(tr_by_run), "n_training": n_tr,
              "detection": float(sum(v.sum() for v in tr_by_run.values()) / n_tr) if n_tr else None}
    if n_tr:
        pooled["interval_method"], ci = pooled_interval(tr_by_run)
        pooled["detection_ci95"] = list(ci)
    return {"pooled": pooled, "per_run": per_run}


def _index_by_run(rows: list[dict]) -> dict[str, list[int]]:
    """Row indices per run, each in window time order."""
    out: dict[str, list[int]] = {}
    for i in sorted(range(len(rows)), key=lambda i: (rows[i]["run_id"], rows[i]["t0_s"])):
        out.setdefault(rows[i]["run_id"], []).append(i)
    return out
