"""The H3 monitor (spec D4; pre-registration §5).

A capture's 1 s-hop window flags, in time order with all its scored intervals
concatenated, drive a persistence alarm. The state carries across the breaks
between scored intervals and resets only at the start of a capture.

* Onset: the end time of the window that makes `need` of the last `of`
  windows positive while no alarm is active (3-of-5 headline).
* Re-arm: the alarm clears at the end time of the `rearm`-th consecutive
  negative window (5, every variant).

False alarms are onsets per monitored hour (F) and the fraction of windows
after which the alarm is active (A). Episodes are classed in the order of
pre-registration §5: truncated, unscorable, pre-alarmed, detected by onset,
missed.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import stats

#: Persistence and re-arm (spec D4).
NEED, OF, REARM = 3, 5, 5
#: Grace after an episode's last step edge, seconds (spec D4).
GRACE_S = 5.0
#: Episodes shorter than this fraction of their run's nominal duration are truncated.
TRUNCATED_FRAC = 0.9
#: Gap guard: pre-alarmed credit only if the gap-window flag rate is <= this.
GAP_GUARD = 0.20
#: Lmin's lower-bound level (pre-registration §5).
LMIN_LEVEL = 0.5
#: Onset blocks for F and A: 600 windows of the 1 s hop (pre-registration §8).
ONSET_BLOCK = 600
#: Episode bootstraps: blocks of 4 consecutive episodes.
EPISODE_BLOCK = 4
N_BOOT_INTERVAL = 2000
N_BOOT_BOUND = 10_000
#: D's interval needs at least this many detected-by-onset episodes.
MIN_FOR_D_INTERVAL = 8
SEED = 0
CONF = 0.95

EPISODE_CLASSES = ("truncated", "unscorable", "pre-alarmed", "detected", "missed")


def by_capture_index(rows: list[dict]) -> dict[str, np.ndarray]:
    """Row indices per capture, in window time order."""
    out = defaultdict(list)
    for i in sorted(range(len(rows)), key=lambda i: (rows[i]["capture_id"], rows[i]["t0_s"])):
        out[rows[i]["capture_id"]].append(i)
    return {k: np.array(v) for k, v in out.items()}


def alarm(flags, need: int = NEED, of: int = OF, rearm: int = REARM
          ) -> tuple[np.ndarray, np.ndarray]:
    """(onset, active) per window of one capture: `onset[i]` if window i raises
    an alarm, `active[i]` if the alarm is active after window i."""
    flags = np.asarray(flags, bool)
    onset = np.zeros(len(flags), bool)
    active = np.zeros(len(flags), bool)
    on, quiet = False, 0
    for i, f in enumerate(flags):
        if not on:
            if flags[max(0, i - of + 1):i + 1].sum() >= need:
                on, quiet, onset[i] = True, 0, True
        else:
            quiet = 0 if f else quiet + 1
            if quiet >= rearm:
                on = False
        active[i] = on
    return onset, active


def alarm_intervals(t1: np.ndarray, onset: np.ndarray, active: np.ndarray
                    ) -> list[tuple[float, float]]:
    """(onset time, clear time) of each alarm, both window end times; the
    clear time is +inf for an alarm still active at the capture's end."""
    out = []
    for i in np.flatnonzero(onset):
        j = i + 1
        while j < len(active) and active[j]:
            j += 1
        out.append((float(t1[i]), float(t1[j]) if j < len(active) else np.inf))
    return out


# --- False alarms (task 5.2) -------------------------------------------------

def garwood_upper(k: int, conf: float = CONF) -> float:
    """Exact one-sided upper Poisson bound on a count of k."""
    return float(stats.chi2.ppf(conf, 2 * (k + 1)) / 2)


def _circular_block_index(n: int, block: int, n_boot: int, rng) -> np.ndarray:
    b = min(block, n)
    k = int(np.ceil(n / b))
    starts = rng.integers(0, n, size=(n_boot, k))
    return (starts[:, :, None] + np.arange(b)).reshape(n_boot, k * b)[:, :n] % n


def false_alarms(seqs: dict[str, tuple[np.ndarray, np.ndarray]], exposure_s: float,
                 n_boot: int = N_BOOT_INTERVAL, seed: int = SEED) -> dict:
    """F and A over one background. `seqs` maps capture -> (onset, active).

    Intervals resample onset blocks within each capture (each keeps its
    window count), so exposure is fixed at E."""
    onsets = sum(int(o.sum()) for o, _ in seqs.values())
    n_win = sum(len(a) for _, a in seqs.values())
    alarmed = sum(int(a.sum()) for _, a in seqs.values())
    hours = exposure_s / 3600
    rng = np.random.default_rng(seed)
    b_on, b_al = np.zeros(n_boot), np.zeros(n_boot)
    for cap in sorted(seqs):
        o, a = seqs[cap]
        idx = _circular_block_index(len(a), ONSET_BLOCK, n_boot, rng)
        b_on += o[idx].sum(1)
        b_al += a[idx].sum(1)
    out = {
        "n_captures": len(seqs), "exposure_s": exposure_s, "exposure_h": hours,
        "n_windows": n_win, "onsets": onsets, "F_per_h": onsets / hours,
        "F_upper95_poisson_model": garwood_upper(onsets) / hours,
        "A": alarmed / n_win,
        "A_ci95_block_approx": [float(np.quantile(b_al / n_win, 0.025)),
                                float(np.quantile(b_al / n_win, 0.975))],
        "onsets_per_capture": {c: int(o.sum()) for c, (o, _) in seqs.items()},
        "A_per_capture": {c: float(a.mean()) for c, (_, a) in seqs.items()},
    }
    if onsets >= 10:
        out["F_ci95_block"] = [float(np.quantile(b_on / hours, 0.025)),
                               float(np.quantile(b_on / hours, 0.975))]
    return out


# --- Episodes (task 5.3) -------------------------------------------------------

def classify_episodes(episodes: list[dict], windows: list[dict],
                      onset: np.ndarray, active: np.ndarray, nominal_s: float) -> list[dict]:
    """Class and delay of each episode of one run.

    `episodes`: dicts with `first`, `last` (step edges, scope clock) and
    `steps` (an (n, 2) array of that episode's step intervals). `windows`: the
    run's 1 s-hop rows in time order, aligned with `onset` and `active`.
    """
    t0 = np.array([w["t0_s"] for w in windows])
    t1 = np.array([w["t1_s"] for w in windows])
    alarms = alarm_intervals(t1, onset, active)
    on_times = np.array([a for a, _ in alarms])
    out = []
    for e in episodes:
        first, last = e["first"], e["last"]
        dur = last - first
        steps = np.asarray(e["steps"])
        complete = bool(((steps[:, 0][:, None] >= t0) & (steps[:, 1][:, None] <= t1)).any())
        rec = {"first": first, "last": last, "duration_s": dur,
               "complete_step_window": complete, "delay_s": None}
        if dur < TRUNCATED_FRAC * nominal_s:
            rec["class"] = "truncated"
        elif not ((t0 < last) & (t1 > first)).any():
            rec["class"] = "unscorable"
        elif any(a <= first < c for a, c in alarms):
            rec["class"] = "pre-alarmed"
        else:
            hit = on_times[(on_times >= first) & (on_times <= last + GRACE_S)]
            if len(hit):
                rec["class"], rec["delay_s"] = "detected", float(hit.min() - first)
            else:
                rec["class"] = "missed"
        out.append(rec)
    return out


def success(eps: list[dict], metric: str) -> np.ndarray:
    """Per scored episode (pre-alarmed, detected or missed), in order: 1 if it
    counts as detected under `metric` ("P" credits pre-alarmed; "onset" does not)."""
    scored = [e for e in eps if e["class"] in ("pre-alarmed", "detected", "missed")]
    ok = ("pre-alarmed", "detected") if metric == "P" else ("detected",)
    return np.array([e["class"] in ok for e in scored], float)


def clopper_pearson_lower(k: int, n: int, conf: float = CONF) -> float:
    return 0.0 if k == 0 else float(stats.beta.ppf(1 - conf, k, n - k + 1))


def within_run_lower_bound(s: np.ndarray, n_boot: int = N_BOOT_BOUND, seed: int = SEED) -> dict:
    """min(Clopper-Pearson one-sided 95% lower bound, 5th percentile of a
    circular block bootstrap over consecutive episodes)."""
    n, k = len(s), int(s.sum())
    if n == 0:
        return {"n": 0, "k": 0, "rate": None, "lower": None}
    cp = clopper_pearson_lower(k, n)
    idx = _circular_block_index(n, EPISODE_BLOCK, n_boot, np.random.default_rng(seed))
    bb = float(np.quantile(s[idx].mean(1), 1 - CONF))
    return {"n": n, "k": k, "rate": k / n, "clopper_pearson": cp, "block_bootstrap_p5": bb,
            "lower": min(cp, bb)}


def delay_summary(eps: list[dict], seed: int = SEED) -> dict:
    """Median delay over detected-by-onset episodes, in episode order, with a
    circular block-bootstrap interval, or the range below 8 episodes."""
    d = np.array([e["delay_s"] for e in eps if e["class"] == "detected"])
    if len(d) == 0:
        return {"n": 0, "median_s": None}
    out = {"n": len(d), "median_s": float(np.median(d))}
    if len(d) >= MIN_FOR_D_INTERVAL:
        idx = _circular_block_index(len(d), EPISODE_BLOCK, N_BOOT_INTERVAL,
                                    np.random.default_rng(seed))
        med = np.median(d[idx], axis=1)
        out["ci95"] = [float(np.quantile(med, 0.025)), float(np.quantile(med, 0.975))]
    else:
        out["range_s"] = [float(d.min()), float(d.max())]
    return out


def cell_summary(eps: list[dict], gap_rate: float) -> dict:
    """Headline metric of one cell under the gap guard, with both metrics,
    the restricted metric, the within-run lower bound and D."""
    counts = {c: sum(e["class"] == c for e in eps) for c in EPISODE_CLASSES}
    headline = "P" if gap_rate <= GAP_GUARD else "onset"
    restricted = [e for e in eps if e["complete_step_window"]]
    return {
        "counts": counts, "n_episodes": len(eps), "gap_flag_rate": gap_rate,
        "headline_metric": headline,
        "P": within_run_lower_bound(success(eps, "P")),
        "onset_only": within_run_lower_bound(success(eps, "onset")),
        "headline": within_run_lower_bound(success(eps, headline)),
        "restricted_headline": within_run_lower_bound(success(restricted, headline)),
        "delay": delay_summary(eps),
    }


def lmin(cells: dict[float, float | None], level: float = LMIN_LEVEL) -> float | None:
    """Shortest nominal duration d such that every cell with nominal duration
    >= d has a headline lower bound >= level. `cells` maps each cell's nominal
    duration to its lower bound; a duration with several cells (duties) is
    passed as its smallest bound."""
    ds = sorted(cells)
    for d in ds:
        if all(cells[x] is not None and cells[x] >= level for x in ds if x >= d):
            return d
    return None


def gap_rates(runs: dict[str, dict], flags: dict[str, np.ndarray]) -> dict[str, float]:
    """Per-run flag rate on gap windows (label `none`). A run with none takes
    the pooled rate of the other runs (pre-registration §5)."""
    gap = {r: (float(flags[r][v["none"]].mean()) if v["none"].any() else None)
           for r, v in runs.items()}
    others = [r for r in runs if gap[r] is not None]
    pooled = float(sum(flags[r][runs[r]["none"]].sum() for r in others)
                   / sum(runs[r]["none"].sum() for r in others))
    return {r: pooled if g is None else g for r, g in gap.items()}


def evaluate_episodes(runs: dict[str, dict], flags: dict[str, np.ndarray],
                      need: int = NEED, of: int = OF, rearm: int = REARM) -> dict:
    """Episode classes, per-cell metrics and Lmin for one detector. `runs` as
    `heldout.episode_runs`; `flags` maps run -> 1 s-hop window flags."""
    gap = gap_rates(runs, flags)
    cells, all_eps = {}, []
    for rid, v in runs.items():
        onset, active = alarm(flags[rid], need, of, rearm)
        eps = classify_episodes(v["eps"], v["rows"], onset, active, v["nominal"])
        all_eps += eps
        cells[rid] = {"condition": v["condition"], "nominal_s": round(v["nominal"], 1),
                      "duty": v["duty"], "gap_from_pooled_other_runs": not v["none"].any(),
                      **cell_summary(eps, gap[rid]), "episodes": eps}
    bound: dict[float, float | None] = {}
    for c in cells.values():
        lb, d = c["headline"]["lower"], c["nominal_s"]
        prev = bound.get(d, np.inf)
        bound[d] = None if lb is None or prev is None else min(prev, lb)
    delays = [e["delay_s"] for e in all_eps if e["class"] == "detected"]
    return {"counts": {c: sum(e["class"] == c for e in all_eps) for c in EPISODE_CLASSES},
            "cells": cells,
            "lmin_bounds_by_duration": {str(x): y for x, y in sorted(bound.items())},
            "Lmin_s": lmin(bound),
            "pooled_median_delay_s": float(np.median(delays)) if delays else None}


def sequences(rows: list[dict], flags: np.ndarray, need: int = NEED, of: int = OF,
              rearm: int = REARM) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(onset, active) per capture of a background's rows."""
    return {cap: alarm(flags[idx], need, of, rearm)
            for cap, idx in by_capture_index(rows).items()}


# --- CUSUM (T4 sensitivity; amendment 2026-10-05) -------------------------------

def cusum(scores, k: float) -> np.ndarray:
    """One-sided CUSUM S_t = max(0, S_{t-1} + s_t - k), S_0 = 0."""
    s = np.zeros(len(scores))
    acc = 0.0
    for i, x in enumerate(scores):
        acc = max(0.0, acc + x - k)
        s[i] = acc
    return s
