"""Cut corpus captures into fixed-length windows, featurise and label them.

Promoted from `scripts/sprint_windows.py` (spec task 1.1). Changes from the
sprint: a `hop_s` parameter; window starts computed as `s0 + k * hop_s`
rather than by accumulation; the complete-step labelling rule of spec §3;
the sprint's cadence-line statistics dropped (spec §7: cadence detectors are
out of scope).

Per window:

* the 25 amplitude-free replication features (amodo `code/features.py`, pinned
  in `corpus.py`), computed on the mean-removed raw 200 kS/s samples;
* `ac_rms_a`, the window's AC RMS in amperes. Gain is not comparable across
  sessions; it exists for R1a's within-capture normalisation (spec D3);
* labels: `n_complete_steps` (optimizer `step` events with both edges inside
  the window), `step_overlap` (any step overlapping it) and `train_frac`
  (fraction of the window covered by steps). `label` is ``"training"`` iff
  ``n_complete_steps >= 1``, ``"partial"`` if steps overlap but none is
  complete, else ``"none"``. Partial windows are excluded from per-window
  detection rates (spec §3).

Decision timestamps are window ends (`t1_s`; spec D4).

Units are amperes in, dimensionless or amperes out. Never watts.
"""

from __future__ import annotations

import math

import numpy as np

import corpus

WINDOW_S = 5.0
#: Corpus families windowed for the paper (spec task 1.4). Smoke, pilot,
#: channel and replication families are not.
FAMILIES = ("conditions", "core-panel", "episodes", "negative")


def window_starts(s0: float, s1: float, window_s: float = WINDOW_S,
                  hop_s: float = WINDOW_S) -> np.ndarray:
    """Start times of every whole window inside the scored interval [s0, s1]."""
    if s1 - s0 < window_s:
        return np.empty(0)
    n = math.floor((s1 - s0 - window_s) / hop_s) + 1
    return s0 + hop_s * np.arange(n)


def expected_count(s0: float, s1: float, window_s: float = WINDOW_S,
                   hop_s: float = WINDOW_S) -> int:
    """floor(duration / hop) - (window / hop - 1): the task 1.4 count check.

    Written independently of `window_starts` so that it checks it. Exact when
    window_s is a whole multiple of hop_s, which every pass here uses.
    """
    ratio = round(window_s / hop_s)
    return max(0, math.floor((s1 - s0) / hop_s) - (ratio - 1))


def step_intervals(cap) -> np.ndarray:
    ev = cap.events("step")
    return (np.array([(e["scope_start_s"], e["scope_end_s"]) for e in ev])
            if ev else np.empty((0, 2)))


def step_labels(steps: np.ndarray, t0: float, t1: float) -> dict:
    """Ground truth of one window from optimizer-step intervals (spec §3)."""
    if steps.size == 0:
        return {"n_complete_steps": 0, "step_overlap": False, "train_frac": 0.0,
                "label": "none"}
    complete = int(((steps[:, 0] >= t0) & (steps[:, 1] <= t1)).sum())
    lo = np.clip(steps[:, 0], t0, t1)
    hi = np.clip(steps[:, 1], t0, t1)
    covered = float(np.clip((hi - lo).sum() / (t1 - t0), 0.0, 1.0))
    overlap = bool(((steps[:, 0] < t1) & (steps[:, 1] > t0)).any())
    label = "training" if complete else ("partial" if overlap else "none")
    return {"n_complete_steps": complete, "step_overlap": overlap,
            "train_frac": covered, "label": label}


def featurise(x_amperes: np.ndarray, fs: float) -> tuple[dict, float]:
    """Features and AC RMS of one window of raw samples, mean removed.

    Mean removal matches the replication's `remove_dc=True` (its measured DC
    offset is the trace mean) to ~1e-15 relative on the features.
    """
    x = x_amperes - x_amperes.mean()
    return corpus.extract_features(x, fs), float(x.std())


def capture_windows(capture_id: str, window_s: float = WINDOW_S,
                    hop_s: float = WINDOW_S, *, archive=None,
                    intervals: list[int] | None = None) -> list[dict]:
    """Every window of one capture's scored intervals, featurised and labelled."""
    cap = (archive or corpus.archive()).open(capture_id)
    fs = 1.0 / cap.sample_interval_s
    n_win = int(round(window_s * fs))
    steps = step_intervals(cap)
    e = cap.entry
    rows = []
    for k, (s0, s1) in enumerate(cap.scored_intervals()):
        if intervals is not None and k not in intervals:
            continue
        for t in window_starts(s0, s1, window_s, hop_s):
            t = float(t)
            i0 = int(round(t * fs))
            feats, rms = featurise(cap.amperes(i0, i0 + n_win), fs)
            rows.append({
                "capture_id": capture_id, "run_id": e["run_id"],
                "condition": e["condition"], "family": e["family"],
                "split": e["split"], "session": e["session"], "interval": k,
                "t0_s": t, "t1_s": t + window_s,
                **step_labels(steps, t, t + window_s),
                "ac_rms_a": rms, "features": feats,
            })
    return rows


def usable_captures(families=FAMILIES, split: str | None = None) -> list[dict]:
    """Index entries to window: the requested families, failures excluded by
    `capture_dir` (the index already omits them; this is a second guard)."""
    failed = corpus.failed_capture_dirs()
    out = []
    for e in corpus.archive().captures(split=split):
        if e["family"] not in families:
            continue
        if corpus.capture_dir(e) in failed:
            raise RuntimeError(f"{e['capture_id']}: index lists a failed capture_dir")
        out.append(e)
    return sorted(out, key=lambda e: e["capture_id"])


def _interval_task(args) -> list[dict]:
    capture_id, k, window_s, hop_s = args
    return capture_windows(capture_id, window_s, hop_s, intervals=[k])


def window_many(capture_ids, window_s: float = WINDOW_S, hop_s: float = WINDOW_S,
                workers: int = 6):
    """Yield each capture's windows, in capture then time order, computed one
    scored interval per task across `workers` processes."""
    from multiprocessing import Pool

    arc = corpus.archive()
    tasks = [(cid, k, window_s, hop_s) for cid in capture_ids
             for k in range(len(arc.open(cid).scored_intervals()))]
    with Pool(workers) as pool:
        for rows in pool.imap(_interval_task, tasks):
            yield from rows


def episode_edges(steps: list[dict], trains: list[dict]) -> list[tuple[float, float] | None]:
    """Each `train-slot`'s training episode as (first step start, last step end)
    on the scope clock, or None for a slot that ran no steps. Takes a capture's
    `step` and `train-slot` events (`cap.events(kind)`).

    Same definition as amodo `scripts/measure_episodes.py::episode_edges`: an
    episode's onset and stop are its first and last optimizer step's edges, not
    the slot's own timestamps (amodo amendment 2026-09-25).
    """
    out = []
    for tr in trains:
        inside = [e for e in steps
                  if tr["scope_start_s"] <= e["scope_start_s"] <= tr["scope_end_s"]]
        out.append((inside[0]["scope_start_s"], inside[-1]["scope_end_s"])
                   if inside else None)
    return out
