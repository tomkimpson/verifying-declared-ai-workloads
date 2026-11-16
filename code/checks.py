"""Spec §3 correctness checks 1, 2, 3 and 5, as functions the tests call.

Each returns a dict with the observed quantity and a boolean `ok`, so a test
can assert it passes on the real input and fails on a perturbed one.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import corpus
import detectors
import windows

REPO = Path(__file__).resolve().parents[1]

#: §3 check 1: amodo's committed two-stage macro-F1 (pinned file), and the tolerance.
REPLICATION_MACRO_F1 = json.loads(
    (corpus.CORPUS_REPO / "results/replication/classification.json").read_text()
)["two_stage"]["macro_f1"]
REPLICATION_TOL = 0.001
#: §3 check 2: relative tolerance on recomputed features.
FEATURE_RTOL = 1e-6
#: §3 check 5: the Phase-2 boundary tolerance, seconds.
EDGE_TOL_S = 1e-3
#: The sprint's committed result that check 3 reproduces.
SPRINT_MONITOR = REPO / "results" / "sprint" / "monitor.json"
SPRINT_FAMILIES = ("conditions", "core-panel")


def check1_replication(y=None) -> dict:
    """Two-stage forest, leave-one-group-out, on amodo `features.json`.
    `y` substitutes labels (the perturbation hook)."""
    _, x, y0, g, _ = detectors.replication_traces()
    f1 = detectors.leave_one_group_out_macro_f1(x, y0 if y is None else y, g)
    return {"macro_f1": f1, "target": REPLICATION_MACRO_F1,
            "ok": abs(f1 - REPLICATION_MACRO_F1) <= REPLICATION_TOL}


def check2_features(run_ids: list[str], offset_samples: int = 0) -> dict:
    """Features of whole replication traces recomputed through the archive
    loader and `windows.featurise`, against amodo `features.json`.
    `offset_samples` drops leading samples (the perturbation hook)."""
    names, x_ref, _, _, ids = detectors.replication_traces()
    ref = dict(zip(ids, x_ref))
    arc = corpus.archive()
    worst = {}
    for rid in run_ids:
        cap = arc.open(rid)
        feats, _ = windows.featurise(cap.amperes(offset_samples), 1.0 / cap.sample_interval_s)
        got = np.array([feats[n] for n in names])
        worst[rid] = float(np.max(np.abs(got - ref[rid]) /
                                  np.maximum(np.abs(ref[rid]), 1e-300)))
    m = max(worst.values())
    return {"n_traces": len(run_ids), "max_rel_dev": m, "rtol": FEATURE_RTOL,
            "per_trace": worst, "ok": m <= FEATURE_RTOL}


def replica_label_counts(rows: list[dict]) -> dict[str, dict[str, int]]:
    """Level-1 replica labels counted per condition, as the sprint tabulated them."""
    names, forest = detectors.fit_replica()
    labels = forest.predict(np.array([[r["features"][n] for n in names] for r in rows]))
    out = defaultdict(Counter)
    for r, lab in zip(rows, labels):
        out[r["condition"]][lab] += 1
    return {c: dict(v) for c, v in out.items()}


def check3_sprint(counts: dict[str, dict[str, int]]) -> dict:
    """Per-condition replica labels against the sprint's committed monitor.json."""
    ref = {c: t["rf_labels"] for c, t in
           json.loads(SPRINT_MONITOR.read_text())["per_condition"].items()}
    diff = {c: {"got": counts.get(c), "sprint": ref.get(c)}
            for c in set(ref) | set(counts) if counts.get(c) != ref.get(c)}
    return {"n_conditions": len(ref), "mismatched": diff, "ok": not diff}


def check5_episodes(shift_s: float = 0.0) -> dict:
    """Episode edges from archive `step` events against amodo
    `episodes-recovery.json`. `shift_s` moves every step event before the
    edges are derived (the perturbation hook).

    Every boundary in the file (first/last step edge of each episode) must
    equal ours within EDGE_TOL_S, and each run must yield the file's episode
    count. Expected counts come from the file, not from literals. The file's
    own recovery tally (reachable boundaries recovered) is amodo's result and
    is reported, not re-checked here.
    """
    rec = json.loads((corpus.CORPUS_REPO / "results/panel/episodes-recovery.json").read_text())
    arc = corpus.archive()
    n_eps, worst, n_bound, bad, count_bad = 0, 0.0, 0, [], []
    for run in rec["runs"]:
        cap = arc.open(run["run_id"])
        steps = [{**e, "scope_start_s": e["scope_start_s"] + shift_s,
                  "scope_end_s": e["scope_end_s"] + shift_s} for e in cap.events("step")]
        eps = windows.episode_edges(steps, cap.events("train-slot"))
        got = sum(e is not None for e in eps)
        n_eps += got
        if got != run["n_episodes"]:
            count_bad.append((run["run_id"], got, run["n_episodes"]))
        for b in run["boundaries"]:
            e = eps[b["episode_index"]]
            ours = e[0] if b["edge"] == "onset" else e[1]
            d = abs(ours - b["commanded_scope_s"])
            worst, n_bound = max(worst, d), n_bound + 1
            if d > EDGE_TOL_S:
                bad.append((run["run_id"], b["episode_index"], b["edge"], d))
    return {"n_runs": len(rec["runs"]), "n_episodes": n_eps,
            "n_boundaries": n_bound, "max_abs_edge_diff_s": worst,
            "amodo_reachable_recovered": rec["n_reachable"] - rec["n_missed"],
            "n_bad": len(bad), "count_mismatches": count_bad,
            "ok": not bad and not count_bad}


#: §3 check 6: a pure-training window from a session R1 never saw must be
#: flagged at >= this rate.
POSITIVE_CONTROL_MIN = 0.99


def check6_positive_control(family: str, alpha: float, amplitude: bool = False,
                            positives=None) -> dict:
    """R1 fitted on all dev declared windows, thresholded on calibration
    declared windows (spec D3), then

    * positive control: dev `training` windows (complete-step label) from
      sessions holding no dev declared run must be flagged at
      >= POSITIVE_CONTROL_MIN;
    * self-scored: the dev declared windows R1 was fitted on, flagged at the
      calibration threshold, must give a rate whose 95% run-level bootstrap
      interval contains alpha.

    `positives` substitutes the positive-control rows (the perturbation hook).
    """
    import declaration as d
    import groups

    _, dev = d.load_windows("development")
    _, cal = d.load_windows("calibration")
    names = detectors.replication_traces()[0]
    fit_rows = d.select(dev, groups.DECLARED)
    cal_rows = d.select(cal, groups.DECLARED)
    seen = {r["session"] for r in fit_rows}
    if positives is None:
        positives = [r for r in d.select(dev, ("training",))
                     if r["session"] not in seen and r["label"] == "training"]
    d.assert_not_hidden(fit_rows + cal_rows + positives)

    model = d.OneClass(family).fit(d.matrix(fit_rows, names, amplitude))
    thr = d.threshold(model.score(d.matrix(cal_rows, names, amplitude)), alpha)
    pos = model.score(d.matrix(positives, names, amplitude)) > thr
    own = model.score(d.matrix(fit_rows, names, amplitude)) > thr
    own_runs = d.by_run(fit_rows, own)
    lo, hi = d.run_bootstrap_ci(own_runs)
    pos_rate = float(pos.mean())
    return {"family": family, "alpha": alpha, "amplitude": amplitude,
            "threshold": thr,
            "positive_runs": sorted({r["run_id"] for r in positives}),
            "positive_sessions": sorted({r["session"] for r in positives}),
            "n_positive": len(positives), "positive_rate": pos_rate,
            "self_rate": float(own.mean()), "self_ci95": [lo, hi],
            "self_rate_per_run": {k: float(v.mean()) for k, v in own_runs.items()},
            "ok_positive": pos_rate >= POSITIVE_CONTROL_MIN,
            "ok_self": lo <= alpha <= hi,
            "ok": pos_rate >= POSITIVE_CONTROL_MIN and lo <= alpha <= hi}
