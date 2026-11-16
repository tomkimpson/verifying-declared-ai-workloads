#!/usr/bin/env python3
"""Task 5.4: T4 robustness (pre-registration §10). No variant replaces a headline.

Each variant changes one named parameter and reuses the 5.1-5.3 code paths
(`heldout.detection`, `heldout.false_positives`, `monitor.sequences`,
`monitor.false_alarms`, `monitor.evaluate_episodes`) unchanged:

* alpha in {0.01, 0.02, 0.05, 0.10}: thresholds from the alpha grid frozen in
  results/heldout/thresholds.json. H2 and H3 for R1, the classifier and OR.
* persistence 2-of-3, 3-of-5, 4-of-5, 5-of-5 (re-arm unchanged): H3.
* mis-specified declaration: R1 refitted on development `serving-decode`
  only, reported under two calibrations (amendment 2026-10-05b): the pooled
  calibration declared runs (pre-registration §3 read literally) and the
  declaration's own calibration run; development `serving-prefill` (and,
  for the second, `idle`) as stress; H2 and H3 for each. Phase 2's unseen-workload hold-outs are copied
  from results/declaration/heldout.json.
* CUSUM on R1's scores, sensitivity only (amendment 2026-10-05): k is R1's
  frozen threshold, h the (1 - alpha) quantile of S over the 1 s-hop
  calibration declared windows (reset per capture); the window flag becomes
  S_t > h, then the same 3-of-5 monitor. The classifier gets no CUSUM: its
  calibration scores are almost all 0, so a median-based k (or one at any
  quantile up to 0.95) is 0 and S never falls.

Writes results/robustness/t4.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

import corpus  # noqa: E402
import coverage as cv  # noqa: E402
import declaration as d  # noqa: E402
import groups  # noqa: E402
import heldout as h  # noqa: E402
import monitor as m  # noqa: E402
import prereg  # noqa: E402

OUT = REPO / "results" / "robustness" / "t4.json"
MANIFEST = REPO / "results" / "windows" / "manifest.json"
PHASE2 = REPO / "results" / "declaration" / "heldout.json"
BACKGROUNDS = {"declared inference": ("serving-decode",), "idle": ("idle",),
               "non-AI busy": ("steady-gemm", "memory-bound")}
PERSISTENCE = ((2, 3), (3, 5), (4, 5), (5, 5))
#: R1's mis-specified declaration: decode only (pre-registration §10).
MISSPEC_DECLARED = ("serving-decode",)
MISSPEC_STRESS = ("serving-prefill", "idle")
CUSUM_RULE = ("R1 only. S_t = max(0, S_{t-1} + s_t - k), reset per capture; k = R1's frozen "
              "threshold; h = (1 - alpha) quantile of S over 1 s-hop calibration declared "
              "windows; flag iff S_t > h; then the 3-of-5 monitor (amendment 2026-10-05)")


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def h3(bg_rows: dict, bg_flags: dict, exposure: dict, runs: dict, ep_flags: dict,
       need: int = m.NEED, of: int = m.OF) -> dict:
    """F and A per background, and episode cells and Lmin, for one flag set."""
    rule = {"need": need, "of": of}
    fa = {bg: m.false_alarms(m.sequences(bg_rows[bg], bg_flags[bg], **rule), exposure[bg])
          for bg in bg_rows}
    ep = m.evaluate_episodes(runs, ep_flags, **rule)
    return {"false_alarms": {bg: {k: v[k] for k in ("onsets", "exposure_h", "F_per_h",
                                                     "F_upper95_poisson_model", "A")}
                             for bg, v in fa.items()},
            "episodes": {"counts": ep["counts"], "Lmin_s": ep["Lmin_s"],
                         "pooled_median_delay_s": ep["pooled_median_delay_s"],
                         "cells": {rid: {"nominal_s": c["nominal_s"], "duty": c["duty"],
                                         "gap_flag_rate": c["gap_flag_rate"],
                                         "headline_metric": c["headline_metric"],
                                         "rate": c["headline"]["rate"],
                                         "lower": c["headline"]["lower"],
                                         "onset_only_rate": c["onset_only"]["rate"],
                                         "median_delay_s": c["delay"]["median_s"]}
                                   for rid, c in ep["cells"].items()}}}


def with_or(fl: dict) -> dict:
    return {**fl, "or": fl["r1"] | fl["classifier"]}


def h2_pooled(det: dict) -> dict:
    return {"pooled": det["all"]["pooled"]["detection"],
            "per_run": {k: v["detection"] for k, v in det["all"]["per_run"].items()}}


def cusum_flags(rows: list[dict], scores: np.ndarray, k: float, thr: float) -> np.ndarray:
    """S_t > h per window, the CUSUM reset at each capture's start."""
    out = np.zeros(len(rows), bool)
    for _, idx in m.by_capture_index(rows).items():
        out[idx] = m.cusum(scores[idx], k) > thr
    return out


def main() -> None:
    if git("status", "--porcelain"):
        raise SystemExit("working tree is dirty (or has untracked files): commit first")
    if not git("ls-files", str(h.THRESHOLDS.relative_to(REPO))):
        raise SystemExit("results/heldout/thresholds.json is not committed")
    thr = h.load_thresholds()
    assert thr["preregistration_sha256"] == prereg.PREREG_SHA256
    hdr, dev = d.load_windows("development", 5)
    _, cal = d.load_windows("calibration", 5)
    _, cal1 = d.load_windows("calibration", 1)
    _, test5 = d.load_windows("test", 5)
    _, test1 = d.load_windows("test", 1)
    assert hdr["corpus_commit"] == corpus.CORPUS_COMMIT
    det = h.fit_h3(dev, cal, thr)
    names = det["names"]
    manifest = {c["capture_id"]: c for c in json.loads(MANIFEST.read_text())["captures"]}

    # H3 inputs, scored once.
    bg_rows = {bg: [r for r in test1 if r["condition"] in c] for bg, c in BACKGROUNDS.items()}
    exposure = {bg: float(sum(manifest[c]["scored_s"] for c in {r["capture_id"] for r in rows}))
                for bg, rows in bg_rows.items()}
    runs = h.episode_runs(test1)
    bg_scores = {bg: h.h3_scores(det, rows) for bg, rows in bg_rows.items()}
    ep_scores = {rid: h.h3_scores(det, v["rows"]) for rid, v in runs.items()}

    def h3_at(t: dict, need=m.NEED, of=m.OF) -> dict:
        bgf = {bg: h.h3_flags(s, t) for bg, s in bg_scores.items()}
        epf = {rid: h.h3_flags(s, t) for rid, s in ep_scores.items()}
        return {k: h3(bg_rows, {bg: f[k] for bg, f in bgf.items()}, exposure, runs,
                      {rid: f[k] for rid, f in epf.items()}, need, of)
                for k in ("r1", "classifier", "or")}

    # H2 inputs (5 s hop, all splits), scored once.
    rows5 = dev + cal + test5
    by_group: dict[str, list[dict]] = {}
    for r in rows5:
        by_group.setdefault(groups.group(r["condition"]), []).append(r)
    test_decl = [r for r in test5 if r["condition"] in h.TEST_DECLARED]
    r1, _ = h.fit_r1(dev)
    r1_group = {g: r1.score(d.matrix(by_group[g], names)) for g in cv.STRATEGY_GROUPS}
    r1_decl = r1.score(d.matrix(test_decl, names))
    clf_group, clf_decl = {}, {}
    for fit in h.FITS:
        model, man = h.fit_classifier(dev, fit, names)
        assert man == thr["fit_manifests"][fit]
        clf_group[fit] = model.p_training(d.matrix(by_group[h.FITS[fit][0]], names))
        clf_decl[fit] = model.p_training(d.matrix(test_decl, names))

    # Alpha sweep.
    alpha = {}
    for a in map(str, d.ALPHAS):
        t_r1 = thr["r1"]["alpha_grid"][a]
        t_fit = {f: thr["classifier"]["fits"][f]["alpha_grid"][a] for f in h.FITS}
        h2 = {"r1": {g: h2_pooled(h.detection(by_group[g], h.flags(r1_group[g], t_r1)))
                     for g in cv.STRATEGY_GROUPS},
              "classifier": {f: h2_pooled(h.detection(by_group[h.FITS[f][0]],
                                                      h.flags(clf_group[f], t_fit[f])))
                             for f in h.FITS},
              "r1_test_fpr": {w: v["fpr"] for w, v in
                              h.false_positives(test_decl, h.flags(r1_decl, t_r1)).items()},
              "classifier_test_fpr": {
                  w: v["fpr"] for w, v in h.false_positives(
                      test_decl, h.flags(clf_decl[h.H2_HEADLINE_FIT],
                                         t_fit[h.H2_HEADLINE_FIT])).items()}}
        t3 = {"r1": t_r1, "classifier": t_fit[h.H3_FIT]}
        alpha[a] = {"thresholds": {"r1": t_r1, "classifier": t_fit}, "h2": h2, "h3": h3_at(t3)}
        print(f"alpha {a}: Z {h2['r1']['co-location']['pooled']:.3f} "
              f"X' {h2['classifier']['co-location']['pooled']:.3f}", flush=True)

    # Persistence.
    persistence = {f"{n}-of-{o}": h3_at(det["thresholds"], n, o) for n, o in PERSISTENCE}
    print("persistence done", flush=True)

    # Mis-specified declaration: R1 fitted on decode only, under two calibrations.
    f = h.frozen_r1()
    ms_fit = d.select(dev, MISSPEC_DECLARED)
    ms = d.OneClass(f["family"]).fit(d.matrix(ms_fit, names))
    cal_rows = h.calibration_rows(cal)
    ms_bg = {bg: ms.score(d.matrix(rows, names)) for bg, rows in bg_rows.items()}
    ms_ep = {rid: ms.score(d.matrix(v["rows"], names)) for rid, v in runs.items()}
    ms_grp = {g: ms.score(d.matrix(by_group[g], names)) for g in cv.STRATEGY_GROUPS}
    ms_decl = ms.score(d.matrix(test_decl, names))

    def misspec_at(t_ms: float, stress_conds: tuple[str, ...]) -> dict:
        tt = {"r1": t_ms, "classifier": det["thresholds"]["classifier"]}
        bgf = {bg: h.h3_flags({"r1": ms_bg[bg], "classifier": bg_scores[bg]["classifier"]}, tt)
               for bg in bg_rows}
        epf = {rid: h.h3_flags({"r1": ms_ep[rid], "classifier": ep_scores[rid]["classifier"]}, tt)
               for rid in runs}
        return {
            "threshold": t_ms,
            "dev_stress_fpr": {c: float(h.flags(ms.score(d.matrix(d.select(dev, (c,)), names)),
                                                t_ms).mean()) for c in stress_conds},
            "h2": {"r1": {g: h2_pooled(h.detection(by_group[g], h.flags(ms_grp[g], t_ms)))
                          for g in cv.STRATEGY_GROUPS},
                   "r1_test_fpr": {w: v["fpr"] for w, v in
                                   h.false_positives(test_decl, h.flags(ms_decl, t_ms)).items()}},
            "h3": {k: h3(bg_rows, {bg: fl[k] for bg, fl in bgf.items()}, exposure, runs,
                         {rid: fl[k] for rid, fl in epf.items()}) for k in ("r1", "or")},
        }

    phase2 = {str(e["alpha"]): {rid: v["fp_rate"] for rid, v in e["check4"]["per_run"].items()
                                if v["unseen_workload"]}
              for e in json.loads(PHASE2.read_text())["results"]
              if e["family"] == f["family"] and e["variant"] == "R1"}
    decl_cal = [r for r in cal_rows if r["condition"] in MISSPEC_DECLARED]
    misspec = {
        "fit_runs": sorted({r["run_id"] for r in ms_fit}),
        "note": ("Two calibrations (amendment 2026-10-05b). 'pooled' is pre-registration "
                 "§3's rule read literally (both calibration declared runs). Calibration "
                 "idle is undeclared under this declaration and scores near the fit's "
                 "ceiling (calibration_score_quantiles), so it sets the threshold above "
                 "every co-location and time-sharing window; its idle stress rate is "
                 "circular and not reported, and its test idle FPR shares that "
                 "circularity. 'declared' calibrates on the declaration's own calibration "
                 "run (serving-decode) only; under it idle is stress, not a false positive."),
        "pooled": misspec_at(d.threshold(ms.score(d.matrix(cal_rows, names)), h.ALPHA),
                             ("serving-prefill",)),
        "declared": misspec_at(d.threshold(ms.score(d.matrix(decl_cal, names)), h.ALPHA),
                               MISSPEC_STRESS),
        "calibration_runs": {"pooled": sorted({r["run_id"] for r in cal_rows}),
                             "declared": sorted({r["run_id"] for r in decl_cal})},
        "calibration_score_quantiles": {
            run: {q: float(np.quantile(ms.score(d.matrix(
                [r for r in cal_rows if r["run_id"] == run], names)), float(q)))
                  for q in ("0.05", "0.5", "0.95")}
            for run in h.CALIBRATION_RUNS},
        "phase2_unseen_workload_holdouts": phase2,
    }
    print(f"mis-specified: pooled {misspec['pooled']['threshold']:.4f} "
          f"{misspec['pooled']['dev_stress_fpr']}, declared {misspec['declared']['threshold']:.4f} "
          f"{misspec['declared']['dev_stress_fpr']}", flush=True)

    # CUSUM on R1 (amendment 2026-10-05).
    cal1_rows = h.calibration_rows(cal1)
    cal1_r1 = h.h3_scores(det, cal1_rows)["r1"]
    kk = det["thresholds"]["r1"]
    s_cal = np.concatenate([m.cusum(cal1_r1[idx], kk)
                            for idx in m.by_capture_index(cal1_rows).values()])
    hh = d.threshold(s_cal, h.ALPHA)
    cus_bg = {bg: cusum_flags(bg_rows[bg], bg_scores[bg]["r1"], kk, hh) for bg in bg_rows}
    cus_ep = {rid: cusum_flags(v["rows"], ep_scores[rid]["r1"], kk, hh) for rid, v in runs.items()}
    cus = {"rule": CUSUM_RULE, "k": kk, "h": hh,
           "calibration_frac_S_positive": float((s_cal > 0).mean()),
           "h3": {"r1": h3(bg_rows, cus_bg, exposure, runs, cus_ep)}}
    print(f"cusum: k {kk:.4f} h {hh:.4f}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "generated_by": "scripts/t4_robustness.py", "spec_task": "5.4",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_commit": git("rev-parse", "HEAD"),
        "thresholds_repo_commit": thr["repo_commit"],
        "corpus_commit": corpus.CORPUS_COMMIT,
        "preregistration_sha256": prereg.PREREG_SHA256,
        "note": ("sensitivity only; no variant replaces a headline number. CUSUM is R1 only: "
                 "the classifier's calibration scores are almost all 0, so its reference k "
                 "would be 0 and its CUSUM could never fall (amendment 2026-10-05)"),
        "alpha_sweep": alpha, "persistence": persistence,
        "misspecified_declaration": misspec, "cusum": cus,
    }, indent=1, default=float) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
