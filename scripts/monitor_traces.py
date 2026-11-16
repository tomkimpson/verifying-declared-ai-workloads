#!/usr/bin/env python3
"""Task 14.1: monitor traces behind F8 (descriptive; amendment 2026-11-13f).

Three configurations of the one-class model s_R run as the H3 monitor over
the test day:

- `full`: R1 as frozen (fit on development decode, prefill and idle; tau_alpha
  from results/heldout/thresholds.json);
- `loud`: the decode-only fit of T4, calibrated on calibration decode alone;
- `silent`: the same decode-only fit, calibrated on the pooled calibration
  declared runs (decode and idle).

The fits, thresholds and monitor rule are those of scripts/h3_episodes.py,
scripts/h3_false_alarms.py and scripts/t4_robustness.py, which already scored
these test windows. Before any trace is written, this script asserts that it
reproduces their committed thresholds, alarm onsets per background and episode
class counts. It adds no number: the traces are descriptive.

The excerpts are fixed by the amendment, from design metadata only (EXCERPTS).
The envelope is the 2 ms RMS of the AC-coupled current in amperes, summarised
per ENV_BIN_S; it is current, never power. Reads the archive (SSD) for the
envelope and the episode step events.

Writes results/monitor/traces.json.
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
import declaration as d  # noqa: E402
import groups  # noqa: E402
import heldout as h  # noqa: E402
import monitor as m  # noqa: E402
import prereg  # noqa: E402

OUT = REPO / "results" / "monitor" / "traces.json"
EPISODES = REPO / "results" / "monitor" / "episodes.json"
FALSE_ALARMS = REPO / "results" / "monitor" / "false_alarms.json"
T4 = REPO / "results" / "robustness" / "t4.json"
BACKGROUNDS = {"declared inference": ("serving-decode",), "idle": ("idle",),
               "non-AI busy": ("steady-gemm", "memory-bound")}
#: The decode-only declaration of T4 (pre-registration §10).
MISSPEC_DECLARED = ("serving-decode",)
#: Displayed excerpts (amendment 2026-11-13f): capture and seconds from the
#: start of its first scored interval.
EXCERPTS = (("idle", "panel-core-idle-r2", 120.0),
            ("declared decode", "panel-core-serving-decode-r2", 240.0),
            ("episodes", "panel-episodes-timeshared-0p5-slot120", 480.0))
#: Envelope summary bin, and the chunk the raw current is read in (its mean is
#: removed per chunk: the probe has no DC response, so a mean is offset).
ENV_BIN_S = 0.25
CHUNK_S = 30.0
ENV_QUANTILES = (0.05, 0.95)
SCORE_DECIMALS = 4
ENV_DECIMALS = 3


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def envelope(cap, t0: float, dur: float) -> dict:
    """2 ms RMS envelope (A) over [t0, t0 + dur], summarised per ENV_BIN_S as
    its mean and ENV_QUANTILES; times are bin centres from t0."""
    fs = 1.0 / cap.sample_interval_s
    env, rate = [], None
    for k in range(int(round(dur / CHUNK_S))):
        i0 = int(round((t0 + k * CHUNK_S) * fs))
        x = cap.amperes(i0, i0 + int(round(CHUNK_S * fs)))
        e, rate = corpus.rms_envelope(x - x.mean(), fs)
        env.append(e)
    env = np.concatenate(env)
    per = int(round(ENV_BIN_S * rate))
    env = env[:len(env) // per * per].reshape(-1, per)
    lo, hi = np.quantile(env, ENV_QUANTILES, axis=1)
    return {"t_s": ((np.arange(len(env)) + 0.5) * per / rate).round(3).tolist(),
            "mean_a": env.mean(1).round(ENV_DECIMALS).tolist(),
            "lo_a": lo.round(ENV_DECIMALS).tolist(), "hi_a": hi.round(ENV_DECIMALS).tolist()}


def main() -> None:
    if git("status", "--porcelain"):
        raise SystemExit("working tree is dirty (or has untracked files): commit first")
    thr = h.load_thresholds()
    assert thr["preregistration_sha256"] == prereg.PREREG_SHA256
    hdr, dev = d.load_windows("development", 5)
    _, cal = d.load_windows("calibration", 5)
    test_hdr, test1 = d.load_windows("test", 1)
    assert hdr["corpus_commit"] == test_hdr["corpus_commit"] == corpus.CORPUS_COMMIT
    det = h.fit_h3(dev, cal, thr)
    names = det["names"]
    f = h.frozen_r1()
    ms = d.OneClass(f["family"]).fit(d.matrix(d.select(dev, MISSPEC_DECLARED), names))
    cal_rows = h.calibration_rows(cal)
    decl_cal = [r for r in cal_rows if r["condition"] in MISSPEC_DECLARED]

    committed_ep = json.loads(EPISODES.read_text())
    committed_fa = json.loads(FALSE_ALARMS.read_text())
    t4 = json.loads(T4.read_text())
    t4m = t4["misspecified_declaration"]

    #: configuration -> (model, tau_alpha, committed source of its counts)
    configs = {
        "full": (det["r1"], det["thresholds"]["r1"]),
        "loud": (ms, d.threshold(ms.score(d.matrix(decl_cal, names)), h.ALPHA)),
        "silent": (ms, d.threshold(ms.score(d.matrix(cal_rows, names)), h.ALPHA)),
    }
    assert configs["full"][1] == thr["r1"]["threshold"]
    assert configs["loud"][1] == t4m["declared"]["threshold"]
    assert configs["silent"][1] == t4m["pooled"]["threshold"]
    committed = {
        "full": {"episodes": committed_ep["detectors"]["r1"]["counts"],
                 "false_alarms": {bg: {k: v["r1"][k] for k in ("onsets", "A", "exposure_h")}
                                  for bg, v in committed_fa["backgrounds"].items()},
                 "source": "results/monitor/episodes.json, results/monitor/false_alarms.json"},
        **{k: {"episodes": t4m[key]["h3"]["r1"]["episodes"]["counts"],
               "false_alarms": {bg: {q: v[q] for q in ("onsets", "A", "exposure_h")}
                                for bg, v in t4m[key]["h3"]["r1"]["false_alarms"].items()},
               "source": f"results/robustness/t4.json misspecified_declaration.{key}"}
           for k, key in (("loud", "declared"), ("silent", "pooled"))},
    }

    bg_rows = {bg: [r for r in test1 if r["condition"] in c] for bg, c in BACKGROUNDS.items()}
    runs = h.episode_runs(test1)

    # Whole-day reproduction, asserted before any trace is kept.
    seqs: dict[str, dict] = {}
    for k, (model, tau) in configs.items():
        seqs[k] = {}
        for bg, rows in bg_rows.items():
            fl = h.flags(model.score(d.matrix(rows, names)), tau)
            s = m.sequences(rows, fl)
            onsets = sum(int(o.sum()) for o, _ in s.values())
            a = sum(int(x.sum()) for _, x in s.values()) / sum(len(x) for _, x in s.values())
            want = committed[k]["false_alarms"][bg]
            assert onsets == want["onsets"] and a == want["A"], (k, bg, onsets, a, want)
        ep_flags = {rid: h.flags(model.score(d.matrix(v["rows"], names)), tau)
                    for rid, v in runs.items()}
        counts = m.evaluate_episodes(runs, ep_flags)["counts"]
        assert counts == committed[k]["episodes"], (k, counts, committed[k]["episodes"])
        print(f"{k}: tau {tau:.4f}; reproduces onsets, A and episode counts {counts}", flush=True)

    # Traces of the displayed excerpts.
    arc = corpus.archive()
    excerpts = []
    for name, cap_id, dur in EXCERPTS:
        cap = arc.open(cap_id)
        assert cap.entry["split"] == "test" and cap.entry["session"] == "2026-09-24", cap_id
        t0 = cap.scored_intervals()[0][0]
        rows = sorted((r for r in test1 if r["capture_id"] == cap_id), key=lambda r: r["t0_s"])
        t1 = np.array([r["t1_s"] for r in rows])
        keep = (t1 >= t0) & (t1 <= t0 + dur)
        ex = {"name": name, "capture_id": cap_id, "condition": rows[0]["condition"],
              "session": cap.entry["session"], "t0_s": float(t0), "duration_s": dur,
              "scored_intervals_s": [[a - t0, b - t0] for a, b in cap.scored_intervals()
                                     if a < t0 + dur],
              "window_end_s": (t1[keep] - t0).round(3).tolist(),
              "envelope": envelope(cap, t0, dur), "episodes": [], "configs": {}}
        if cap_id in runs:
            ex["episodes"] = [[e["first"] - t0, e["last"] - t0] for e in runs[cap_id]["eps"]
                              if e["first"] < t0 + dur]
        for k, (model, tau) in configs.items():
            s = model.score(d.matrix(rows, names))
            # The alarm state runs over the whole capture, then is cut to the excerpt.
            onset, active = m.alarm(h.flags(s, tau))
            alarms = [[a - t0, min(c, t0 + dur) - t0] for a, c in m.alarm_intervals(t1, onset, active)
                      if a <= t0 + dur]
            ex["configs"][k] = {"score": np.round(s[keep], SCORE_DECIMALS).tolist(),
                                "alarm_active": active[keep].astype(int).tolist(),
                                "alarms_s": alarms,
                                "frac_windows_flagged": float(h.flags(s[keep], tau).mean()),
                                "frac_time_in_alarm": float(active[keep].mean())}
        excerpts.append(ex)
        print(f"{name}: {cap_id}, {int(keep.sum())} windows, "
              + ", ".join(f"{k} A {v['frac_time_in_alarm']:.2f}" for k, v in ex["configs"].items()),
              flush=True)

    OUT.write_text(json.dumps({
        "generated_by": "scripts/monitor_traces.py", "spec_task": "14.1",
        "preregistered": False,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_commit": git("rev-parse", "HEAD"),
        "thresholds_repo_commit": thr["repo_commit"],
        "t4_repo_commit": t4["repo_commit"],
        "episodes_repo_commit": committed_ep["repo_commit"],
        "false_alarms_repo_commit": committed_fa["repo_commit"],
        "corpus_commit": corpus.CORPUS_COMMIT,
        "notes": ["descriptive (amendment 2026-11-13f): the fits, thresholds and monitor rule "
                  "of the pre-registered H3 run and of T4's decode-only rows, re-scored on "
                  "test-day windows those runs already scored",
                  "three configurations of s_R, never one monitor; loud and silent share one "
                  "decode-only fit and differ only in the threshold set",
                  "the episodes are time-shared training, not co-location",
                  "the excerpts are three separate captures of session 2026-09-24",
                  "committed_totals are copied from the committed sources and were reproduced "
                  "by this run (asserted); envelope: 2 ms RMS of the AC-coupled current in "
                  "amperes, mean removed per chunk; current, not power"],
        "rule": {"need": m.NEED, "of": m.OF, "rearm": m.REARM, "hop_s": 1, "window_s": 5},
        "fit_conditions": {"full": list(groups.DECLARED),
                           "loud": list(MISSPEC_DECLARED), "silent": list(MISSPEC_DECLARED)},
        "threshold_set_runs": {"full": sorted({r["run_id"] for r in cal_rows}),
                               "loud": sorted({r["run_id"] for r in decl_cal}),
                               "silent": sorted({r["run_id"] for r in cal_rows})},
        "thresholds": {k: tau for k, (_, tau) in configs.items()},
        "committed_totals": committed,
        "envelope": {"frame_s": corpus.FRAME_S, "bin_s": ENV_BIN_S,
                     "quantiles": list(ENV_QUANTILES), "chunk_s": CHUNK_S},
        "excerpts": excerpts,
    }, default=float) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
