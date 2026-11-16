#!/usr/bin/env python3
"""Task 12.3: T7, loud and silent failure of the signature check s_R (spec §6).

One row per way of building R1's reference. Each row says which declared
workloads are unlike the fit (declared, but absent from the fit data) and
whether they sit in the threshold set (the calibration windows behind
tau_alpha), then what the monitor does: co-location detection, time-shared
episodes missed, how often it flags the declared work unlike the fit, and its
alarm onsets on held-out decode and idle. The failure type follows from those
cells by the rule in `failure()`, not by hand.

Rows and sources (committed JSONs only; no archive, no windows):
- full declaration: results/heldout/h2.json, results/heldout/r1_scores.json,
  results/monitor/episodes.json, results/monitor/false_alarms.json;
- prefill held out of the fit: Phase 2's development hold-out,
  results/declaration/heldout.json (isolation forest, R1, alpha = 0.05). They
  scored declared windows only, so its co-location, episode and onset cells
  are in no committed artefact and show "--". The idle hold-out row is no
  longer shown: its detection was never scored (amendment 2026-11-16b);
- decode-only fit under the decode and the pooled threshold set: T4,
  results/robustness/t4.json `misspecified_declaration`.
Workloads per fit and per threshold set come from each source's run lists,
mapped to conditions by results/windows/manifest.json.

    python3 scripts/table_t7.py   # writes figures/t7.tex
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
SRC = {
    "h2": RES / "heldout" / "h2.json",
    "r1_scores": RES / "heldout" / "r1_scores.json",
    "episodes": RES / "monitor" / "episodes.json",
    "false_alarms": RES / "monitor" / "false_alarms.json",
    "heldout": RES / "declaration" / "heldout.json",
    "t4": RES / "robustness" / "t4.json",
    "manifest": RES / "windows" / "manifest.json",
}
OUT = REPO / "figures" / "t7.tex"
#: Declared workloads (spec D3), in table order, with short labels.
DECLARED = {"serving-decode": "decode", "serving-prefill": "prefill", "idle": "idle"}
#: Phase 2 hold-out entry behind R1 (the frozen family, variant and alpha).
HOLDOUT = {"family": "iforest", "variant": "R1", "alpha": 0.05}
#: A flag rate at or above this on declared work unlike the fit is a loud failure.
LOUD = 0.5
DASH = "--"


def failure(row: dict) -> str:
    """Loud: declared work unlike the fit is flagged. Silent: co-location or
    every episode goes undetected with no alarm on declared work. Quiet: no
    alarm on declared work, but detection was not measured."""
    flagged = [v for v in row["unlike_flagged"].values() if v is not None]
    if any(v >= LOUD for v in flagged):
        return "loud"
    if row["colocation"] is None:
        return r"quiet$^{\ast}$"
    blind = row["colocation"] == 0 or row["missed"] == row["scored"]
    quiet = all(v == 0 for v in row["onsets"].values())
    if blind and quiet:
        return "silent"
    return "none"


def workloads(runs: list[str], cond: dict[str, str]) -> list[str]:
    """Short labels of the declared workloads among these runs."""
    return [label for w, label in DECLARED.items() if w in {cond[r] for r in runs}]


def holdout_entry(h: dict) -> dict:
    (entry,) = [r for r in h["results"]
                if all(r[k] == v for k, v in HOLDOUT.items())]
    return entry


def scored(counts: dict) -> int:
    return sum(counts.values()) - counts["truncated"] - counts["unscorable"]


def rows(j: dict, cond: dict[str, str]) -> tuple[list[dict], list[str]]:
    trace: list[str] = []
    out: list[dict] = []

    # Full declaration: the pre-registered R1 (H2, H3).
    ep = j["episodes"]["detectors"]["r1"]["counts"]
    bg = j["false_alarms"]["backgrounds"]
    out.append({"name": "complete reference",
                "fit": workloads(j["r1_scores"]["fit_runs"]["r1"], cond),
                "tau_set": workloads(j["r1_scores"]["calibration_runs"], cond),
                "colocation": j["h2"]["headline"]["Z_r1"]["detection"],
                "missed": ep["missed"], "scored": scored(ep),
                "pre_alarmed": ep["pre-alarmed"],
                "unlike_flagged": {},
                "onsets": {"decode": bg["declared inference"]["r1"]["onsets"],
                           "idle": bg["idle"]["r1"]["onsets"]}})
    trace.append("full: fit r1_scores.fit_runs.r1; tau set r1_scores.calibration_runs; "
                 "co-location h2.headline.Z_r1.detection; episodes "
                 "episodes.detectors.r1.counts; onsets false_alarms.backgrounds"
                 ".{declared inference,idle}.r1.onsets")

    # Phase 2 hold-outs: one development declared run left out of the fit.
    h = j["heldout"]
    sel = ",".join(f"{k}={v}" for k, v in HOLDOUT.items())
    per_run = holdout_entry(h)["check4"]["per_run"]
    for run in ("panel-serving-prefill",):
        w = cond[run]
        fit = [r for r in h["fit_runs"] if r != run]
        out.append({"name": f"{DECLARED[w]} held out of the fit",
                    "fit": workloads(fit, cond),
                    "tau_set": workloads(h["calibration_runs"], cond),
                    "colocation": None, "missed": None, "scored": None, "pre_alarmed": None,
                    "unlike_flagged": {DECLARED[w]: per_run[run]["fp_rate"]},
                    "onsets": {}})
        trace.append(f"{DECLARED[w]} held out: fit heldout.fit_runs minus {run}; tau set "
                     f"heldout.calibration_runs; flagged heldout.results[{sel}]"
                     f".check4.per_run.{run}.fp_rate; co-location, episodes, onsets: none")

    # T4: the decode-only declaration under its two calibrations.
    m = j["t4"]["misspecified_declaration"]
    for cal, label in (("declared", "decode"), ("pooled", "decode + idle")):
        v = m[cal]
        fa = v["h3"]["r1"]["false_alarms"]
        counts = v["h3"]["r1"]["episodes"]["counts"]
        fit = workloads(m["fit_runs"], cond)
        stress = v["dev_stress_fpr"]
        out.append({"name": f"decode-only fit, {label} threshold set",
                    "fit": fit, "tau_set": workloads(m["calibration_runs"][cal], cond),
                    "colocation": v["h2"]["r1"]["co-location"]["pooled"],
                    "missed": counts["missed"], "scored": scored(counts),
                    "pre_alarmed": counts["pre-alarmed"],
                    # Under pooled calibration idle sets the threshold, so T4
                    # omits its rate as circular (amendment 2026-10-05b): "--".
                    "unlike_flagged": {DECLARED[w]: rate for w, rate in stress.items()},
                    "onsets": {"decode": fa["declared inference"]["onsets"],
                               "idle": fa["idle"]["onsets"]}})
        t4m = "t4.misspecified_declaration"
        trace.append(f"decode only, {label}: fit {t4m}.fit_runs; tau set {t4m}.calibration_runs"
                     f".{cal}; co-location {t4m}.{cal}.h2.r1.co-location.pooled; episodes "
                     f"{t4m}.{cal}.h3.r1.episodes.counts; flagged {t4m}.{cal}.dev_stress_fpr; "
                     f"onsets {t4m}.{cal}.h3.r1.false_alarms.{{declared inference,idle}}.onsets")
    return out, trace


def episodes(r: dict) -> str:
    """Missed episodes; where most episodes start inside an alarm already on,
    say so instead, since a missed count of 0 would then overstate detection
    (the gap guard fails in all runs but one; appendix a:impl)."""
    if r["missed"] is None:
        return DASH
    if r["pre_alarmed"] > r["scored"] / 2:
        return f"{r['pre_alarmed']} of {r['scored']} start in alarm"
    return f"{r['missed']} of {r['scored']} missed"


def fmt_row(r: dict) -> str:
    unlike = [w for w in DECLARED.values() if w not in r["fit"]]
    in_tau = [w for w in unlike if w in r["tau_set"]]
    pct = lambda x: DASH if x is None else f"{100 * x:.1f}"
    flagged = ", ".join(f"{w} {pct(r['unlike_flagged'].get(w))}" for w in unlike) or DASH
    cells = [
        r["name"],
        ", ".join(unlike) or DASH,
        (", ".join(in_tau) or "none") if unlike else DASH,
        pct(r["colocation"]),
        episodes(r),
        flagged,
        " / ".join(str(r["onsets"][k]) for k in ("decode", "idle")) if r["onsets"] else DASH,
        failure(r),
    ]
    return " & ".join(cells) + r" \\"


def main() -> None:
    j = {k: json.loads(p.read_text()) for k, p in SRC.items()}
    cond = {c["run_id"]: c["condition"] for c in j["manifest"]["captures"]}
    table_rows, trace = rows(j, cond)
    lines = [r"\begin{tabular}{lllrrlrl}", r"\toprule",
             r"reference & unlike the fit & in $\tau$ set & co-location (\%) & "
             r"episodes & unlike-fit work flagged (\%) & onsets dec.\ / idle & "
             r"failure \\", r"\midrule"]
    lines += [fmt_row(r) for r in table_rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    commits = ", ".join(f"{k} {j[k]['repo_commit'][:7]}" for k in SRC)
    OUT.write_text("% generated by scripts/table_t7.py from (repo_commit of each source):\n"
                   f"% {commits}\n"
                   "% cell sources:\n" + "".join(f"%   {t}\n" for t in trace)
                   + "\n".join(lines) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
