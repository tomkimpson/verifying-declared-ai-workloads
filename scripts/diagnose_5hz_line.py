#!/usr/bin/env python3
"""Where the 5 Hz envelope line appears, and its exact period (spec task 1.5).

Development captures only (exploration never touches calibration or test),
plus the replication traces, which are all development. For each capture:

* `snr_hz_median`: per line frequency in LINES_HZ (5 Hz and its harmonics),
  the median over its first N_WINDOWS 5 s windows of the envelope line SNR:
  the largest bin within +/-0.3 Hz of that frequency over the median of a
  +/-1.5 Hz neighbourhood excluding +/-0.3 Hz (the sprint's definition);
* `period_ms`: the fold period in 195-205 ms that maximises the folded
  envelope's peak-to-peak over the first FOLD_S seconds, and that peak-to-peak
  as a fraction of the mean envelope;
* `telemetry`: whether an NVML sampler thread ran (it polls at 10 Hz);
* `scope_host_rate_ppm`: the fitted scope-to-host clock rate, so the period is
  known in host time to within a few ppm.

Also, from the development 5 s-hop windows of scripts/make_windows.py, the
fraction of each condition's windows whose envelope autocorrelation peak
(`env_ac_peak_lag_s`) falls in AC_LAG_BAND_S, i.e. at the line's period.

Replication traces are 5 s each: SNR only, pooled per class.

    python3 scripts/diagnose_5hz_line.py    # writes results/line5hz/line5hz.json

Units: amperes in; dimensionless and hertz out. Never watts.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
import corpus  # noqa: E402
import windows  # noqa: E402

OUT = REPO / "results" / "line5hz" / "line5hz.json"
SPLITS = ("development",)                  # exploration: dev only (CLAUDE.md)
N_WINDOWS = 30
FOLD_S = 60.0
FOLD_STEP_MS = 0.01
FOLD_PERIODS_MS = np.arange(195.0, 205.0 + FOLD_STEP_MS / 2, FOLD_STEP_MS)
FOLD_BINS = 40
FOLD_FRAME_S = 1e-3                           # fold envelope frame
LINES_HZ = (5.0, 10.0, 15.0, 25.0)
AC_LAG_BAND_S = (0.19, 0.21)
WINDOWS_DEV = REPO / "results" / "windows" / "windows-hop5s-development.jsonl"
SNR_HALF_WIDTH_HZ, SNR_GUARD_HZ, SEARCH_HZ = 1.5, 0.3, 0.3
NFFT = 16384


def line_snrs(x: np.ndarray, fs: float) -> dict[str, float]:
    """Envelope line SNR at each of LINES_HZ, keyed by frequency."""
    env, rate = corpus.rms_envelope(x, fs, frame_s=corpus.FRAME_S)
    e = env / env.mean() - 1.0
    p = np.abs(np.fft.rfft(e * np.hanning(e.size), n=NFFT)) ** 2
    f = np.fft.rfftfreq(NFFT, d=1.0 / rate)
    out = {}
    for f0 in LINES_HZ:
        near = np.abs(f - f0) <= SEARCH_HZ
        i = np.flatnonzero(near)[np.argmax(p[near])]
        d = np.abs(f - f[i])
        out[f"{f0:g}"] = float(p[i] / np.median(p[(d <= SNR_HALF_WIDTH_HZ) & (d > SNR_GUARD_HZ)]))
    return out


def fold_period(x: np.ndarray, fs: float) -> tuple[float, float]:
    env, rate = corpus.rms_envelope(x, fs, frame_s=FOLD_FRAME_S)
    t = np.arange(env.size) / rate
    best = (0.0, float("nan"))
    for p_ms in FOLD_PERIODS_MS:
        ph = ((t % (p_ms * 1e-3)) / (p_ms * 1e-3) * FOLD_BINS).astype(int)
        prof = np.bincount(ph, env, FOLD_BINS) / np.bincount(ph, None, FOLD_BINS)
        best = max(best, ((prof.max() - prof.min()) / env.mean(), float(p_ms)))
    return best[1], best[0]


def capture_row(capture_id: str) -> dict:
    cap = corpus.archive().open(capture_id)
    fs = 1.0 / cap.sample_interval_s
    n = int(round(windows.WINDOW_S * fs))
    intervals = cap.scored_intervals()
    starts = [t for s0, s1 in intervals for t in windows.window_starts(s0, s1)][:N_WINDOWS]
    snrs = [line_snrs(cap.amperes(int(round(t * fs)), int(round(t * fs)) + n), fs)
            for t in starts]
    i0 = int(round(intervals[0][0] * fs))
    period_ms, p2p = fold_period(cap.amperes(i0, i0 + int(FOLD_S * fs)), fs)
    fit = (cap.sidecar.get("alignment") or {}).get("fit") or {}
    e = cap.entry
    return {"capture_id": capture_id, "condition": e["condition"],
            "split": e["split"], "session": e["session"],
            "telemetry": e.get("telemetry_path") is not None,
            "snr_hz_median": {k: float(np.median([s[k] for s in snrs])) for k in snrs[0]},
            "n_windows": len(snrs),
            "period_ms": period_ms, "fold_p2p_frac": p2p,
            "scope_host_rate_ppm": fit.get("rate_ppm")}


def replication_snr(run_id: str) -> tuple[str, float]:
    cap = corpus.archive().open(run_id)
    return cap.entry["class"], line_snrs(cap.amperes(), 1.0 / cap.sample_interval_s)["5"]


def ac_lag_table() -> dict:
    """Per condition: dev 5 s windows with env_ac_peak_lag_s in AC_LAG_BAND_S."""
    lags = defaultdict(list)
    with WINDOWS_DEV.open() as fh:
        assert json.loads(next(fh))["split"] == "development"
        for line in fh:
            r = json.loads(line)
            lags[r["condition"]].append(r["features"]["env_ac_peak_lag_s"])
    lo, hi = AC_LAG_BAND_S
    return {c: {"n": len(v), "n_in_band": int(((np.array(v) >= lo) & (np.array(v) <= hi)).sum()),
                "median_lag_s": float(np.median(v))} for c, v in sorted(lags.items())}


def main() -> int:
    caps = [e["capture_id"] for e in windows.usable_captures()
            if e["split"] in SPLITS]
    reps = [e["capture_id"] for e in corpus.archive().captures(family="replication")]
    with Pool(6) as pool:
        rows = pool.map(capture_row, caps)
        rep = pool.map(replication_snr, reps)
    by_cls = defaultdict(list)
    for cls, s in rep:
        by_cls[cls].append(s)
    rep_summary = {c: {"n": len(v), "snr_5hz_median": float(np.median(v)),
                       "frac_snr_above_10": float(np.mean(np.array(v) > 10))}
                   for c, v in sorted(by_cls.items())}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "scripts/diagnose_5hz_line.py", "spec_task": "1.5",
        "corpus_commit": corpus.CORPUS_COMMIT, "splits": SPLITS,
        "params": {"n_windows": N_WINDOWS, "fold_s": FOLD_S,
                   "fold_periods_ms": [FOLD_PERIODS_MS[0], FOLD_PERIODS_MS[-1], FOLD_STEP_MS],
                   "fold_frame_s": FOLD_FRAME_S, "fold_bins": FOLD_BINS,
                   "lines_hz": LINES_HZ, "search_hz": SEARCH_HZ, "nfft": NFFT,
                   "snr_half_width_hz": SNR_HALF_WIDTH_HZ, "snr_guard_hz": SNR_GUARD_HZ,
                   "ac_lag_band_s": AC_LAG_BAND_S},
        "captures": sorted(rows, key=lambda r: (r["condition"] or "", r["capture_id"])),
        "replication": rep_summary,
        "ac_peak_lag_in_band": ac_lag_table(),
    }, indent=1))
    for r in sorted(rows, key=lambda r: -r["snr_hz_median"]["5"]):
        print(f"{r['capture_id']:40s} {r['condition']:26s} "
              + " ".join(f"{k}Hz={v:6.1f}" for k, v in r["snr_hz_median"].items())
              + f" P={r['period_ms']:.2f} ms p2p={100 * r['fold_p2p_frac']:.2f}% "
              f"ppm={r['scope_host_rate_ppm']}")
    print(json.dumps(rep_summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
