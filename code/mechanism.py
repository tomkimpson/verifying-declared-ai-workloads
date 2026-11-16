"""H1 mechanism (spec task 4.2): how much step-locked structure each run keeps.

The envelope is the 2 ms RMS frame series of the raw 200 kS/s current
(amodo `channel.envelope.rms_envelope`, spec §3), computed per scored interval
on the mean-removed waveform and divided by that interval's median, so it is
amplitude-free.

Step-locked profile (amendment 2026-10-02g, replacing the fixed-window "dip
depth" of 2026-10-02f, which read one of several micro-step troughs and was not
comparable across runs). Segments one median step duration long, centred on
each `step` end, are averaged. For back-to-back steps the segment spans the
run's four micro-steps, whatever its step period; for the gapped `-0p25` it is
half step, half idle gap. The statistic is the profile's CV (standard deviation
over mean across the segment): how strongly the envelope is locked to the
optimizer-step cycle. It counts a step-locked level change (as at `-0p25`'s
step/gap boundary) as well as dips, so `step_locked` also reports the
within-step CV, over the half of the segment before the step end, which lies
inside the step for every run (amendment 2026-10-02g, PR #4 fix round 3).

Null: the same statistic for segments cut at uniformly random times in the same
intervals, as many as there are step ends, over N_NULL independent draws. A
run's step-locked structure is above noise if its CV exceeds the null's 95th
percentile. Interval: a basic (reverse-percentile) circular block bootstrap over
consecutive segments, since a dispersion statistic is biased upward under
resampling and the basic interval corrects for that shift.
Development data only; not pre-registered (pre-registration §11).
"""

from __future__ import annotations

import numpy as np

import corpus

#: Segments per bootstrap block: 20 consecutive steps.
BLOCK_SEGMENTS = 20
N_BOOT = 2000
N_NULL = 200
SEED = 0


def interval_envelopes(capture_id: str, archive=None):
    """Yield (s0, s1, normalised envelope, envelope rate) per scored interval."""
    cap = (archive or corpus.archive()).open(capture_id)
    fs = 1.0 / cap.sample_interval_s
    for s0, s1 in cap.scored_intervals():
        i0 = int(round(s0 * fs))
        x = cap.amperes(i0, int(round(s1 * fs)))
        env, er = corpus.rms_envelope(x - x.mean(), fs)
        # Frame k covers [i0 + k*frame, ...): its time origin is the interval's first sample.
        yield i0 / fs, s1, env / np.median(env), er


def step_duration_s(cap) -> float:
    """Median optimizer-step duration on the scope clock."""
    return float(np.median([e["scope_end_s"] - e["scope_start_s"] for e in cap.events("step")]))


def micro_step_hz(cap) -> float | None:
    """The micro-step rhythm: micro-steps per step (from the step payloads)
    over the median step duration. None for a run without steps."""
    ev = cap.events("step")
    if not ev:
        return None
    return float(np.median([e["payload"]["micro_steps"] for e in ev])) / step_duration_s(cap)


def cut(envs: list, times_per_interval: list, half_s: float) -> np.ndarray:
    """Segments of +-half_s around each time, from the matching interval envelope."""
    out = []
    for (t0, _, env, er), times in zip(envs, times_per_interval):
        h = int(round(half_s * er))
        for t in times:
            i = int(round((t - t0) * er))
            if i - h >= 0 and i + h <= len(env):
                out.append(env[i - h:i + h])
    return np.array(out)


def step_times(envs: list, ends: np.ndarray, half_s: float) -> list:
    return [ends[(ends > t0 + half_s) & (ends < s1 - half_s)] for t0, s1, _, _ in envs]


def random_times(envs: list, counts: list, half_s: float, rng) -> list:
    return [rng.uniform(t0 + half_s, s1 - half_s, size=n)
            for (t0, s1, _, _), n in zip(envs, counts)]


def profile_cv(segs: np.ndarray) -> float:
    """CV of the mean profile across the segment."""
    p = segs.mean(0)
    return float(p.std() / p.mean())


def cv_with_ci(segs: np.ndarray, block: int = BLOCK_SEGMENTS, n_boot: int = N_BOOT,
               seed: int = SEED) -> dict:
    """Profile CV with a 95% basic circular block bootstrap interval over
    consecutive segments."""
    n = len(segs)
    b = min(block, n)
    k = int(np.ceil(n / b))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, k))
    idx = (starts[:, :, None] + np.arange(b)).reshape(n_boot, k * b)[:, :n] % n
    boot = np.array([profile_cv(segs[i]) for i in idx])
    cv = profile_cv(segs)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"cv": cv, "n_segments": int(n), "ci95": [float(2 * cv - hi), float(2 * cv - lo)]}


def step_locked(capture_id: str, archive=None, n_null: int = N_NULL, seed: int = SEED) -> dict:
    """Step-aligned profile CV with its interval, the random-alignment null, and
    the mean profile, for one run."""
    arc = archive or corpus.archive()
    cap = arc.open(capture_id)
    half = step_duration_s(cap) / 2
    ends = np.sort([e["scope_end_s"] for e in cap.events("step")])
    envs = list(interval_envelopes(capture_id, arc))
    times = step_times(envs, ends, half)
    segs = cut(envs, times, half)
    if not len(segs):
        raise ValueError(f"{capture_id}: no step segments")
    rng = np.random.default_rng(seed)
    counts = [len(t) for t in times]
    h = segs.shape[1] // 2
    null, null_in = [], []
    for _ in range(n_null):
        r = cut(envs, random_times(envs, counts, half, rng), half)
        null.append(profile_cv(r))
        null_in.append(profile_cv(r[:, :h]))
    out = cv_with_ci(segs)
    within = cv_with_ci(segs[:, :h])
    out.update(half_window_s=half, envelope_rate_hz=envs[0][3],
               null_cv_median=float(np.median(null)), null_cv_p95=float(np.quantile(null, 0.95)),
               above_null=bool(out["cv"] > np.quantile(null, 0.95)),
               within_step={"cv": within["cv"], "ci95": within["ci95"],
                            "null_cv_p95": float(np.quantile(null_in, 0.95))},
               mean_profile=[float(f"{a:.5g}") for a in segs.mean(0)])
    return out


#: 4.2 5 Hz item, Stage A (amendment 2026-10-02f). Over a whole scored interval
#: (~176 s) the envelope spectrum resolves ~0.006 Hz (0.0038 Hz zero-padded bin
#: spacing), so the harness line at
#: 5.000 Hz separates from `-0p5`'s 5.21 Hz micro-step line, which Phase 1's
#: 5 s windows (0.2 Hz resolution, +-0.3 Hz search) could not. SNR as Phase 1
#: (`scripts/diagnose_5hz_line.py`): peak power over the median of a noise band
#: around it, with narrow bins.
HARNESS_HZ = 5.000
LINE_SEARCH_HZ = 0.02
LINE_GUARD_HZ = 0.05
LINE_NOISE_HALF_WIDTH_HZ = 0.5
#: Stage A's rejection level: the harness line is absent below this SNR.
LINE_ABSENT_SNR = 5.0


def envelope_spectrum(env: np.ndarray, rate: float) -> tuple[np.ndarray, np.ndarray]:
    """(frequencies, power) of a long envelope record, Hann windowed."""
    e = env / env.mean() - 1.0
    nfft = 1 << int(np.ceil(np.log2(len(e))))
    p = np.abs(np.fft.rfft(e * np.hanning(len(e)), n=nfft)) ** 2
    return np.fft.rfftfreq(nfft, d=1.0 / rate), p


def line_snr(f: np.ndarray, p: np.ndarray, f0: float) -> tuple[float, float]:
    """(SNR, peak frequency) of the strongest bin within LINE_SEARCH_HZ of f0."""
    near = np.abs(f - f0) <= LINE_SEARCH_HZ
    i = np.flatnonzero(near)[np.argmax(p[near])]
    d = np.abs(f - f[i])
    noise = np.median(p[(d <= LINE_NOISE_HALF_WIDTH_HZ) & (d > LINE_GUARD_HZ)])
    return float(p[i] / noise), float(f[i])


#: Stage B (amendment 2026-10-02f): the harness line's narrowband component is
#: taken from the unwindowed long-record envelope spectrum within this
#: half-width of each removed frequency.
REMOVE_HALF_WIDTH_HZ = LINE_SEARCH_HZ


def harness_modulation(env: np.ndarray, rate: float, freqs_hz=(HARNESS_HZ,),
                       half_width_hz: float = REMOVE_HALF_WIDTH_HZ) -> np.ndarray:
    """m(t) at frame rate: the fractional envelope modulation within
    `half_width_hz` of each frequency in `freqs_hz`, from the whole record."""
    e = env / env.mean() - 1.0
    spec = np.fft.rfft(e)
    f = np.fft.rfftfreq(len(e), d=1.0 / rate)
    keep = np.zeros(len(f), bool)
    for f0 in freqs_hz:
        keep |= np.abs(f - f0) <= half_width_hz
    return np.fft.irfft(np.where(keep, spec, 0), n=len(e))


def demodulate(x: np.ndarray, fs: float, freqs_hz=(HARNESS_HZ,)) -> np.ndarray:
    """The mean-removed interval `x` divided by (1 + m(t)), m interpolated from
    frame centres to every sample. A multiplicative modulation of the current
    scales its RMS envelope by the same factor, so this removes it."""
    x = x - x.mean()
    env, rate = corpus.rms_envelope(x, fs)
    m = harness_modulation(env, rate, freqs_hz)
    frame = int(round(fs / rate))
    centres = (np.arange(len(env)) + 0.5) * frame
    return x / (1.0 + np.interp(np.arange(len(x)), centres, m))
