"""Amplitude-free envelope and spectral features on the published recipe.

Spec task 4.2. The published baseline (spec §2, arXiv:2609.00309) builds its
classifier on features that are **explicitly amplitude-free**, with the
envelope defined as RMS current over 2 ms frames. That choice is not a detail
to be improved on: the probe's absolute scale depends on how it is clamped
around the auxiliary conductors, which is not reproducible between sessions,
let alone between rigs. A feature that moved with gain would encode the clamp,
not the workload.

So every feature here satisfies `f(g * x) == f(x)` for any gain `g > 0`, by
construction rather than by hope, and `tests/test_features.py` checks it
against a synthetic gain change. Three constructions do the work:

* **shape statistics of a self-normalised envelope** -- the envelope is
  divided by its own mean before anything is measured on it;
* **fractions of spectral power**, where the gain cancels between numerator
  and denominator;
* **normalised moments** -- flatness, centroid, roll-off, autocorrelation --
  each of which is a ratio of like-scaled quantities.

What this module is not
-----------------------
It is not detector development, which spec §7 puts downstream in the two paper
repos. In particular the bands below are *descriptive* -- decades, plus the
20 kHz landmark the published Figure 2 identifies -- and are deliberately not
a selection tuned to separate these classes. §7 says the 0.3-1.7 Hz versus
200 Hz-2 kHz question is not settled here, and tuning bands against this
corpus is how it would get settled here by accident.

What a five-second window cannot do
-----------------------------------
The envelope is sampled at 500 Hz (one frame per 2 ms) and a trace is five
seconds, so the envelope spectrum has 2500 points and, at the segment length
used here, a resolution of about 0.5 Hz. **The lowest band therefore starts at
0.5 Hz, not at the 0.3 Hz the published band division mentions: a five-second
record cannot resolve it.** That is not a limitation of this code. It is the
limitation of the five-second protocol, and it is a concrete instance of the
axis §1 says this campaign exists to add.

Units are amperes in, dimensionless out (except the two frequency-valued
features, which are in hertz). Never watts (spec §3, §7).
"""

from __future__ import annotations

import numpy as np

from channel.aliasing import band_power, welch_psd
from channel.envelope import rms_envelope

#: The published envelope frame. 2 ms is 400 samples at the frozen 200 kS/s.
FRAME_S = 2e-3

#: Welch segment for the waveform PSD: 65536 samples is 0.33 s and 3.05 Hz
#: at 200 kS/s, giving about 30 segments across a five-second trace.
WAVEFORM_NPERSEG = 1 << 16

#: Welch segment for the envelope PSD: 1024 frames is 2.05 s and 0.49 Hz at
#: the 500 Hz envelope rate, giving 3 segments. Longer would resolve lower and
#: average less; this is the compromise a 5 s record allows.
ENVELOPE_NPERSEG = 1024

#: Envelope bands, in Hz. The lowest starts at 0.5 because 5 s cannot resolve
#: below it -- see the module docstring.
ENVELOPE_BANDS = (
    ("0p5_2hz", 0.5, 2.0),
    ("2_10hz", 2.0, 10.0),
    ("10_50hz", 10.0, 50.0),
    ("50_150hz", 50.0, 150.0),
    ("150_250hz", 150.0, 250.0),
)

#: Waveform bands, in Hz. Decades from the probe's nominal corner to the
#: retained Nyquist, split at the 20 kHz landmark below which the published
#: Figure 2 puts almost all spectral power.
WAVEFORM_BANDS = (
    ("34_100hz", 34.0, 100.0),
    ("100hz_1khz", 100.0, 1e3),
    ("1_10khz", 1e3, 1e4),
    ("10_20khz", 1e4, 2e4),
    ("20_50khz", 2e4, 5e4),
    ("50_100khz", 5e4, 1e5),
)

FEATURE_NAMES: tuple[str, ...] = (
    "env_cv", "env_skew", "env_kurtosis",
    "env_p10_over_p50", "env_p90_over_p50", "env_p99_over_p50",
    "env_iqr_over_median",
    "env_ac_lag1", "env_ac_peak", "env_ac_peak_lag_s",
    "env_spec_flatness",
    *(f"env_frac_{name}" for name, _, _ in ENVELOPE_BANDS),
    "wav_spec_flatness", "wav_spec_centroid_hz", "wav_spec_rolloff95_hz",
    *(f"wav_frac_{name}" for name, _, _ in WAVEFORM_BANDS),
)


def _flatness(psd: np.ndarray) -> float:
    """Geometric mean over arithmetic mean: 1 for white, →0 for a pure line.

    Computed in the log domain, because the geometric mean of tens of
    thousands of PSD bins underflows to zero as a product. Bins at exactly
    zero are dropped rather than floored: a floor is an arbitrary constant
    that sets the answer for a sparse spectrum.
    """
    p = psd[psd > 0]
    if p.size < 2:
        return float("nan")
    return float(np.exp(np.log(p).mean()) / p.mean())


def _rolloff(freqs: np.ndarray, psd: np.ndarray, fraction: float) -> float:
    """The frequency below which `fraction` of the spectrum's power sits."""
    total = psd.sum()
    if total <= 0:
        return float("nan")
    idx = int(np.searchsorted(np.cumsum(psd), fraction * total))
    return float(freqs[min(idx, freqs.size - 1)])


def _autocorrelation(e: np.ndarray) -> np.ndarray:
    """Normalised autocorrelation of a mean-removed series, lag 0 first."""
    x = e - e.mean()
    n = 1 << int(np.ceil(np.log2(2 * x.size)))
    spec = np.fft.rfft(x, n)
    ac = np.fft.irfft(spec * np.conj(spec), n)[:x.size]
    return ac / ac[0] if ac[0] > 0 else np.full(x.size, np.nan)


def _cadence_lag(ac: np.ndarray) -> int:
    """The lag of the strongest autocorrelation peak past the zero crossing.

    Returns -1 when there is no periodic structure to find.

    **Read the name carefully: this is a peak lag, not a period.** Where the
    envelope can resolve the workload's repetition at all, the lag lands on an
    integer multiple of it — and which multiple is decided by the correlation
    decay and by noise rather than by the signal.

    **The multiples are measured, not asserted here.** `extract_features.py`
    emits them per cell into `features.json` under `peak_lag_multiples`,
    against the ground truth the corpus already carries
    (`workload_stats.mean_period_s` for training, `mean_step_s` for non-AI).
    An earlier version of this docstring listed them inline, was not
    re-measured after the corpus was recaptured, and shipped a table that the
    corpus in the same commit contradicted — which is the whole reason the
    numbers now come out of a script.

    Two things that list obscured, and which the emitted block reports
    explicitly:

    * **The multiple is not always stable within a workload.** It is for some
      cells and is not for others, so "stable within a workload" is a
      per-cell fact to be read off `peak_lag_multiples`, never a property of
      the feature.
    * **A workload whose period is shorter than about two envelope frames
      cannot be resolved at all.** The envelope is sampled at `1/FRAME_S`, so
      a repetition near or below that interval aliases and the lag becomes
      arbitrary. `resolvable` in the emitted block is that test.

    The history is worth keeping because this has now been wrong twice.

    1. **`argmax` over every lag above zero.** A smooth envelope's
       autocorrelation falls monotonically away from lag 0, so the largest
       value above it is always the smallest lag searched. Returned 4 ms for
       a 3 Hz modulation; the cadence test caught it.
    2. **Global max after the first zero crossing** -- the standard fix for
       (1), and what this still is. Its standard failure is the octave error
       above, which the synthetic test structurally could not see because
       `synthetic_trace()` is one clean sinusoid whose first harmonic peak
       dominates.

    A sub-multiple correction was tried and rejected: at a 0.85 acceptance
    tolerance it fixed `gemm` (6.00x -> 1.01x) and broke `tiny` (3.00x ->
    0.25x), i.e. it traded one octave error for a sub-octave one and the
    tolerance was a constant tuned on one afternoon. Estimating a period
    robustly needs a real pitch tracker (YIN or a harmonic comb), and spec §7
    puts that kind of development downstream in the paper repos. **This module
    owes the classifier an amplitude-free feature, not a cadence
    measurement** -- and the corpus already carries the cadence exactly, from
    the workload's own logs, which is where anyone wanting a period should
    read it.

    What the feature is good for: it is scale-invariant, and where the cell is
    resolvable the multiple is consistent enough to separate classes. What it
    must not be used for is quoting a cadence — the corpus carries that
    exactly, in the workload's own logs.

    The search runs to half the record; beyond that an autocorrelation is
    estimated from fewer samples than lags and is mostly its own variance.
    """
    hi = max(3, ac.size // 2)
    below = np.flatnonzero(ac[:hi] <= 0)
    # No zero crossing is a monotone decay: nothing periodic to find. Serving
    # is the case that matters -- having no concentrated cadence is its
    # electrical character, not a measurement failure.
    if below.size == 0:
        return -1
    lo = int(below[0])
    if lo >= hi - 1:
        return -1
    peak = int(np.argmax(ac[lo:hi])) + lo
    return peak if ac[peak] > 0 else -1


def envelope_features(envelope: np.ndarray, envelope_rate_hz: float) -> dict:
    """Shape and spectrum of the self-normalised RMS envelope."""
    mean = float(envelope.mean())
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("envelope mean is not positive; cannot normalise")
    e = envelope / mean

    median = float(np.median(e))
    q25, q75 = (float(v) for v in np.percentile(e, [25, 75]))
    centred = e - e.mean()
    sd = float(centred.std())
    # Fisher skewness and excess kurtosis, written out rather than pulled from
    # scipy. scipy exists in the `analysis` dependency group as a transitive
    # dependency of scikit-learn, but not in the capture environment, and this
    # module runs in both -- the two lines below are the whole of what would
    # have been used from it anyway.
    skew = float((centred ** 3).mean() / sd ** 3) if sd > 0 else float("nan")
    kurt = float((centred ** 4).mean() / sd ** 4 - 3.0) if sd > 0 else float("nan")

    ac = _autocorrelation(e)
    peak_lag = _cadence_lag(ac)
    found = peak_lag > 0

    freqs, psd = welch_psd(e, envelope_rate_hz, nperseg=ENVELOPE_NPERSEG)
    total = band_power(freqs, psd, 0.0, envelope_rate_hz / 2)

    out = {
        "env_cv": sd,                     # e has unit mean, so sd IS std/mean
        "env_skew": skew,
        "env_kurtosis": kurt,
        "env_p10_over_p50": float(np.percentile(e, 10)) / median,
        "env_p90_over_p50": float(np.percentile(e, 90)) / median,
        "env_p99_over_p50": float(np.percentile(e, 99)) / median,
        "env_iqr_over_median": (q75 - q25) / median,
        "env_ac_lag1": float(ac[1]),
        "env_ac_peak": float(ac[peak_lag]) if found else 0.0,
        # 0.0 is the "no periodic structure" sentinel, and it is a fixed
        # value on purpose. An earlier version returned half the record
        # length, which is 2.5 s on a 5 s trace and 300 s on a Phase 6
        # ten-minute one -- a sentinel that scales with the record makes the
        # feature incomparable between corpora, and a forest splits on it as
        # a category. No real lag is 0.
        "env_ac_peak_lag_s": peak_lag / envelope_rate_hz if found else 0.0,
        "env_spec_flatness": _flatness(psd),
    }
    for name, lo_hz, hi_hz in ENVELOPE_BANDS:
        out[f"env_frac_{name}"] = (
            band_power(freqs, psd, lo_hz, hi_hz) / total if total > 0
            else float("nan"))
    return out


def waveform_features(x_amperes: np.ndarray, sample_rate_hz: float) -> dict:
    """Shape of the waveform's own spectrum, as fractions and normalised moments."""
    freqs, psd = welch_psd(x_amperes, sample_rate_hz, nperseg=WAVEFORM_NPERSEG)
    total = band_power(freqs, psd, 0.0, sample_rate_hz / 2)

    out = {
        "wav_spec_flatness": _flatness(psd),
        "wav_spec_centroid_hz": (float((freqs * psd).sum() / psd.sum())
                                 if psd.sum() > 0 else float("nan")),
        "wav_spec_rolloff95_hz": _rolloff(freqs, psd, 0.95),
    }
    for name, lo_hz, hi_hz in WAVEFORM_BANDS:
        out[f"wav_frac_{name}"] = (
            band_power(freqs, psd, lo_hz, hi_hz) / total if total > 0
            else float("nan"))
    return out


def extract_features(x_amperes: np.ndarray, sample_rate_hz: float) -> dict:
    """Every feature for one trace, keyed by `FEATURE_NAMES`.

    `x_amperes` must already have had the channel's instrumental DC offset
    removed -- `Capture.amperes(remove_dc=True)` -- because the RMS envelope
    squares the signal and an uncorrected offset enters in quadrature.
    """
    envelope, envelope_rate_hz = rms_envelope(x_amperes, sample_rate_hz,
                                              frame_s=FRAME_S)
    features = {**envelope_features(envelope, envelope_rate_hz),
                **waveform_features(x_amperes, sample_rate_hz)}
    missing = set(FEATURE_NAMES) - set(features)
    extra = set(features) - set(FEATURE_NAMES)
    if missing or extra:
        raise AssertionError(
            f"feature set drifted from FEATURE_NAMES: missing {missing}, "
            f"unexpected {extra}")
    return features
