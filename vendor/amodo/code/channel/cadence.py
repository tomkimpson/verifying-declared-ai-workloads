"""Recover a commanded period from an RMS envelope (sub-spec 5B.5).

The periodic condition drives a non-AI workload at a commanded period.
The verification is that the period comes back out of the *envelope* of
the trace -- not the raw spectrum, because the periods sit far below the
probe's 34 Hz corner and reach the channel only as modulation of the
switching activity the probe does pass (spec §3; `channel.envelope`).

Two readings, both from the Phase 3 machinery:

* **recovered**: the frequency of the largest peak in the envelope's
  periodogram inside a search band around the commanded frequency. This
  is the recovery proper -- the band is wide (an octave either side by
  default) so the answer is not assumed.
* **resolvable**: the tone amplitude at the commanded frequency against
  the envelope's own noise floor beside it (`tone_amplitude`,
  `tone_noise_floor`), with the same ≥ 3 rule `envelope_fidelity` uses
  for `cadence_resolvable`. A recovered peak that the tone test cannot
  resolve is a number, not a result.

Units are amperes; nothing here is in watts (spec §3).
"""

from __future__ import annotations

import numpy as np

from channel.envelope import rms_envelope, tone_amplitude, tone_noise_floor

#: Same rule as `reduction.envelope_fidelity`.
RESOLVABLE_SNR = 3.0


def recover_cadence(envelope: np.ndarray, envelope_rate_hz: float,
                    commanded_hz: float, *, search_octaves: float = 1.0) -> dict:
    """Peak search and tone test on one envelope segment."""
    y = np.asarray(envelope, dtype=np.float64)
    n = y.size
    if n < 8:
        raise ValueError(f"an envelope of {n} samples cannot be searched")
    lo, hi = commanded_hz / 2 ** search_octaves, commanded_hz * 2 ** search_octaves
    if hi >= envelope_rate_hz / 2:
        raise ValueError(f"search band reaches {hi:g} Hz, above the envelope Nyquist "
                         f"{envelope_rate_hz / 2:g} Hz")
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft((y - y.mean()) * win))
    freqs = np.fft.rfftfreq(n, 1.0 / envelope_rate_hz)
    band = (freqs >= lo) & (freqs <= hi)
    if not band.any():
        raise ValueError(f"no periodogram bin inside [{lo:g}, {hi:g}] Hz: the segment "
                         f"({n / envelope_rate_hz:.1f} s) is too short for {commanded_hz:g} Hz")
    k = np.flatnonzero(band)[np.argmax(spec[band])]
    recovered = float(freqs[k])
    amp = tone_amplitude(y, envelope_rate_hz, commanded_hz)
    floor = tone_noise_floor(y, envelope_rate_hz, commanded_hz)
    snr = amp / floor if floor > 0 else float("nan")
    return {
        "segment_s": n / envelope_rate_hz,
        "bin_hz": envelope_rate_hz / n,
        "search_band_hz": [lo, hi],
        "recovered_hz": recovered,
        "recovered_period_s": 1.0 / recovered,
        "recovered_over_commanded": recovered / commanded_hz,
        "recovered_within_one_bin": abs(recovered - commanded_hz) <= envelope_rate_hz / n,
        "tone_amplitude_a": amp,
        "tone_noise_floor_a": floor,
        "tone_snr": snr,
        "mean_envelope_a": float(y.mean()),
        "modulation_depth": amp / float(y.mean()) if y.mean() > 0 else float("nan"),
        "cadence_resolvable": bool(np.isfinite(snr) and snr >= RESOLVABLE_SNR),
    }


def recover_from_trace(x_amperes: np.ndarray, sample_rate_hz: float, commanded_hz: float,
                       *, frame_s: float = 2e-3, search_octaves: float = 1.0) -> dict:
    """Envelope first (spec §3: before any reduction), then the recovery."""
    env, env_rate = rms_envelope(x_amperes, sample_rate_hz, frame_s=frame_s)
    out = recover_cadence(env, env_rate, commanded_hz, search_octaves=search_octaves)
    out["frame_s"] = frame_s
    out["envelope_rate_hz"] = env_rate
    return out
