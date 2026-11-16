"""The RMS envelope, and the amplitude of a known tone in it.

Shared by spec tasks 3.3 and 3.4, which need the same observable for different
reasons: 3.3 measures how a commanded duty cycle shows up in it, 3.4 checks
that reducing the sample rate does not destroy it.

Why the envelope is the observable, and not the waveform
--------------------------------------------------------
The probe is AC-coupled with a ~34 Hz corner (datasheet, never measured here --
§7). A commanded cadence below that corner is therefore absent from the
*waveform*: there is no 3 Hz component in the current for the coil to pass. But
the cadence is not lost, because it rides as amplitude modulation on the much
faster switching activity the probe does pass. §3 states this, and it is the
reason the envelope must be computed **before** the waveform is reduced --
averaging the carrier away and then taking RMS destroys the thing the cadence
is written on, and no later step recovers it.

So the quantity here is RMS current in short frames, which is also how the
published baseline defines its feature (RMS over 2 ms frames, Gargiulo & Kulp).
Computing it the same way is what makes 3.4's band-power comparison against
their figures meaningful.

The DC caveat
-------------
`sqrt(mean(x^2))` inherits any DC offset in quadrature, and this channel
carries one that is instrumental rather than signal -- the coil has no DC
response, so a non-zero mean is scope and probe offset. On the committed soak
it biases the envelope by 11%. Every function here therefore expects `x` to
have had it removed already (`ChannelData.amperes(remove_dc=True)`), and
`rms_envelope` re-centres per frame as a second line of defence.

Units are amperes. Never watts (spec §3, §7).
"""

from __future__ import annotations

import numpy as np


def rms_envelope(x: np.ndarray, sample_rate_hz: float,
                 frame_s: float = 2e-3,
                 frame_samples: int | None = None) -> tuple[np.ndarray, float]:
    """RMS current in consecutive frames of `frame_s`, or of `frame_samples`.

    Returns `(envelope, envelope_rate_hz)`. Frames do not overlap: the
    published feature does not overlap them, and an overlap would correlate
    adjacent envelope samples and narrow the apparent linewidth of anything
    measured in the envelope spectrum.

    The trailing partial frame is dropped rather than computed over fewer
    samples, which would give it a different noise variance from every other
    frame and put a step at the end of the record.

    `frame_samples` overrides `frame_s` and exists for comparing two series at
    different rates. Rounding a duration to samples independently at each rate
    gives frame grids that drift apart -- 2 ms at 10.417 MS/s is 20833 samples
    and at 1.0417 MS/s is 2083, which is 20830 of the original, so the grids
    slip by three samples per frame. Over a few thousand frames that is a
    fraction of a frame of misalignment, and because this envelope is very
    nearly white frame to frame (measured lag-1 autocorrelation 0.021) even a
    third of a frame destroys any point-by-point comparison while leaving the
    means identical. `envelope_fidelity` therefore pins the sample counts so
    both series cover exactly the same windows.
    """
    frame = int(frame_samples if frame_samples is not None
                else round(frame_s * sample_rate_hz))
    if frame < 2:
        raise ValueError(
            f"a {frame_s * 1e3:.3f} ms frame is {frame} samples at "
            f"{sample_rate_hz:,.0f} Hz; it needs at least 2")
    n_frames = x.size // frame
    if n_frames < 2:
        raise ValueError(
            f"{x.size:,} samples give {n_frames} frames of {frame:,}; "
            f"an envelope needs at least 2")
    block = np.asarray(x[:n_frames * frame], dtype=np.float64).reshape(n_frames, frame)
    # Re-centre per frame. The caller should already have removed the
    # channel's DC offset; this also removes any slow residual drift that a
    # single global mean would leave inside the frame.
    block = block - block.mean(axis=1, keepdims=True)
    return np.sqrt((block ** 2).mean(axis=1)), sample_rate_hz / frame


def tone_amplitude(y: np.ndarray, rate_hz: float, freq_hz: float) -> float:
    """Amplitude of a sinusoid at exactly `freq_hz` in `y`, by projection.

    A single-bin DFT at an arbitrary frequency rather than an FFT peak: the
    commanded duty frequencies of task 3.3 are not bin-centred at any practical
    record length, and reading the nearest FFT bin would scallop the amplitude
    by up to 36% with a rectangular window -- a systematic error across a
    frequency sweep, which is exactly what a sweep must not have.

    A Hann window is applied and the result corrected for its coherent gain
    (0.5), which keeps neighbouring components from leaking into the estimate.
    Returns the amplitude of the component, not its RMS.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n < 4:
        raise ValueError(f"need at least 4 samples to project a tone, got {n}")
    if not 0 < freq_hz < rate_hz / 2:
        raise ValueError(
            f"{freq_hz} Hz is not inside (0, {rate_hz / 2}) -- the envelope "
            f"sampled at {rate_hz} Hz cannot represent it")
    t = np.arange(n) / rate_hz
    win = np.hanning(n)
    y = (y - y.mean()) * win
    phasor = np.exp(-2j * np.pi * freq_hz * t)
    # 2/sum(win) converts the one-sided projection to an amplitude and undoes
    # the window's coherent gain in the same constant.
    return float(2.0 * np.abs(np.dot(y, phasor)) / win.sum())


def tone_noise_floor(y: np.ndarray, rate_hz: float, freq_hz: float, *,
                     n_probes: int = 12, spacing_bins: float = 4.0) -> float:
    """How large a tone this record would show at a frequency carrying none.

    Projects at frequencies either side of `freq_hz`, offset by multiples of
    the record's own bin width, and returns the RMS of those amplitudes. That
    is the level at which `tone_amplitude` cannot tell signal from the
    envelope's own fluctuation.

    It is not optional bookkeeping. An honest continuous training loop is busy
    essentially all the time, so its optimizer cadence modulates the envelope
    by only about 1%, while the envelope itself fluctuates by ~50% frame to
    frame. Measured on a 5 s reference the cadence came out at 0.0511 A
    against a noise floor of 0.060 A -- the tone was *below* the floor, and the
    ratio of two such measurements looked like a result while being entirely
    noise. Anything reading a cadence amplitude must read this beside it.
    """
    n = y.size
    bin_hz = rate_hz / n
    amps = []
    for k in range(1, n_probes // 2 + 1):
        for sign in (-1, 1):
            f = freq_hz + sign * k * spacing_bins * bin_hz
            if 0 < f < rate_hz / 2:
                amps.append(tone_amplitude(y, rate_hz, f))
    if not amps:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(amps))))


def modulation_depth(y: np.ndarray, rate_hz: float, freq_hz: float) -> dict:
    """The tone at `freq_hz` in an envelope, against that envelope's own level.

    The ratio is what a duty-response sweep should report rather than the bare
    amplitude. The drive is a GPU workload, not a calibrated current source, so
    the absolute current it produces depends on how completely each busy window
    saturates the device -- which itself changes with the commanded frequency
    as the windows get shorter. Normalising by the mean envelope removes the
    part of that variation which is a change in delivered load rather than a
    property of the chain.

    It does not remove all of it, and this is not a calibrated transfer
    function. Spec task 3.3 is explicit that the result is the combined
    GPU + power-delivery + probe response and must not be described as
    isolating the probe.
    """
    amp = tone_amplitude(y, rate_hz, freq_hz)
    mean_level = float(np.mean(y))
    return {
        "amplitude_a": amp,
        "mean_envelope_a": mean_level,
        "modulation_depth": amp / mean_level if mean_level > 0 else float("nan"),
    }
