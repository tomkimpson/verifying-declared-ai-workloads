"""The candidate reduction paths, and what each one does to the envelope.

Spec task 3.4's second clause: compare each candidate acquisition setting
against a high-rate reference on cadence structure and envelope fidelity. Task
3.5 then freezes one of these.

The three paths, and why they are not equivalent
------------------------------------------------
* **subsample** -- keep every nth sample, no filter. What
  `PICO_RATIO_MODE_DECIMATE` does, and what streaming raw at a low rate
  amounts to. Aliasing is total: every component above the new Nyquist arrives
  in the retained band with its power intact. Crucially, that means RMS is
  *conserved*, so an envelope computed after subsampling keeps its amplitude
  even though the waveform is now wrong in detail.
* **boxcar** -- average each block of n samples to one.
  `PICO_RATIO_MODE_AVERAGE`, whose response is
  |sin(pi f n / f_s) / (n sin(pi f / f_s))| -- about -3.9 dB at the retained
  Nyquist with a first sidelobe near -13.5 dB (§3). It is a genuine low-pass,
  so it *removes* power rather than folding it, and an envelope computed
  afterwards loses the part of the carrier it removed.
* **filtered** -- a proper anti-alias filter, then decimate. Removes the
  out-of-band power deliberately and knows it has.

§3's warning is that the second and third destroy what the first preserves:
"the envelope must be computed **before** the raw AC waveform is reduced.
Averaging the waveform down to a low retained rate and *then* taking RMS
destroys the carrier the cadence rides on, and no later step recovers it."
This module exists to measure how much that costs on the real signal rather
than to argue about it, so `envelope_fidelity` compares each path's
after-reduction envelope against the one computed at the raw rate.

There is a fourth path, and it is the one that cannot lose: compute the
envelope online at the acquisition rate and store *that* instead of the
waveform. It has no entry here because it is the reference by construction.

Units are amperes. Never watts (spec §3, §7).
"""

from __future__ import annotations

import numpy as np

from .envelope import rms_envelope, tone_amplitude, tone_noise_floor

PATHS = ("subsample", "boxcar", "filtered")


def subsample(x: np.ndarray, factor: int) -> np.ndarray:
    """Keep every `factor`-th sample. No filter, so aliasing is total."""
    if factor < 1:
        raise ValueError(f"the decimation factor must be >= 1, got {factor}")
    return np.asarray(x)[::factor]


def boxcar(x: np.ndarray, factor: int) -> np.ndarray:
    """Average each block of `factor` samples: PICO_RATIO_MODE_AVERAGE.

    The trailing partial block is dropped rather than averaged over fewer
    samples, which would give the last point a different variance from every
    other one.
    """
    if factor < 1:
        raise ValueError(f"the decimation factor must be >= 1, got {factor}")
    x = np.asarray(x, dtype=np.float64)
    n = x.size // factor
    if n < 1:
        raise ValueError(f"{x.size} samples do not fill one block of {factor}")
    return x[:n * factor].reshape(n, factor).mean(axis=1)


def filtered(x: np.ndarray, factor: int, *, numtaps: int | None = None) -> np.ndarray:
    """Low-pass to the new Nyquist, then decimate.

    A windowed-sinc FIR applied by FFT convolution, rather than scipy's
    `decimate`: the repo has no scipy dependency and this is the one filter it
    needs. The cutoff sits at 0.9 of the new Nyquist, leaving a transition band
    rather than placing the corner exactly where aliasing begins.

    **What it actually rejects, measured, not quoted from a window table.** A
    Hamming-windowed sinc reaches ~53 dB deep in its stopband, but the
    transition band is wide and the numbers that matter are at the new Nyquist,
    not deep in the stop. At factor 20 (1 MS/s -> 50 kS/s, new Nyquist 25 kHz):

        26 kHz   -13.1 dB          40 kHz   -55.5 dB
        30 kHz   -28.0 dB         187 kHz   -76.4 dB

    53 dB is first reached at 33.1 kHz, i.e. 1.32x the new Nyquist, and the peak
    of |H| anywhere above the new Nyquist is -10.7 dB. An earlier version of
    this docstring claimed "about 53 dB of stopband rejection" without that
    qualification; the tests probe at 7.5x and 12x Nyquist and so could not
    contradict it.

    (The 40 kHz and 187 kHz figures are the filter's true response, computed
    from `h` directly. A first correction quoted -50.2 and -65.8 dB, which were
    the RMS of this function's *output* including the FFT-convolution start and
    end transients -- a property of the measurement, not of the filter.)

    **At large factors the filter degenerates and this function stops being an
    anti-alias filter at all.** `numtaps` caps at 2001, so at factor 1002 -- the
    2 kS/s candidate `measure_reduction.py` sweeps -- the response is -1.21 dB
    at half the new Nyquist, -3.99 dB at the 0.9 passband edge and -4.96 dB at
    the new Nyquist, with an aliased/wanted power ratio of 13.9% on white input.
    Any "filtered" result at that factor is partly a measurement of an
    under-designed filter rather than of reduction damage, and must be read that
    way. Raise the cap if a large-factor result ever needs to be load-bearing.

    The convolution is done through the FFT and not with `np.convolve`, which
    is direct and therefore O(n * taps). The reference records here are tens of
    millions of samples against up to 2001 taps -- of order 1e11 operations,
    which does not finish.
    """
    if factor < 1:
        raise ValueError(f"the decimation factor must be >= 1, got {factor}")
    if factor == 1:
        return np.asarray(x, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if numtaps is None:
        # Enough taps for the transition band to be narrow at large factors,
        # capped so an enormous factor does not produce an enormous filter.
        numtaps = int(min(8 * factor | 1, 2001)) | 1

    cutoff = 0.9 / factor               # fraction of the ORIGINAL Nyquist
    n = np.arange(numtaps) - (numtaps - 1) / 2
    h = cutoff * np.sinc(cutoff * n) * np.hamming(numtaps)
    h /= h.sum()

    # 'same' alignment, computed through the FFT: the output is trimmed so it
    # is not shifted relative to the reference envelope it is compared against.
    # A shift would show up as a correlation deficit and be misread as the
    # filter having destroyed structure it merely moved.
    n_full = x.size + numtaps - 1
    n_fft = 1 << int(np.ceil(np.log2(n_full)))
    y = np.fft.irfft(np.fft.rfft(x, n_fft) * np.fft.rfft(h, n_fft), n_fft)
    start = (numtaps - 1) // 2
    return y[start:start + x.size][::factor]


def reduce_path(x: np.ndarray, factor: int, path: str) -> np.ndarray:
    if path == "subsample":
        return subsample(x, factor)
    if path == "boxcar":
        return boxcar(x, factor)
    if path == "filtered":
        return filtered(x, factor)
    raise ValueError(f"unknown reduction path {path!r}; expected one of {PATHS}")


def envelope_fidelity(x: np.ndarray, raw_rate_hz: float, retained_rate_hz: float,
                      path: str, *, frame_s: float = 2e-3,
                      cadence_hz: float | None = None) -> dict:
    """One candidate setting, against the envelope computed at the raw rate.

    The reference is `rms_envelope` of the unreduced signal; the candidate is
    `rms_envelope` of the same signal after `path` has taken it to
    `retained_rate_hz`. Both use the same frame duration, so both envelopes
    land on the same time grid and can be compared point by point.

    Returns the relative error of the envelope, its correlation with the
    reference, and -- when `cadence_hz` is given -- what each one recovers of
    the workload's cadence. The cadence figure is the one that matters for the
    campaign: an envelope can track the mean level well and still have lost the
    modulation that identifies the workload.
    """
    factor = int(round(raw_rate_hz / retained_rate_hz))
    if factor < 1:
        raise ValueError(
            f"the retained rate ({retained_rate_hz:,.0f} Hz) is above the raw "
            f"rate ({raw_rate_hz:,.0f} Hz)")

    # Pin both frame grids to the SAME time windows. Rounding 2 ms to samples
    # independently at each rate slips the grids by a few samples per frame,
    # and this envelope is very nearly white frame to frame, so a fraction of
    # a frame of drift wrecks the point-by-point comparison while leaving the
    # means identical. Measured before the fix: level error 0.0009 -- the two
    # envelopes agreed to four decimal places on the first frames -- alongside
    # a 25% RMS error and a correlation of 0.845, which is a comparison of two
    # correct series that had slid apart.
    #
    # Making the reference frame an exact multiple of the decimation factor
    # means the candidate frame is that multiple divided by the factor, and
    # the two cover identical windows for the whole record.
    ref_frame = int(round(frame_s * raw_rate_hz))
    ref_frame -= ref_frame % factor
    if ref_frame < factor:
        raise ValueError(
            f"a {frame_s * 1e3:.3f} ms frame at {raw_rate_hz:,.0f} Hz does not "
            f"span one decimated sample at a factor of {factor}")

    # Recomputed per candidate rather than passed in. `ref_frame` depends on
    # the decimation factor -- it is rounded down to a multiple of it -- so a
    # single reference envelope cannot serve every candidate, and reusing one
    # would reintroduce the grid mismatch this alignment exists to remove. The
    # cost is a reshape and a mean over the record; the expensive step in this
    # function is the filter, not this.
    ref_env, ref_rate = rms_envelope(x, raw_rate_hz, frame_samples=ref_frame)
    reduced = reduce_path(x, factor, path)
    cand_env, cand_rate = rms_envelope(reduced, raw_rate_hz / factor,
                                       frame_samples=ref_frame // factor)

    n = min(ref_env.size, cand_env.size)
    ref_env, cand_env = ref_env[:n], cand_env[:n]

    denom = float(np.sqrt(np.mean(ref_env ** 2)))
    rel_rms_error = float(np.sqrt(np.mean((cand_env - ref_env) ** 2)) / denom)
    rel_level_error = float(np.mean(cand_env) / np.mean(ref_env) - 1.0)
    corr = float(np.corrcoef(ref_env, cand_env)[0, 1]) if n > 2 else float("nan")

    out = {
        "path": path,
        "retained_rate_hz": raw_rate_hz / factor,
        "decimation_factor": factor,
        "frame_s": frame_s,
        "frame_samples_reference": ref_frame,
        "frame_samples_candidate": ref_frame // factor,
        "envelope_rate_hz": ref_rate,
        "reference_mean_a": float(np.mean(ref_env)),
        "candidate_mean_a": float(np.mean(cand_env)),
        # How much of the envelope's LEVEL survives. Negative means the path
        # removed power the reference had.
        "relative_level_error": rel_level_error,
        # How much of the envelope's SHAPE survives, as a fraction of the
        # reference's own RMS.
        "relative_rms_error": rel_rms_error,
        "correlation": corr,
    }
    if cadence_hz is not None and cadence_hz < ref_rate / 2:
        ref_tone = tone_amplitude(ref_env, ref_rate, cadence_hz)
        cand_tone = tone_amplitude(cand_env, cand_rate, cadence_hz)
        # The floor beside the tone, always. A cadence ratio computed from two
        # amplitudes that are both below the noise is a number, not a result,
        # and the first run of this produced exactly that -- "cadence kept"
        # above 1 on every path, which a reduction cannot do.
        ref_floor = tone_noise_floor(ref_env, ref_rate, cadence_hz)
        cand_floor = tone_noise_floor(cand_env, cand_rate, cadence_hz)
        ref_snr = ref_tone / ref_floor if ref_floor > 0 else float("nan")
        out["cadence_hz"] = cadence_hz
        out["reference_cadence_a"] = ref_tone
        out["candidate_cadence_a"] = cand_tone
        out["reference_cadence_noise_a"] = ref_floor
        out["candidate_cadence_noise_a"] = cand_floor
        out["reference_cadence_snr"] = ref_snr
        out["candidate_cadence_snr"] = (cand_tone / cand_floor
                                        if cand_floor > 0 else float("nan"))
        out["cadence_retained"] = cand_tone / ref_tone if ref_tone > 0 else float("nan")
        # BOTH sides, not just the reference. Gating on `ref_snr` alone was the
        # defect the PR #4 review caught: it passed rows whose *candidate* tone
        # was buried in its own noise -- train/subsample at 2 kS/s has candidate
        # SNR 0.504 -- and the report then quoted those ratios as measurements
        # when they are upper bounds on noise. A ratio is only as trustworthy
        # as its weaker term.
        cand_snr = out["candidate_cadence_snr"]
        out["cadence_resolvable"] = bool(ref_snr >= 3.0 and cand_snr >= 3.0)
        # Which side failed, so a reader does not have to compare two SNRs to
        # find out whether the candidate really lost the cadence or the
        # measurement simply could not see it.
        out["cadence_limited_by"] = (
            None if out["cadence_resolvable"]
            else ("reference" if ref_snr < 3.0 else "candidate"))
    return out
