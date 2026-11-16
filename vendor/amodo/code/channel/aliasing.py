"""Out-of-band energy, and the aliasing bound §3 check 1 turns on.

Spec task 3.2. Given a high-rate burst that spans the probe's whole passband,
this answers one question: if we retain samples at some lower rate with no
anti-alias filter at all, what fraction of the retained band's power was folded
into it from above its Nyquist?

The arithmetic
--------------
Sampling at `f_r` maps every frequency `f` to an alias in `[0, f_r/2]` by
reflecting repeatedly about multiples of `f_r/2`. Under a worst case of **zero
rejection** every component above `f_r/2` arrives in the retained band with its
power intact, so the folded power is simply the total power above `f_r/2` and
the bound is

    folded_fraction = P(f > f_r/2) / P(0 < f <= f_r/2)

That is the ratio §3 check 1 caps at 1%. It needs no model of where each
component lands, because the check is on the aggregate.

Why the DC bin is excluded from both
------------------------------------
The probe is an AC-coupled Rogowski coil with no DC response (§3), so the
channel's mean is instrumental -- scope and probe offset -- and is not signal.
It is also large enough to matter: the committed soak measures +45.1 counts
against an AC RMS of 0.284 A. Leaving it in the in-band term would inflate the
denominator and make the bound look better than it is, which is the wrong
direction for a check that is supposed to be conservative. Every estimator here
removes the mean first.

Why the noise floor is reported separately
------------------------------------------
A burst spanning 50 MHz collects 50 MHz of front-end and ADC noise, and that
noise folds in exactly as signal does. It is therefore *part of* the honest
aliasing bound -- the retained band really does receive it -- but it is not
"the out-of-band energy a real training workload produces", which is what task
3.2 names. Both figures are computed and reported: the total, which is what the
check gates on, and the workload's excess over a matched idle burst, which is
what the task's wording describes. If they disagree about the verdict, that is
a finding for the channel report, not something to resolve by choosing the
convenient one.

Units are amperes throughout, or the channel's raw counts. Never watts (§3).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class BandPower:
    """Power in and above a retained band, as measured from one burst.

    "Power" here is mean-square current, in A^2. It is a signal-processing
    power and not an electrical one -- there is no voltage channel and none of
    this is ever watts (§3) -- and it is named for what it is so the
    distinction does not have to be made twice.
    """

    retained_rate_hz: float
    retained_nyquist_hz: float
    burst_rate_hz: float
    burst_nyquist_hz: float
    in_band_a2: float
    out_of_band_a2: float
    # The bound §3 check 1 gates on: out-of-band power as a fraction of
    # in-band, under a worst case of zero rejection.
    folded_fraction: float
    in_band_rms_a: float
    out_of_band_rms_a: float

    def as_dict(self) -> dict:
        return asdict(self)


def welch_psd(x: np.ndarray, sample_rate_hz: float, nperseg: int = 1 << 16,
              noverlap: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """One-sided Welch PSD in A^2/Hz, with the mean removed per segment.

    `reader.amplitude_spectral_density` returns the square root of this and is
    the right thing for plotting against an RMS current. Band *power* wants the
    PSD itself, and taking it as the square of the ASD would be a needless
    round trip through a square root, so the estimator is here in the form this
    module integrates.

    The normalisation is the standard one: dividing by the window's sum of
    squares makes the result independent of both the window and the segment
    length, so a PSD computed with one `nperseg` integrates to the same band
    power as one computed with another. `test_channel.py` asserts exactly that.
    """
    x = np.asarray(x, dtype=np.float64)
    if noverlap is None:
        noverlap = nperseg // 2
    if x.size < nperseg:
        nperseg = 1 << int(np.floor(np.log2(max(x.size, 2))))
        noverlap = nperseg // 2
    step = nperseg - noverlap

    win = np.hanning(nperseg)
    win_norm = (win ** 2).sum()
    n_segments = 1 + (x.size - nperseg) // step

    acc = np.zeros(nperseg // 2 + 1)
    for i in range(n_segments):
        seg = x[i * step: i * step + nperseg]
        # Per segment, not once globally: a slow drift across the record
        # otherwise leaks into the lowest bins and inflates in-band power.
        seg = seg - seg.mean()
        acc += np.abs(np.fft.rfft(seg * win)) ** 2

    psd = acc / (n_segments * win_norm * sample_rate_hz)
    psd[1:-1] *= 2.0                      # one-sided
    freqs = np.fft.rfftfreq(nperseg, 1.0 / sample_rate_hz)
    return freqs, psd


def band_power(freqs: np.ndarray, psd: np.ndarray,
               low_hz: float, high_hz: float) -> float:
    """Integrate a one-sided PSD over (low_hz, high_hz], in A^2.

    The interval is half-open on purpose. `in_band` and `out_of_band` below
    share the retained Nyquist as a boundary, and a closed interval on both
    sides would count the bin sitting on it twice -- inflating the numerator
    and the denominator of the very ratio being gated.

    The rule is a rectangle sum, `sum(psd) * df`, and not a trapezoid. Two
    reasons, both of which bit during development:

    1. **It partitions exactly.** A trapezoid over the masked bins drops the
       one segment that straddles the boundary between two adjacent bands, so
       `in_band + out_of_band` came out short of the whole by a bin. For a
       ratio that is gated at 1% a seam at the boundary is precisely the wrong
       place to lose power.
    2. **It is the Parseval-exact rule here.** Each Welch bin already holds the
       density averaged over its resolution bandwidth, so the band power is the
       sum of the bins times the bin width. A trapezoid additionally
       half-weights the two end bins, which is a real error at a band edge and
       not a refinement.
    """
    if high_hz <= low_hz:
        return 0.0
    mask = (freqs > low_hz) & (freqs <= high_hz)
    if not mask.any():
        return 0.0
    df = float(freqs[1] - freqs[0])
    return float(psd[mask].sum() * df)


def folded_power(x_amperes: np.ndarray, burst_rate_hz: float,
                 retained_rate_hz: float, *,
                 nperseg: int = 1 << 16,
                 psd: tuple[np.ndarray, np.ndarray] | None = None) -> BandPower:
    """Bound the power folded into `retained_rate_hz` from a high-rate burst.

    `x_amperes` must come from a burst sampled fast enough that its own Nyquist
    is above anything the probe can deliver -- otherwise the out-of-band term
    is itself truncated and the bound is not a bound. `run_burst` defaults to
    100 MS/s against the probe's 30 MHz corner for that reason.

    Pass `psd` to reuse an estimate already computed; otherwise it is measured
    here.
    """
    if retained_rate_hz >= burst_rate_hz:
        raise ValueError(
            f"the retained rate ({retained_rate_hz:,.0f} Hz) must be below the "
            f"burst rate ({burst_rate_hz:,.0f} Hz) -- there is no out-of-band "
            f"region to measure otherwise")

    freqs, p = psd if psd is not None else welch_psd(x_amperes, burst_rate_hz,
                                                     nperseg=nperseg)
    retained_nyquist = retained_rate_hz / 2.0
    burst_nyquist = burst_rate_hz / 2.0

    # The DC bin is excluded by starting the in-band integral above 0 Hz; see
    # the module docstring.
    in_band = band_power(freqs, p, 0.0, retained_nyquist)
    out_of_band = band_power(freqs, p, retained_nyquist, burst_nyquist)

    return BandPower(
        retained_rate_hz=retained_rate_hz,
        retained_nyquist_hz=retained_nyquist,
        burst_rate_hz=burst_rate_hz,
        burst_nyquist_hz=burst_nyquist,
        in_band_a2=in_band,
        out_of_band_a2=out_of_band,
        folded_fraction=(out_of_band / in_band) if in_band > 0 else float("inf"),
        in_band_rms_a=float(np.sqrt(in_band)),
        out_of_band_rms_a=float(np.sqrt(out_of_band)),
    )


def excess_over_baseline(workload: BandPower, baseline: BandPower) -> dict:
    """The workload's own out-of-band energy, with the idle noise floor removed.

    Power adds for uncorrelated contributions, so the workload-attributable
    part of each band is the difference. A negative difference means the two
    bursts' noise floors differed by more than the workload contributed to that
    band, which is a real outcome at low signal and is reported as such rather
    than clamped to zero -- clamping would silently turn "we cannot resolve it"
    into "it is zero".

    The ratio returned is the workload's excess out-of-band power over its
    excess in-band power: what task 3.2 names, as distinct from the total the
    check gates on.
    """
    d_in = workload.in_band_a2 - baseline.in_band_a2
    d_out = workload.out_of_band_a2 - baseline.out_of_band_a2
    if d_in > 0:
        fraction = d_out / d_in
        resolvable = True
    else:
        fraction = float("nan")
        resolvable = False
    return {
        "excess_in_band_a2": d_in,
        "excess_out_of_band_a2": d_out,
        "excess_folded_fraction": fraction,
        # False when the workload did not raise in-band power above the
        # baseline burst's, which makes the ratio meaningless.
        "resolvable": resolvable,
        "baseline_in_band_a2": baseline.in_band_a2,
        "baseline_out_of_band_a2": baseline.out_of_band_a2,
    }


def rejection_from_resampling(burst_rms_a: float, retained_rms_a: float, *,
                              in_band_a2: float | None = None,
                              out_of_band_a2: float | None = None) -> dict:
    """How much out-of-band power the chain actually rejects, from the signal itself.

    A direct empirical test of the zero-rejection worst case, needing no
    calibrated source -- which matters, because §7 makes a SigGen sweep (task
    3.1) a non-goal for this block and the worst case is otherwise all the
    credit the chain gets.

    The argument: if the chain really rejected nothing above the retained
    Nyquist, then every component of the wideband signal would arrive in the
    retained band with its power intact, and total AC RMS would be **conserved**
    when the same physical signal is captured at the lower rate. Any RMS
    deficit at the retained rate is power that did not arrive -- i.e. rejection.

    Returns the conserved-power ratio and the implied broadband rejection in dB.
    A ratio near 1 confirms the worst case; a ratio well below 1 means the
    bound computed by `folded_power` is pessimistic by that factor.

    **Pass the band powers.** Without them this returns a number with no stated
    leverage, and the PR #4 review caught exactly that error in the channel
    report: the figure was quoted from a *training* burst, where out-of-band
    power is 0.3% of the total, so even total rejection of everything above the
    retained Nyquist would move the ratio by only -0.014 dB -- while the two
    measurements taken disagreed with each other by 0.14 dB. The statistic
    could not distinguish zero rejection from complete rejection, and was cited
    as evidence that rejection was zero. Given the band powers, this reports
    `full_rejection_db` (what the ratio would be if everything above the
    Nyquist were removed) and `resolvable`, which is false when the measurement
    has no power to tell those cases apart. The same comparison on an *idle*
    burst, where out-of-band is 93% of the total, has -11.5 dB of leverage and
    settles the question.

    The caveat travels with the number: this compares two captures taken at
    different times, so it is only meaningful when the GPU was in the same
    commanded state for both, and it measures the chain in aggregate rather
    than resolving rejection against frequency. It is not a substitute for the
    swept curve, and the channel report must not present it as one.
    """
    if burst_rms_a <= 0:
        raise ValueError("the burst RMS must be positive to form a ratio")
    ratio = (retained_rms_a ** 2) / (burst_rms_a ** 2)
    out = {
        "burst_rms_a": burst_rms_a,
        "retained_rms_a": retained_rms_a,
        "power_ratio": ratio,
        "implied_rejection_db": float(10.0 * np.log10(ratio)) if ratio > 0 else float("-inf"),
    }
    if in_band_a2 is not None and out_of_band_a2 is not None:
        total = in_band_a2 + out_of_band_a2
        full = in_band_a2 / total if total > 0 else float("nan")
        out["full_rejection_power_ratio"] = full
        out["full_rejection_db"] = (float(10.0 * np.log10(full)) if full > 0
                                    else float("-inf"))
        out["out_of_band_share"] = out_of_band_a2 / total if total > 0 else float("nan")
        # The measurement can only speak if the effect it is looking for is
        # larger than the scatter between repeats. 1 dB is the scale of the
        # disagreement observed between two nominally identical comparisons on
        # this chain, so an effect smaller than that is not resolvable here.
        out["resolvable"] = bool(np.isfinite(out["full_rejection_db"])
                                 and abs(out["full_rejection_db"]) > 1.0)
        # None, not False, when the comparison cannot resolve the effect.
        # `False` is a positive claim -- "the worst case is NOT tight", i.e.
        # the chain does reject -- and it is exactly the claim the leverage
        # check exists to refuse. The unresolvable training comparison would
        # otherwise sit in the artefact reading `worst_case_is_tight: false`
        # two lines from prose saying it establishes nothing.
        out["worst_case_is_tight"] = (bool(ratio > 0.5) if out["resolvable"]
                                      else None)
    else:
        out["resolvable"] = None
        out["worst_case_is_tight"] = None
    return out


def out_of_band_bands(freqs, psds_train, psds_idle, nyquist_hz,
                       window_hz=5e3, n=6):
    """Where above the retained Nyquist the workload changes the spectrum.

    Integrated over fixed windows, and across every available burst.

    The first version of this took the argmax bin of the training PSD and read
    the idle PSD at that same bin. That is invalid, and it inverted a result.
    A VRM switching line moves and broadens under load, so reading the quiet
    spectrum at the busy spectrum's peak compares a line centre against a line
    shoulder and cannot help but return a large ratio. It reported "a line at
    501.5 kHz that training lifts 6.7x"; integrating the same data over
    499-504 kHz shows training *reducing* that band by 37%, consistently across
    all three burst pairs. The peak location did not even reproduce -- 501.5 kHz
    in one burst, 505.7 and 505.8 kHz in the other two -- while the idle peak
    sat stably at 504.0 kHz.

    So: windows wide enough to contain a line wherever it has moved to, the
    same window for both conditions, and every burst rather than the first.
    `spread` is the ratio of the largest to the smallest single-burst ratio in
    that window; a band whose spread is comparable to its ratio is telling you
    about burst-to-burst scatter rather than about the workload.
    """
    edges = np.arange(nyquist_hz, freqs[-1], window_hz)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        t = np.array([band_power(freqs, p, lo, hi) for p in psds_train])
        i = np.array([band_power(freqs, p, lo, hi) for p in psds_idle])
        if not i.all():
            continue
        per_burst = t / i[:len(t)] if len(t) == len(i) else None
        rows.append({
            "low_hz": float(lo),
            "high_hz": float(hi),
            "train_a2": float(t.mean()),
            "idle_a2": float(i.mean()),
            # Amplitude ratio, to match how the ASD figure reads.
            "ratio": float(np.sqrt(t.mean() / i.mean())),
            "spread": (float(np.sqrt(per_burst.max() / per_burst.min()))
                       if per_burst is not None and per_burst.min() > 0 else None),
            "n_bursts": int(min(len(t), len(i))),
        })
    # Ranked by how much the workload moved the band, either way.
    rows.sort(key=lambda r: abs(np.log(r["ratio"])), reverse=True)
    return rows[:n]



def _provenance() -> dict:
    """The commit this rollup was computed at.

    Every per-run sidecar carries one; the rollups did not, and they are what
    the channel report and the spec's §11 ticks quote their numbers from. The
    estimators in `code/channel/` were rewritten several times during Phase 3,
    so a number without a commit cannot be told apart from a regression later
    (spec §5, and task 7.1's verification clause).
    """
    return {
        "git_commit": git_commit(),
        "environment_lock_sha256": environment_lock_sha256(),
        "working_tree_modified_paths": dirty_paths(),
    }


def line_core(freqs, psds_train, psds_idle, nyquist_hz, half_width_hz=2.5e3):
    """The strongest idle line above the Nyquist, and what the workload does to it.

    `_out_of_band_bands` lays a fixed grid and so can only report windows that
    happen to straddle a line. This pins the line itself: find the idle
    spectrum's peak above the Nyquist (idle, because at idle the line is narrow
    and stable), then integrate BOTH conditions over one window centred on it.

    It exists because the numbers describing the correction to the original
    501.5 kHz claim -- the idle peak frequency, the per-burst training argmax
    positions, and the core depletion -- were first computed ad hoc in a
    terminal. Quoting an unsourced number is exactly the failure being
    corrected, so they are emitted here instead.
    """
    sel = freqs > nyquist_hz
    f = freqs[sel]
    idle_peaks = [float(f[int(np.argmax(p[sel]))]) for p in psds_idle]
    train_peaks = [float(f[int(np.argmax(p[sel]))]) for p in psds_train]
    centre = float(np.median(idle_peaks))
    lo, hi = centre - half_width_hz, centre + half_width_hz
    t = np.array([band_power(freqs, p, lo, hi) for p in psds_train])
    i = np.array([band_power(freqs, p, lo, hi) for p in psds_idle])
    # The ratio at the idle peak bin and at the training peak bin, which is the
    # comparison that went wrong: reading one spectrum at the other's argmax.
    k_idle = int(np.argmin(np.abs(freqs - centre)))
    k_train = int(np.argmin(np.abs(freqs - train_peaks[0])))
    return {
        "idle_peak_hz_per_burst": idle_peaks,
        "train_peak_hz_per_burst": train_peaks,
        "window_hz": [lo, hi],
        "idle_a2_per_burst": [float(v) for v in i],
        "train_a2_per_burst": [float(v) for v in t],
        "idle_a2": float(i.mean()),
        "train_a2": float(t.mean()),
        "core_power_change": float(t.mean() / i.mean() - 1.0),
        "asd_ratio_at_idle_peak": float(np.sqrt(psds_train[0][k_idle]
                                                / psds_idle[0][k_idle])),
        "asd_ratio_at_train_peak": float(np.sqrt(psds_train[0][k_train]
                                                 / psds_idle[0][k_train])),
    }
