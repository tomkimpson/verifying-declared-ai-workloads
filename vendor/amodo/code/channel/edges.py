"""Recovering an interruption's boundaries from the trace (sub-spec 5C.4).

What a boundary looks like in this channel
------------------------------------------
The probe is a Rogowski coil, AC-coupled at ~34 Hz (parent §3). A step
change in the GPU's load is a step in the aux current, and the coil
answers a current step with a **transient**: a spike in the AC signal
decaying with the high-pass time constant. In the RMS envelope at 0.2 ms
frames, an optimizer step's end is a single ~15 A frame immediately
before the step boundary the CUDA events record, and the first kernel
after a pause is a rise within ~0.5 ms of it; a level crossing of a
coarser envelope lags the transient by a millisecond or more, which is
why the first version of this module, a change-point in mean of 1 ms
frames, put a checkpoint's start 1.2 ms late on a boundary the trace
marks to a fifth of that.

Two kinds of boundary, two coarse locators, two fine locators
--------------------------------------------------------------
- A **checkpoint** idles the GPU for a second or more (the copies and the
  disk write draw no aux current worth the name), so its boundaries are
  level changes. Coarse: the single change-point in mean of the 1 ms
  envelope inside a window around where the event log says the boundary
  is (`change_point`, the classical CUSUM split, no threshold). Fine: the
  **edge of the quiet plateau** at 0.2 ms -- the last active frame before
  it for a pause, the first active frame after it for a resume
  (`plateau_edge`).
- An **eval** is forward-only compute at a load close to training's, with
  no level change to find. What changes is the rhythm: the optimizer-step
  period stops. Coarse: the run's own **step template** -- the mean
  envelope over training steps far from any interruption, aligned at
  their start -- matched along the trace around the eval
  (`template_peaks`), keeping only matches that sit in a periodic train;
  the eval's start is the end of the last step matched before it and its
  end is the start of the first matched after. Fine: **both** boundaries
  come from re-matching the template at 0.2 ms around the coarse match
  (`template_refine`) -- the step before the eval and the step after --
  and the eval's start is that earlier step's refined start plus one
  period.

  Matching the whole step, rather than hunting the transient at its end,
  is what makes this work. The step's last frames and the eval's first
  are both load-step transients ~0.14 ms apart, and on two of the hour's
  ten evals the eval's own first forward pass peaked *higher* (13.1 A)
  than the step's end (8.3 A); a locator that takes the largest frame
  near the boundary takes the wrong one, and one restricted to look only
  backwards takes a frame inside the previous step, because the coarse
  estimate is itself up to 3 ms early. The template has a whole period of
  structure to align on and neither failure mode.

**What an eval's start can and cannot claim.** The step's end and the
eval's first forward pass are 0.14 ms apart, below the 0.2 ms frame, so
the two are not separately resolvable: what is recovered is the
*junction* between the last training step and the interruption, and it
is compared against the eval's recorded start because that is the
boundary the annotation names. A checkpoint's boundaries carry no such
caveat -- the GPU goes quiet for a second.

The template is built from steps clear of every interruption, and the
locators use the event log only to say roughly where to look, as the
marker recovery does. The envelope is taken at the acquisition rate
before any reduction, as Phase 3 requires (parent §3). Nothing here is in
watts (spec §3).
"""

from __future__ import annotations

import numpy as np

from channel.envelope import rms_envelope

COARSE_FRAME_S = 1e-3
FINE_FRAME_S = 0.2e-3
#: How far from a coarse estimate the fine locators look.
FINE_HALF_WINDOW_S = 3e-3
#: A frame this far above the quiet floor is active.
ACTIVE_ABOVE_FLOOR_A = 0.5
#: The coarse template match clips the envelope here: the pauses between
#: kernels (~0.3 A) carry the step's rhythm, the kernels' amplitude does not.
CLIP_A = 1.0
#: How far off a period a neighbour may sit and still count as the same
#: periodic train. `template_peaks` keeps only candidates that have one, which
#: is what makes a lone step -- one resuming after a marker hold, say -- an
#: unmatchable boundary rather than a wrongly matched one.
TRAIN_TOLERANCE = 0.02


def change_point(y: np.ndarray, *, min_side: int = 2) -> tuple[int, float]:
    """Index `k` splitting `y` into `y[:k]` and `y[k:]` with the largest
    CUSUM statistic sqrt(nl*nr/n) * |mean_left - mean_right|, and the
    contrast |mean_left - mean_right| / pooled within-side scatter."""
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n < 2 * min_side:
        raise ValueError(f"{n} frames cannot hold two sides of {min_side}")
    cs = np.cumsum(y)
    cs2 = np.cumsum(y * y)
    ks = np.arange(min_side, n - min_side + 1)
    nl, nr = ks, n - ks
    sl, sr = cs[ks - 1], cs[-1] - cs[ks - 1]
    ml, mr = sl / nl, sr / nr
    vl = np.maximum(cs2[ks - 1] / nl - ml ** 2, 0.0)
    vr = np.maximum((cs2[-1] - cs2[ks - 1]) / nr - mr ** 2, 0.0)
    pooled = np.sqrt((nl * vl + nr * vr) / n)
    stat = np.sqrt(nl * nr / n) * np.abs(ml - mr)
    i = int(np.argmax(stat))
    contrast = float(np.abs(ml[i] - mr[i]) / pooled[i]) if pooled[i] > 0 else float("inf")
    return int(ks[i]), contrast


def _window(y: np.ndarray, rate_hz: float, t_s: float, half_s: float) -> tuple[int, int]:
    lo = max(0, int(round((t_s - half_s) * rate_hz)))
    hi = min(y.size, int(round((t_s + half_s) * rate_hz)) + 1)
    if hi - lo < 3:
        raise ValueError("window holds fewer than three frames")
    return lo, hi


def plateau_edge(y: np.ndarray, rate_hz: float, *, t_coarse_s: float, quiet_after: bool,
                 half_window_s: float = FINE_HALF_WINDOW_S,
                 floor_span_s: float = 0.05) -> dict:
    """The edge of a quiet plateau near `t_coarse_s` in the envelope `y`
    (frame 0 at time 0).

    The floor is the median of the plateau side, `floor_span_s` long,
    starting clear of the window; a frame more than `ACTIVE_ABOVE_FLOOR_A`
    above it is active. A pause (`quiet_after`) ends at the end of the last
    active frame in the window; a resume begins at the start of the first.
    """
    lo, hi = _window(y, rate_hz, t_coarse_s, half_window_s)
    span = int(round(floor_span_s * rate_hz))
    if quiet_after:
        floor_seg = y[hi:hi + span]
    else:
        floor_seg = y[max(0, lo - span):lo]
    if floor_seg.size < 3:
        raise ValueError("no plateau beside the window to take a floor from")
    floor = float(np.median(floor_seg))
    thr = floor + ACTIVE_ABOVE_FLOOR_A
    active = np.flatnonzero(y[lo:hi] > thr)
    if active.size == 0:
        return {"edge_s": None, "floor": floor, "threshold": thr, "active_frames": 0}
    if quiet_after:
        k = int(active[-1]) + 1          # the load stopped at that frame's end
    else:
        k = int(active[0])               # the load began at that frame's start
    return {"edge_s": (lo + k) / rate_hz, "floor": floor, "threshold": thr,
            "active_frames": int(active.size),
            "peak_a_rms": float(y[lo:hi].max())}


def step_template(env: np.ndarray, rate_hz: float, step_starts_s, *,
                  period_s: float) -> np.ndarray:
    """Mean envelope over the given steps, aligned at their starts, one
    period long. `env`'s frame 0 is at time 0."""
    n = int(round(period_s * rate_hz))
    acc = [env[a:a + n] for t in step_starts_s
           if (a := int(round(t * rate_hz))) >= 0 and a + n <= env.size]
    if len(acc) < 3:
        raise ValueError(f"only {len(acc)} whole steps inside the envelope; need 3")
    return np.mean(acc, axis=0)


def _ncc(env: np.ndarray, template: np.ndarray) -> np.ndarray:
    t = (template - template.mean()) / (template.std() or 1e-12)
    m = t.size
    win = np.lib.stride_tricks.sliding_window_view(env, m)
    mu = win.mean(axis=1, keepdims=True)
    sd = win.std(axis=1, keepdims=True)
    sd[sd == 0] = 1e-12
    return ((win - mu) / sd * t).mean(axis=1)


def template_peaks(env: np.ndarray, rate_hz: float, template: np.ndarray, *,
                   period_s: float, threshold: float = 0.7,
                   train_tolerance: float = TRAIN_TOLERANCE,
                   clip_a: float | None = CLIP_A) -> list[dict]:
    """Where `template` matches `env` (normalised cross-correlation of the
    envelopes **clipped at `clip_a`**, so the match is carried by where the
    GPU pauses between kernels rather than by kernel amplitude -- a
    forward-only stretch's combs look like training's at 1 ms, its pauses
    do not): local maxima above `threshold` of the best that sit in a
    periodic train (another candidate one period away within
    `train_tolerance`), then the best of any that remain within 0.6
    periods of each other. Times are of the template's start; `env`'s
    frame 0 is at time 0."""
    if env.size < template.size + 2:
        return []
    if clip_a is not None:
        env, template = np.minimum(env, clip_a), np.minimum(template, clip_a)
    ncc = _ncc(env, template)
    best = float(ncc.max())
    left = np.r_[-np.inf, ncc[:-1]]
    right = np.r_[ncc[1:], -np.inf]
    idx = np.flatnonzero((ncc >= threshold * best) & (ncc >= left) & (ncc >= right))
    starts = idx / rate_hz
    in_train = [i for i, t in zip(idx, starts)
                if (np.abs(np.abs(starts - t) - period_s) <= train_tolerance * period_s).any()]
    sep = 0.6 * period_s
    keep: list[tuple[int, float]] = []
    for i in sorted(in_train, key=lambda i: -ncc[i]):
        if all(abs(i - j) / rate_hz >= sep for j, _ in keep):
            keep.append((int(i), float(ncc[i])))
    return sorted(({"start_s": i / rate_hz, "ncc": c} for i, c in keep),
                  key=lambda p: p["start_s"])


def template_refine(env_fine: np.ndarray, rate_hz: float, template_fine: np.ndarray, *,
                    t_coarse_s: float, half_window_s: float = FINE_HALF_WINDOW_S) -> dict:
    """Re-match the template at the fine frame within `half_window_s` of a
    coarse match; the best start is the step's start."""
    m = template_fine.size
    lo = max(0, int(round((t_coarse_s - half_window_s) * rate_hz)))
    hi = min(env_fine.size - m, int(round((t_coarse_s + half_window_s) * rate_hz)))
    if hi <= lo:
        raise ValueError("no room to refine the match")
    ncc = _ncc(env_fine[lo:hi + m], template_fine)
    k = int(np.argmax(ncc))
    return {"edge_s": (lo + k) / rate_hz, "ncc": float(ncc[k])}


def recover_level_boundary(x_amperes: np.ndarray, sample_rate_hz: float, *,
                           window_start_s: float, expected_s: float) -> dict:
    """A checkpoint boundary: coarse by the level change-point, fine by the
    plateau's edge. `x_amperes` is the trace from scope time
    `window_start_s`."""
    env_c, rate_c = rms_envelope(x_amperes, sample_rate_hz, frame_s=COARSE_FRAME_S)
    k, contrast = change_point(env_c)
    before, after = float(env_c[:k].mean()), float(env_c[k:].mean())
    quiet_after = after < before
    t_coarse = window_start_s + k / rate_c
    env_f, rate_f = rms_envelope(x_amperes, sample_rate_hz, frame_s=FINE_FRAME_S)
    fine = plateau_edge(env_f, rate_f, t_coarse_s=t_coarse - window_start_s,
                        quiet_after=quiet_after)
    t = None if fine["edge_s"] is None else window_start_s + fine["edge_s"]
    return {
        "method": "level change-point at 1 ms, then the plateau's edge at 0.2 ms",
        "kind_of_edge": "pause" if quiet_after else "resume",
        "coarse_scope_s": float(t_coarse), "coarse_contrast": contrast,
        "level_before_a_rms": before, "level_after_a_rms": after,
        "plateau_floor_a_rms": fine["floor"], "active_threshold_a_rms": fine["threshold"],
        "recovered_scope_s": t, "expected_scope_s": float(expected_s),
        "offset_s": None if t is None else float(t - expected_s),
        "resolvable": t is not None and fine["active_frames"] > 0,
        "frame_s": FINE_FRAME_S,
    }


def recover_rhythm_boundaries(x_amperes: np.ndarray, sample_rate_hz: float, *,
                              window_start_s: float, expected_start_s: float,
                              expected_end_s: float, template_coarse: np.ndarray,
                              template_fine: np.ndarray, period_s: float,
                              min_ncc: float = 0.3) -> dict:
    """An eval's two boundaries from the step train around it. `x_amperes`
    spans the eval with a few step periods either side."""
    env_c, rate_c = rms_envelope(x_amperes, sample_rate_hz, frame_s=COARSE_FRAME_S)
    env_f, rate_f = rms_envelope(x_amperes, sample_rate_hz, frame_s=FINE_FRAME_S)
    peaks = template_peaks(env_c, rate_c, template_coarse, period_s=period_s)
    starts = np.array([window_start_s + p["start_s"] for p in peaks])
    ends = starts + period_s
    before = ends[ends <= expected_start_s + 0.5 * period_s]
    after = starts[starts >= expected_end_s - 0.5 * period_s]
    inside = starts[(starts > expected_start_s - 0.5 * period_s)
                    & (starts + period_s < expected_end_s + 0.5 * period_s)]
    out = {"method": ("step template matched at 1 ms in a periodic train, then re-matched at "
                      "0.2 ms on the step before the eval (its start plus one period is where "
                      "training stopped) and on the step after it"),
           "n_matched_steps": len(peaks), "steps_matched_inside_eval": int(inside.size),
           "matched_step_starts_s": starts.tolist(), "start": None, "end": None}
    if before.size:
        # The last step matched before the eval; its refined start plus one
        # period is where training stopped.
        t_coarse = float(before.max())
        fine = template_refine(env_f, rate_f, template_fine,
                               t_coarse_s=t_coarse - period_s - window_start_s)
        t = window_start_s + fine["edge_s"] + period_s
        out["start"] = {"coarse_scope_s": t_coarse, "recovered_scope_s": float(t),
                        "expected_scope_s": float(expected_start_s),
                        "offset_s": float(t - expected_start_s),
                        "fine_ncc": fine["ncc"], "resolvable": fine["ncc"] > min_ncc,
                        "frame_s": FINE_FRAME_S}
    if after.size:
        t_coarse = float(after.min())
        fine = template_refine(env_f, rate_f, template_fine, t_coarse_s=t_coarse - window_start_s)
        t = window_start_s + fine["edge_s"]
        out["end"] = {"coarse_scope_s": t_coarse, "recovered_scope_s": float(t),
                      "expected_scope_s": float(expected_end_s),
                      "offset_s": float(t - expected_end_s),
                      "fine_ncc": fine["ncc"], "resolvable": fine["ncc"] > min_ncc,
                      "frame_s": FINE_FRAME_S}
    return out
