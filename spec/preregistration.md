# Pre-registration: hidden-training-current-monitoring

**Spec task:** 3.1 (`spec/spec.md` §11, Phase 3)
**Written:** 2026-10-02, before any detector or summary has read a test-split window.
**Revised:** 2026-10-02, after the PR #3 review and still before any test window was
scored. Earlier freezes (`912eb3b`, `0c2e058`, `01d50bd`) are superseded; spec §12 amendment
2026-10-02d lists the changes.
**Corpus:** amodo-power-experiments at `73ffebe`, archive `/Volumes/Extreme SSD/amodo-raw-export`.
**Frozen R1:** `results/declaration/choice.json` (task 2.4), scored once by the kill check
`results/declaration/killcheck.json` (task 2.5; verdict: does not fire).

This file fixes every choice that touches test data, for Phases 4–7. After it is
committed, a change here is an amendment: it goes in spec §12 and is reported in the
paper as a deviation from the pre-registration, with the reason. The loader refuses
to read test-split windows unless this file's SHA-256 matches the constant in
`code/prereg.py` (task 3.2).

Current, never power. Nothing derived from the probe is reported in watts or joules.

## 0. What has touched test data before this file

- **Task 1.4** windowed every split, test included: `results/windows/windows-hop{5,1}s-test.jsonl`
  (gitignored) holds the 25 features, `ac_rms_a` and step labels of each test window.
  `results/windows/manifest.json` (committed) lists the test captures with window
  counts and label counts. No detector, threshold or summary statistic has been
  computed on those windows.
- **Corpus metadata** (run IDs, conditions, scored durations, episode counts) was
  read when the spec was written. Test holds, in scored hours from the manifest
  (nominal hours in brackets): idle 8.35 h (8.5; 4 runs), `serving-decode` 0.475 h
  (0.5; 2 runs), non-AI busy 0.950 h (1.0; `steady-gemm` and `memory-bound`, 2 runs
  each), `training` and `lora` (2 runs each), and 7 time-shared episode runs.
- **Episode labels.** amodo `results/panel/episodes-recovery.json` was revised after
  its authors viewed the test-day episodes (amodo `code/dataset/splits.py`). The
  revision fixes episode boundaries (labels), not any detector. §3 check 5 confirms
  369/369 reachable boundaries within 1 ms.
- **Kill check (2.5)** scored dev hidden-training windows only.

## 1. Windows and labels

- **Window:** 5 s on the raw 200 kS/s waveform, mean removed, featurised by amodo
  `code/features.py` at the pinned commit (`code/windows.py`).
- **Hop:** 5 s for snapshot metrics (H1, H2, S2); 1 s for the monitor (H3).
- **Features:** the 25 amplitude-free replication features listed in
  `results/declaration/choice.json` → `frozen.features`. No amplitude feature in any
  headline detector. R1a is not used (2.4 chose R1).
- **Window label:** `training` iff ≥ 1 complete optimizer step (both `step` edges
  inside the window); `partial` if a step overlaps but none is complete; else `none`.
- **Per-window detection rate:** the fraction of `training` windows flagged.
  `partial` windows are excluded from it and reported as a count.
- **Decision time:** the window's end (`t1_s`). Every delay includes the 5 s
  observation.
- **Failed captures** are excluded by `capture_dir` (corpus manifest), never by
  `run_id`.

## 2. Detectors

### 2.1 Two-stage forest (levels 1 and 2)

- The G&K recipe as reimplemented in amodo `scripts/classify_replication.py`
  (`code/detectors.py::TwoStageForest`). Stage 1: AI against non-AI. Stage 2, on AI
  rows only: training against inference. A row stage 1 calls non-AI is non-AI.
- Each stage: `RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)`,
  all other settings sklearn 1.7.1 defaults. Fixed a priori, never tuned.
- **Level 1 (replica):** fitted on the 300 replication traces (`fit_replica`).
- **Window classes for fitting on corpus windows (level 2).** The spec does not
  state them; this fixes them:

  | Window | Class |
  |---|---|
  | label `training`, any condition | training |
  | label `partial`, any condition | not used in fitting |
  | label `none` in `serving-decode`, `serving-prefill` | inference |
  | label `none` in a `mixture-*` run (the serving process is running) | inference |
  | label `none` in `idle`, `steady-gemm`, `memory-bound`, `periodic-gemm-*` | non-AI |
  | label `none` in any other run (honest, jitter, padding, power cap) | not used in fitting |

  `idle` is non-AI, as in the replication's non-AI class (no AI process running).
- **Level 2 for H1 (strategy held out, all days).** For each held-out group in
  {time-sharing, co-location, jitter, padding, power cap}: fit on honest and every
  other strategy group from all three splits, plus negatives from development and
  calibration only. Score the held-out group's windows on every split where it exists.
- **The all-splits fit takes test-day time-shared windows.** Under the class map, the
  `none` windows of the test `panel-episodes-timeshared-*` runs train as inference,
  and their `training` windows as training. This follows D2 (strategies from all
  days; only negatives restricted to dev and cal). It is confined to H1: H2 and H3
  fits are dev only.
- **"Seen" reference for H1 (task 4.1).** For each strategy group, level 2 with that
  group *included* but the scored run held out (leave-one-run-out within the group).
  Padding has one run (`panel-cadence-pad-half`), so it has no seen reference. T1
  reports n/a for it rather than an unseen number relabelled.
- **H1 flag:** `predict(x) == "training"` (the G&K format; no calibration threshold).
- **Level 2 for H2 and H3 (dev only).** The same recipe and class map, fitted on
  development windows only (negatives included), with the scored strategy group held
  out. For H3 the scored group is **time-sharing** (all test episode runs are
  `mixture-timeshared-*`), and the H3 false-alarm backgrounds are scored by that same
  fit.
- **H2/H3 classifier score:** `p_training` (stage-2 P(training) on rows stage 1 calls
  AI, 0 elsewhere).

### 2.2 R1, the declaration test

Frozen in Phase 2 and not changed here (`results/declaration/choice.json`):

- isolation forest (`n_estimators=500, random_state=0, n_jobs=-1`) on the 25 features;
  score `-score_samples`;
- fitted on dev declared windows of `panel-core-idle-r1`,
  `panel-core-serving-decode-r1`, `panel-serving-decode`, `panel-serving-prefill`;
- α = 0.05; threshold 0.6289591641951192.

## 3. Thresholds (H2, H3, S2)

- **Shared target α = 0.05.** α names the rule, not an achieved rate (amendment
  2026-10-02b).
- **Rule, both detectors:** the (1 − α) quantile (`numpy.quantile`, default linear
  interpolation) of the detector's score on the 5 s-hop calibration declared windows
  (`panel-core-idle-r4`, `panel-core-serving-decode-r4`, pooled). A window is flagged
  iff its score is **strictly greater** than the threshold. For the classifier this
  means `p_training > t`; if the calibration quantile is 0 (many calibration windows
  have P(training) = 0), a window is flagged iff `p_training > 0`.
- **Primary rule only. No fallback for either detector.** The cross-validated
  fallback of D3/§10 is not used: R1's rule was frozen by 2.5, and the classifier
  takes the same rule so the comparison matches rules. Each classifier's threshold
  stability under a run-level bootstrap of the calibration runs is reported, not
  acted on.
- **Each classifier fit has its own threshold** (one per held-out group), computed
  once from calibration windows and written to JSON before any test window is scored
  by it.
- The H3 monitor reuses these thresholds on 1 s-hop windows; they are not recomputed
  at the 1 s hop.

## 4. H2: matched-rule comparison (task 5.1)

- Detectors: R1 and the dev-only level-2 classifier with the scored group held out.
- Per hidden-training configuration (each run of each strategy group, and the group
  pooled over windows): per-window detection at each detector's threshold, with its
  split. Time-sharing is reported for dev runs and test runs separately, then pooled.
  Co-location is dev-only attack data, never fitted or selected on by R1; it is never
  called day-held-out. Intervals by run count, as §8 (per run: block bootstrap;
  pooled over ≥ 3 runs: run-level; over 2 runs: stratified block).
- Achieved false-positive rate on **test** declared windows, per workload
  (`serving-decode`, `idle`), per detector, with 95% intervals by run count, as §8
  (decode, 2 runs: stratified block; idle, 4 runs: run-level). The classifier's headline row is the co-location-held-out fit (the
  one behind X′). The other four fits' rates go in an appendix table.
- The headline: Z (R1, co-location) against X′ (classifier, co-location held out).
- Test ROC for both detectors: descriptive only. No operating point is read off it.

## 5. H3: the monitor (tasks 5.2, 5.3)

- **Input:** 1 s-hop window flags of one capture, in time order, with all its scored
  intervals concatenated. The monitor's state (the last 5 flags and the alarm)
  **carries across the breaks between scored intervals** (about 5 s every 180 s) and
  resets only at the start of a capture. So a detector stuck positive raises exactly
  one onset per capture.
- **Alarm onset:** the end time of the window that makes 3 of the last 5 windows
  positive while no alarm is active.
- **Re-arm:** the alarm clears at the end time of the 5th consecutive negative
  window.
- **Detectors:** R1, the classifier (§2.1, time-sharing held out), and their OR
  (window positive iff either is). **R1 is the headline detector** for F, A, P, D
  and Lmin; the classifier and OR are reported beside it.
- **Backgrounds (false alarms):** test declared inference (`serving-decode`, 0.475 h
  scored), test idle (8.35 h), test non-AI busy (`steady-gemm` + `memory-bound`,
  0.950 h).
  Reported separately; never pooled across backgrounds.
- **F:** onsets per monitored hour. Exposure E is the summed scored duration
  (`scored_s` in the window manifest) of the background's captures.
  Bound: exact one-sided 95% upper Poisson (Garwood) bound on the onset count,
  divided by E, labelled "Poisson model". Where a background has ≥ 10 onsets, a
  block bootstrap (§8, onset blocks) is reported beside it.
- **A:** the fraction of a background's 1 s-hop windows after which the alarm is
  active, with a 95% block-bootstrap interval (§8, onset blocks). With about 830
  windows per short run, these intervals are approximate and under-cover; the paper
  says so. §9 reads point estimates only.
- **D's interval:** per cell, 95% percentile interval of the median delay under a
  circular block bootstrap over consecutive detected-by-onset episodes in the run
  (blocks of 4), 2000 resamples, seed 0. A cell with fewer than 8 detected-by-onset
  episodes reports its delays' range and no interval.
- **Episodes:** from `events` (`step` edges) per §3 check 5; 192 in the 7 test
  episode runs. An episode runs from its first to its last step edge. Its window is
  W = [first step edge, last step edge + 5 s] (the 5 s grace).
- **Cells.** A cell is a (nominal duration, duty) pair. Each test episode run has one
  nominal duration, the run's median episode duration (5.1, 10.3, 15.1, 60.1 or
  240.2 s), and one duty. So each of the 7 cells is exactly one run, from one
  session.
- **Episode classes.** Every episode gets exactly one class, tested in this order:
  1. *Truncated:* duration < 0.9 × the run's nominal duration (e.g. episodes cut short
     by the run's start or end, such as the 33 s episode in the 240 s run). Counted
     and reported. Excluded from P, D and Lmin.
  2. *Unscorable:* no 1 s-hop window overlaps the episode (it falls wholly in a
     break between scored intervals). Counted, reported, excluded.
  3. *Pre-alarmed:* the alarm is active at the first step edge, meaning the latest
     onset ≤ that edge has not cleared by it. No delay.
  4. *Detected by onset:* an alarm onset falls in W. Delay = the first such onset −
     first step edge. One onset may detect two adjacent episodes when their
     windows overlap (gaps under 5 s at duty 0.75). Each episode is judged on its
     own.
  5. *Missed:* everything else, including episodes too short to hold a
     complete-step window.
  - truncated + unscorable + pre-alarmed + detected by onset + missed = 192.
- **Operational metric (headline):** P = (pre-alarmed + detected by onset) /
  (pre-alarmed + detected by onset + missed), per cell. An alarm that is active
  anywhere in W counts as detected. This matters because at duty 0.75 the gaps
  between episodes (≈ 4.4 s) hold no whole 5 s window, so even a perfect detector's
  alarm never clears and most episodes begin pre-alarmed.
- **Gap guard on the pre-alarmed credit.** Crediting pre-alarmed episodes is safe
  only if the detector is quiet between episodes. For each detector and each
  episode run, G is the detector's per-window flag rate on that run's 1 s-hop
  windows labelled `none` (windows wholly inside gaps, which hold only the serving
  process). The guard is **per run**, so per cell. If **G ≤ 0.20**, P as above is
  that cell's headline. If G > 0.20, the onset-only metric is that cell's headline,
  and P with pre-alarmed credit is reported as unreliable. The duty-0.75 run has no
  `none` windows, so its G is the pooled G of the other six runs, and the paper
  says so. Lmin uses each cell's headline.
- **Onset-only metric (secondary, or headline under the gap guard):** detected by
  onset / (pre-alarmed + detected by onset + missed), the rule as first drafted,
  always reported beside P.
- **Restricted metric:** the cell's headline metric (P, or onset-only under the gap
  guard) over episodes for which ≥ 1 1 s-hop window contains a complete step *of
  that episode* (both edges of one of its steps inside the window).
- **D:** per cell, the median delay over episodes detected by onset; pre-alarmed
  episodes contribute none. A pooled median over all cells is secondary.
- **Within-run lower bound** per cell (not cluster-aware: every cell is one run, so
  no between-run variance enters it, and the paper says so). It is the minimum of
  (i) the exact one-sided 95% Clopper–Pearson lower bound on P's numerator over its
  denominator, and (ii) the 5th percentile of a circular block bootstrap over
  consecutive episodes in the run (blocks of 4), 10 000 resamples, seed 0.
- **Lmin:** the shortest nominal duration d such that every cell with nominal
  duration ≥ d has a headline (gap-guarded) within-run lower bound ≥ **0.5**. If none, "not
  established". Duty effects are claimed at 15.1 s only.

## 6. H1 intervals (task 4.1)

Per run: circular block bootstrap of the per-window rate, blocks of 12 windows (60 s
at 5 s hop). Per group: the pooled rate with the interval §8 assigns by run count
(run-level bootstrap for ≥ 3 runs; within-run stratified for 2), always beside the
per-run rates. 2000 resamples, seed 0. As `code/declaration.py`.

## 7. S2: TOST on the flag rate (task 6.1)

- Arms: `training` arm 1, `training-discarded`, `training-copying`. **All windows** of
  each run, `partial` included. This is the flag rate a monitor sees. Restricting to
  `training`-labelled windows would make the positive control below unable to fail
  for R1, which flags those windows at 1.0 in both arm 1 and the control
  (`results/declaration/killcheck.json`).
- Detectors: the classifier and R1 at their frozen thresholds (§3). All three arms
  are development runs (`panel-training-arm1`, `-arm2`, `-arm3`), so the classifier
  is the dev-only fit with the time-sharing group **and those three runs** held out,
  thresholded by the §3 rule. No arm is scored by a model fitted on it, and the
  positive control's group is unseen.
- S2 reads no test data. It is pre-registered because its δ and procedure fix the
  claim. **R1's S2 outcome is already known**: `killcheck.json` gives R1 a flag rate
  of 1.0 on all windows of arm 1, arm 2 and arm 3. Its TOST difference is therefore
  0 with zero width ("equivalent"), and its control (1.0 against 0.665) rejects. The
  paper says R1's S2 result was visible before this file. The classifier's half is
  blind.
- Statistic (S2): difference in per-window flag rate, genuine − discarded. Secondary:
  genuine − copying (the equal-cost control), by the same procedure.
- δ = 0.05 absolute, the same for every detector.
- Equivalence iff the 90% interval of the difference (two one-sided tests at 5%) lies
  inside (−δ, δ). Interval: independent circular moving-block bootstraps of each arm
  (12-window blocks), 10 000 resamples, percentile. Each run gets its own RNG stream:
  seed 0 for arm 1, 1 for discarded, 2 for copying, 3 for the control run.
- n = 1 run per arm: the test covers within-run variability only, and the paper says
  so.
- Positive control, **per detector**: genuine `training` arm 1 against
  `panel-mixture-timeshared-0p5` (dev) by the same procedure must reject
  equivalence. A detector whose control does not reject has its S2 result reported
  as uninformative. Other detectors are unaffected.

## 8. Bootstrap units and seeds

- Never by window. Which interval is primary is fixed by the run count, never chosen:
  - **One run:** circular block bootstrap within the run.
  - **A rate pooled over ≥ 3 runs:** run-level bootstrap (runs drawn with replacement,
    windows pooled; `run_bootstrap_ci`). It carries between-run variation, which
    matters when runs disagree (e.g. co-location's three runs).
  - **A rate pooled over 2 runs:** a run-level bootstrap is degenerate (the range of
    two rates; Phase 2), so the interval is the **stratified circular block
    bootstrap** (blocks drawn within each run, each run keeps its window count,
    rate pooled). It is labelled "within-run variation only".
  - Every pooled rate is reported beside its per-run rates, so between-run spread is
    always visible.
- Per-window rate blocks: 12 windows at the 5 s hop, 60 at the 1 s hop (60 s).
- Onset blocks (F and A): 600 windows of the 1 s hop (10 min), over the alarm-state
  sequence computed once on the full capture. A resample counts the onsets and
  alarmed windows inside its blocks; F = onsets / resampled exposure.
- 2000 resamples for intervals (as Phase 2), 10 000 for the episode lower bound and
  the TOST. Seed 0 except where a section names others.

## 9. Kill and finding thresholds (spec §8)

- **R1 trivial or useless:** co-location detection < 0.50 at α ≤ 0.05, or like-for-like
  held-out dev declared false positives > 3α. *Already applied in 2.5: does not fire*
  (co-location 1.000). The all-run 3α outcome (fires) is reported beside it.
- **Test false-alert rate unusable (H3):** if every detector's monitor on test
  declared inference exceeds **10 onsets/h** or spends more than **10%** of monitored
  time in alarm, H3 reports that as the finding. It does not stop the paper. Both are
  judged on the **point estimates** (F = onsets / E; A as §5), not on their bounds.
- **H1 harness artefact:** decided by Tom on 4.2's evidence (spec §8). 4.2's mechanism
  check and 5 Hz item use dev co-location data only and are not pre-registered here.
- **T5 transfer gate (D6):** applied to each G&K forest separately (the **hardened**
  forest, trained on their genuine and adversarial corpus, is T5's headline; the
  genuine-only forest is secondary). A forest's co-location number is interpreted
  only if that forest passes. On point estimates, both must hold:
  - our honest `training` windows (label `training`, all splits) flagged as training
    at ≥ **0.80**;
  - our declared inference windows (`serving-decode` and `serving-prefill`, all
    splits) flagged as training at ≤ **0.20**.

  If either fails for a forest, T5 reports a transfer failure for it, and its
  co-location number is printed as uninterpretable or not at all. The second clause stops a forest that calls everything training from
  passing.
- **T5 sanity check:** G&K's reported genuine accuracy reproduced within their stated
  range on their own data, before scoring ours (task 7.1).

## 10. Robustness (T4, task 5.4)

Each varies one named parameter and reuses the 5.1–5.3 code paths:

- mis-specified declaration: R1 refitted on dev `serving-decode` only, same rule, with
  prefill as stress; Phase 2's unseen-workload hold-outs (`serving-prefill`, `idle`)
  reported here;
- α ∈ {0.01, 0.02, 0.05, 0.10};
- persistence 2-of-3, 3-of-5, 4-of-5, 5-of-5 (re-arm unchanged);
- CUSUM on scores, sensitivity only.

No T4 variant replaces a headline number.

## 11. What is not pre-registered

- Phase 4.2's mechanism analysis and its 5 Hz item (dev co-location only).
- Phase 7's data handling (download, anti-alias filter and decimation, extraction of
  the 25 shared features, the forests' training on G&K data). These touch none of our
  windows. The gate above, its inputs and its levels are fixed here, and Phase 7
  scores none of our windows before the G&K own-data check of task 7.1 passes.
- Figure styling.
- Anything descriptive (ROC curves, score distributions), labelled as such.

## Sign-off

- [x] Approved by Tom Kimpson — date: 2026-10-02 — note: approved in session ("ok go")
  on the draft at `ca60e89`, after the six choices beyond the briefed defaults
  (level-2 class map, episode lower bound, R1 as H3 headline, H3 classifier holding
  out time-sharing, S2 arms held out of the classifier, §0 disclosure) were put to
  him. Recorded by Claude at his instruction. *(First freeze, `912eb3b`.)*
- [x] Re-approved by Tom Kimpson — date: 2026-10-02 — note: approved in session
  ("fix then merge") on the PR #3 review's proposed fixes: the H3 episode rule,
  monitor state across interval breaks, S2's window set and per-detector control,
  the within-run bound, truncated episodes, point-estimate thresholds, stratified
  block-bootstrap intervals, and the two-sided T5 gate. Recorded by Claude at his
  instruction. *(Second freeze, `0c2e058`.)* Round 3 of the review then changed
  the text under the same "fix then merge" instruction, without Tom reviewing it
  item by item: intervals fixed by run count (§6, §8), the gap guard on
  pre-alarmed credit (§5), D per cell, the restricted metric's window, E's
  definition, R1's S2 disclosure, and the per-forest T5 gate. *(Third freeze,
  `01d50bd`.)* A second "fix then merge" from Tom authorised round 4, which
  aligned §4's intervals with §8 and made the gap guard per run, again without
  item-by-item review. This version is the one the guard's hash freezes. Tom may amend it via spec §12 before any test
  window is scored.
