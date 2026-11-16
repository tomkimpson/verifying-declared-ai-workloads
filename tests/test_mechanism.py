import numpy as np

import mechanism as mech

RATE = 500.0  # 2 ms frames


def fake_envs(locked_amp, n_steps=300, period_s=0.4, seed=0):
    """One interval whose envelope has a step-locked bump of `locked_amp` at
    each step end, plus white noise; returns (envs, step ends)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(n_steps * period_s * RATE)) / RATE
    ends = np.arange(1, n_steps) * period_s
    phase = (t % period_s) / period_s
    env = 1 + locked_amp * np.cos(2 * np.pi * phase) + 0.2 * rng.standard_normal(len(t))
    return [(0.0, float(t[-1]), env, RATE)], ends


def test_profile_cv_of_a_flat_profile_is_zero():
    assert mech.profile_cv(np.ones((10, 50))) == 0.0


def test_step_locked_structure_separates_from_its_null():
    for amp, expect in ((0.3, True), (0.0, False)):
        envs, ends = fake_envs(amp)
        half = 0.2
        times = mech.step_times(envs, ends, half)
        segs = mech.cut(envs, times, half)
        rng = np.random.default_rng(1)
        null = [mech.profile_cv(mech.cut(envs, mech.random_times(envs, [len(times[0])], half, rng), half))
                for _ in range(50)]
        assert bool(mech.profile_cv(segs) > np.quantile(null, 0.95)) == expect


def test_cv_interval_contains_its_estimate():
    envs, ends = fake_envs(0.05)
    segs = mech.cut(envs, mech.step_times(envs, ends, 0.2), 0.2)
    r = mech.cv_with_ci(segs, n_boot=300)
    assert r["ci95"][0] <= r["cv"] <= r["ci95"][1]


def test_long_record_snr_separates_5p00_from_5p21_hz():
    rate, n = 500.0, int(176 * 500)
    t = np.arange(n) / rate
    rng = np.random.default_rng(0)
    env = 1 + 0.3 * np.sin(2 * np.pi * 5.21 * t) + 0.1 * rng.standard_normal(n)
    f, p = mech.envelope_spectrum(env, rate)
    assert mech.line_snr(f, p, 5.21)[0] > 100
    assert mech.line_snr(f, p, mech.HARNESS_HZ)[0] < mech.LINE_ABSENT_SNR
    env2 = env + 0.05 * np.sin(2 * np.pi * 5.0 * t)
    f, p = mech.envelope_spectrum(env2, rate)
    snr, f_peak = mech.line_snr(f, p, mech.HARNESS_HZ)
    assert snr > 50 and abs(f_peak - 5.0) < 0.005


def test_demodulate_removes_the_harness_line_and_keeps_another():
    fs, dur = 20_000.0, 60.0
    t = np.arange(int(fs * dur)) / fs
    rng = np.random.default_rng(1)
    carrier = rng.standard_normal(len(t))
    x = carrier * (1 + 0.2 * np.sin(2 * np.pi * 5.0 * t)) * (1 + 0.2 * np.sin(2 * np.pi * 5.21 * t))

    def snrs(sig):
        env, rate = mech.corpus.rms_envelope(sig - sig.mean(), fs)
        f, p = mech.envelope_spectrum(env, rate)
        return mech.line_snr(f, p, 5.0)[0], mech.line_snr(f, p, 5.21)[0]

    before, after = snrs(x), snrs(mech.demodulate(x, fs))
    assert before[0] > 100 and after[0] < mech.LINE_ABSENT_SNR
    assert after[1] > 0.8 * before[1]


def test_within_step_half_ignores_a_step_gap_level_change():
    # A run that is high during each step and low in the gap after it: the
    # whole-window CV sees the level change, the before-end half does not.
    n = 2000
    profile = np.r_[np.full(n // 2, 1.6), np.full(n // 2, 1.0)]
    segs = np.tile(profile, (5, 1))
    assert mech.profile_cv(segs) > 0.2
    assert mech.profile_cv(segs[:, : n // 2]) < 1e-12
