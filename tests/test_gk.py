"""The G&K corpus helpers (spec Phase 7): decimation and label parsing."""

import numpy as np
import pytest

import gk

N = gk.DOWN * 1000          # 0.32 s at FS_IN, an exact number of output samples
EDGE = 2000                 # output samples trimmed for filter transients


def tone(f_hz: float) -> np.ndarray:
    return np.sin(2 * np.pi * f_hz * np.arange(N) / gk.FS_IN)


def amplitude(y: np.ndarray) -> float:
    return float(np.sqrt(2) * y[EDGE:-EDGE].std())


def test_rate_is_exact():
    assert gk.FS_IN * gk.UP / gk.DOWN == gk.FS_OUT
    assert gk.N_OUT == 1_000_000
    assert len(gk.decimate(np.zeros(N, dtype=np.float16))) == N * gk.UP // gk.DOWN


def test_passband_tone_kept():
    assert abs(amplitude(gk.decimate(tone(1e3))) - 1) < 1e-3


def test_alias_tone_rejected():
    """A 150 kHz tone would alias to 50 kHz without the filter: >= 60 dB down."""
    assert 20 * np.log10(amplitude(gk.decimate(tone(150e3)))) <= -60


def test_alias_test_can_fail():
    """Naive slicing at the same overall rate lets the 150 kHz tone through."""
    x = tone(150e3)
    naive = x[np.round(np.arange(len(x) * gk.UP // gk.DOWN) * gk.DOWN / gk.UP).astype(int)]
    assert 20 * np.log10(amplitude(naive)) > -60


@pytest.mark.parametrize("label, family, kind, strategy, cls", [
    ("llama8b_training_full_bs8_seq512", "llama8b", "training_full", None, "training"),
    ("qwen4b_inference_bs8_seq512", "qwen4b", "inference", None, "inference"),
    ("olmo13b_training_accum2_bs4_seq512", "olmo13b", "training_accum", None, "training"),
    ("llama8b_training_lora_bs8_seq512", "llama8b", "training_lora", None, "training"),
    ("other_matmul_bf16", "matmul_bf16", "other", None, "non-ai"),
    ("llama8b_advlora_dilute_bs8_seq512", "llama8b", "adversarial", "lora_dilute", "training"),
    ("apertus8b_advchunked_opt_bs8_seq512", "apertus8b", "adversarial", "chunked_opt", "training"),
])
def test_parse_label(label, family, kind, strategy, cls):
    lab = gk.parse_label(label)
    assert (lab.family, lab.kind, lab.strategy, lab.cls) == (family, kind, strategy, cls)


@pytest.mark.parametrize("bad", ["llama8b_pretrain_bs8_seq512", "llama8b_advnovel_bs8_seq512"])
def test_parse_label_rejects_unknown(bad):
    with pytest.raises(ValueError):
        gk.parse_label(bad)


def test_paths():
    p = ("adversarial_10MSs_5s/power/apertus8b_advdilute_bs8_seq512/channels/power/"
         "shard_000000/capture_000000003.npy")
    assert gk.capture_key(p) == ("adversarial_10MSs_5s", "apertus8b_advdilute_bs8_seq512",
                                 "capture_000000003")
    assert gk.sidecar_paths(p)["records"] == (
        "adversarial_10MSs_5s/power/apertus8b_advdilute_bs8_seq512/records/"
        "shard_000000/capture_000000003.json")
