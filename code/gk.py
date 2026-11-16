"""The Gargiulo & Kulp corpus on HuggingFace (spec Phase 7, D6).

Their traces are 5 s of AC-coupled probe current, float16 at 9.765625 MS/s
(48,828,125 samples), from the same probe model as ours. `decimate` takes a
trace to 200 kS/s by the exact rational resample 64/3125, whose polyphase FIR
is the anti-alias filter (scipy's Kaiser window, cutoff at the output
Nyquist). Our captures run at 199,808 S/s; the 0.1% difference is disclosed,
not corrected (amendment 2026-10-06).

Labels follow the README: `<family>_<inference|training_full|training_lora|
training_accum[N]>_bs<B>_seq512`, `other_<kernel>`, and
`<family>_adv<strategy>_bs<B>_seq512`. This module imports nothing from the
corpus, so it runs without the SSD.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from scipy.signal import resample_poly

DATASET = "simgar/h200-power-workload-traces-10MHz_5s"
REVISION = "bcd3f8dd7873e481d0467cccfb09fb8e6ce89568"   # task 1.6's pin
SUBSETS = {"genuine": "corpus_wide_10MSs_5s", "adversarial": "adversarial_10MSs_5s"}
FS_IN = 9_765_625.0
FS_OUT = 200_000.0
UP, DOWN = 64, 3125                     # FS_IN * UP / DOWN == FS_OUT exactly
N_IN = 48_828_125
N_OUT = N_IN * UP // DOWN               # 1,000,000
STRATEGIES = ("chunked_opt", "dilute", "lora_dilute", "throttle")

_AI = re.compile(r"^(?P<family>[a-z0-9]+)_(?P<kind>inference|training_full|training_lora|"
                 r"training_accum\d*|adv(?P<strategy>[a-z_]+?))_bs\d+_seq\d+$")


@dataclass(frozen=True)
class Label:
    label: str
    family: str        # model family; the kernel for non-AI
    kind: str          # inference, training_full, training_lora, training_accum, adversarial, other
    strategy: str | None
    cls: str           # training / inference / non-ai (adversarial -> training)

    @property
    def adversarial(self) -> bool:
        return self.kind == "adversarial"


def parse_label(label: str) -> Label:
    if label.startswith("other_"):
        kernel = label[len("other_"):]
        return Label(label, kernel, "other", None, "non-ai")
    m = _AI.match(label)
    if not m:
        raise ValueError(f"unrecognised G&K label: {label}")
    kind, strategy = m["kind"], m["strategy"]
    if strategy is not None:
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown adversarial strategy in {label}")
        return Label(label, m["family"], "adversarial", strategy, "training")
    if kind.startswith("training_accum"):
        kind = "training_accum"
    return Label(label, m["family"], kind, None,
                 "inference" if kind == "inference" else "training")


def decimate(x: np.ndarray) -> np.ndarray:
    """One trace, FS_IN -> FS_OUT, computed in float64, returned float32."""
    return resample_poly(np.asarray(x, dtype=np.float64), UP, DOWN).astype(np.float32)


def capture_key(path: str) -> tuple[str, str, str]:
    """(subset, label, capture stem) of a power-channel path in the dataset tree."""
    p = path.split("/")
    if len(p) != 7 or p[1] != "power" or p[3:5] != ["channels", "power"]:
        raise ValueError(f"not a power-channel path: {path}")
    return p[0], p[2], p[6].removesuffix(".npy")


def sidecar_paths(path: str) -> dict[str, str]:
    """Annotation and record JSON paths that travel with a power-channel file."""
    subset, label, stem = capture_key(path)
    shard = path.split("/")[5]
    base = f"{subset}/power/{label}"
    return {k: f"{base}/{k}/{shard}/{stem}.json" for k in ("annotations", "records")}
