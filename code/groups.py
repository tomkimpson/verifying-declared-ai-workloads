"""Strategy groups of spec D2, from the manifest `condition`.

`DECLARED` is R1's declared workload (spec D3). `HIDDEN_GROUPS` are the
evasion strategies: the hidden-training configurations that Phase 2a must
never score (spec task 2.4) and that 2.5's kill check scores once.
"""

from __future__ import annotations

#: R1's declared workload: inference, including idle (spec D3).
DECLARED = ("serving-decode", "serving-prefill", "idle")

#: Honest training (spec D2). Not hidden: Phase 2a's positive control (§3
#: check 6) draws on `training` from it.
HONEST = ("training", "training-checkpoint-eval", "training-copying",
          "training-discarded", "lora")

NEGATIVES = ("idle", "serving-decode", "serving-prefill", "steady-gemm",
             "memory-bound")
NEGATIVE_PREFIXES = ("periodic-gemm-",)

#: Evasion groups, matched by condition prefix (spec D2).
HIDDEN_GROUPS = {
    "time-sharing": ("mixture-timeshared-",),
    "co-location": ("mixture-concurrent-",),
    "jitter": ("cadence-jitter-",),
    "padding": ("cadence-pad-",),
    "power cap": ("powercap-",),
}


def group(condition: str) -> str:
    """The D2 group of a manifest condition. Raises on an unknown one."""
    if condition in HONEST:
        return "honest"
    if condition in NEGATIVES or condition.startswith(NEGATIVE_PREFIXES):
        return "negatives"
    for g, prefixes in HIDDEN_GROUPS.items():
        if condition.startswith(prefixes):
            return g
    raise ValueError(f"condition {condition!r} is in no D2 group")


def is_hidden(condition: str) -> bool:
    return group(condition) in HIDDEN_GROUPS
