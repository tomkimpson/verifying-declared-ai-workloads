"""Channel characterisation: what the acquisition chain measures (spec Phase 3).

The committed findings live in `results/channel/channel-report.md`; this package
is the code that produces them.
"""

from .aliasing import (
    BandPower,
    band_power,
    excess_over_baseline,
    folded_power,
    line_core,
    out_of_band_bands,
    rejection_from_resampling,
    welch_psd,
)
from .envelope import (
    modulation_depth,
    rms_envelope,
    tone_amplitude,
)
from .reduction import (
    PATHS,
    boxcar,
    envelope_fidelity,
    filtered,
    reduce_path,
    subsample,
)

__all__ = [
    "PATHS",
    "BandPower",
    "band_power",
    "boxcar",
    "envelope_fidelity",
    "excess_over_baseline",
    "filtered",
    "folded_power",
    "line_core",
    "modulation_depth",
    "out_of_band_bands",
    "reduce_path",
    "rejection_from_resampling",
    "rms_envelope",
    "subsample",
    "tone_amplitude",
    "welch_psd",
]
