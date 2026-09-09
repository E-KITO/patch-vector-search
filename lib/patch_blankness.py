"""Per-patch blankness measurement and the corpus background-exclusion criterion.

`blankness_metrics` is the raw measurement (mean intensity, std, saturated-pixel
fraction). It was first written inline in
scripts/blank_patch_similarity_diagnostic.py (2026-09-05) to give that
diagnostic a background/tissue label, after finding that
lib.query_embedding._is_blank_tile (bright + low pixel variance only) misses
background that is bright but simply unstained.

`is_background` turns those measurements into the verdict used to drop rows
from the corpus manifest (scripts/measure_corpus_blankness.py -> a filtered
build_patch_manifest). The thresholds are provisional: measure_corpus_blankness
writes the raw fractions for every patch so the cut can be retuned against the
full-corpus distribution without re-cropping.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

# A pixel counts as stained tissue if its HSV saturation exceeds this fraction.
SAT_THRESHOLD = 0.10


def blankness_metrics(img: Image.Image) -> dict[str, float]:
    """Raw measurements, no verdict.

    sat_frac is the fraction of pixels with any real colour: slide background
    is bright *and* unsaturated, whereas pale tissue is bright but still
    stained -- which mean/std alone (lib.query_embedding._is_blank_tile) misses.
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
    return {
        "mean_intensity": float(rgb.mean()),
        "std_intensity": float(rgb.std()),
        "sat_frac": float((hsv[..., 1] > SAT_THRESHOLD).mean()),
    }


def is_background(
    mean_intensity,
    sat_frac,
    *,
    sat_frac_max: float = 0.10,
    mean_min: float = 215.0,
):
    """Provisional corpus background criterion: bright and essentially unstained.

    Matches scripts/blank_patch_similarity_diagnostic.py's is_background column
    (sat_frac < 0.10 & mean_intensity > 215), calibrated by eye on ~2000
    sampled patches. Parameterised so the threshold pass before the filtered
    manifest build can sweep it against the full-corpus distribution. Bias the
    final cut toward false negatives -- leaving a little background in is safer
    than dropping genuine sparse tissue (sinusoidal dilation, oedema, early
    necrosis with cell dropout).

    Accepts scalars or numpy arrays; returns the same shape.
    """
    return (np.asarray(sat_frac) < sat_frac_max) & (np.asarray(mean_intensity) > mean_min)
