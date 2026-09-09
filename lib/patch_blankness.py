"""Per-patch blankness measurement and the corpus background-exclusion criterion.

`blankness_metrics` is the raw measurement (mean intensity, std, saturated-pixel
fraction). It was first written inline in
scripts/blank_patch_similarity_diagnostic.py (2026-09-05) to give that
diagnostic a background/tissue label, after finding that
lib.query_embedding._is_blank_tile (bright + low pixel variance only) misses
background that is bright but simply unstained.

`is_background` turns those measurements into the verdict used to drop rows
from the corpus manifest (scripts/measure_corpus_blankness.py ->
experiments/0017). The cut is `sat_frac < 0.10`, fixed after the job 10491
audit (outputs/measure_corpus_blankness/audit/): every sat_frac bin below 0.10
is slide background / section-edge slivers / RBC-in-empty-field / coverslip
artifact with no diagnostic tissue, and real tissue content only appears from
~0.15. An earlier `& mean_intensity > 215` guard was dropped -- it protected
no dark tissue (there is none below sat_frac 0.10) and only kept ~1,500
obvious-garbage patches (out-of-focus grey, coverslip cracks, half-black
edge-of-scan).
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


def is_background(sat_frac, *, sat_frac_max: float = 0.10):
    """Corpus background criterion: essentially no stained pixels.

    A patch is background if fewer than `sat_frac_max` of its pixels carry any
    real stain colour. Fixed at 0.10 by the job 10491 audit (see the module
    docstring). Accepts a scalar or a numpy array; returns the same shape.
    """
    return np.asarray(sat_frac) < sat_frac_max
