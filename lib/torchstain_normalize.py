"""Macenko stain normalization via the torchstain library (not lib.stain_normalize's
custom implementation).

Required to reproduce data/trident_processed_macenko's uni_v2 corpus: that corpus's
raw WSIs were Macenko-normalized with `00-utils/wsi_preprocess`'s pipeline, which
depends on `torchstain>=1.4.1` (see that project's pyproject.toml). Confirmed by a
self-consistency test — embedding a known v2 corpus patch, cropped fresh from
data/raw_wsi at its stored coordinates, and comparing to its own stored feature
vector: lib.stain_normalize's from-scratch numpy Macenko only reached ~0.5-0.84
cosine similarity (should be ~1.0 for a correctly reproduced pipeline, and IS
~1.0 for the uni_v1 corpus using the same test), while torchstain's Macenko
(same reference patch, data/baseline/63958_x38976_y7616.png) reached 0.96-0.996.
The two implementations differ enough internally (OD conversion, stain-vector
estimation, concentration rescaling) that matching the reference image alone
isn't sufficient — the library itself has to match.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _load_rgb_array(image) -> np.ndarray:
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    return image


def normalize_to_reference(image, reference_image) -> Image.Image:
    """Fit a torchstain NumpyMacenkoNormalizer on reference_image, transform image.

    Args:
        image: Source image to normalize (path/PIL.Image/np.ndarray).
        reference_image: Target stain appearance to normalize toward
            (path/PIL.Image/np.ndarray) — must be the same reference used to
            build whichever corpus you're trying to match embeddings against.

    Returns:
        RGB PIL Image, same size as the input.
    """
    from torchstain.numpy.normalizers import NumpyMacenkoNormalizer

    normalizer = NumpyMacenkoNormalizer()
    normalizer.fit(_load_rgb_array(reference_image))

    arr = _load_rgb_array(image)
    norm_arr, _, _ = normalizer.normalize(I=arr, stains=True)
    norm_arr = np.clip(norm_arr, 0, 255).astype(np.uint8)
    return Image.fromarray(norm_arr)
