"""Macenko H&E stain normalization (numpy/PIL only, no tiatoolbox/staintools dependency).

00-utils/wsi_preprocess/normalize_stains.py already does this via
tiatoolbox.tools.stainnorm.VahadaneNormalizer, but tiatoolbox isn't installed
in this project's .venv and Vahadane's NMF fit occasionally fails to converge
(LinAlgError, handled there with a try/except). Macenko (SVD-based stain
vector estimation from the optical-density plane) is a standard, numerically
simpler alternative that needs only numpy/PIL, both already in .venv.

Reference: Macenko et al., "A method for normalizing histology slides for
quantitative analysis", ISBI 2009.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

ImageLike = "str | Path | Image.Image | np.ndarray"


def _to_array(image) -> np.ndarray:
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    return image


def _rgb_to_od(rgb: np.ndarray) -> np.ndarray:
    """Optical density transform: OD = -log10(I / 255), clipped away from 0."""
    rgb = rgb.astype(np.float64)
    rgb[rgb == 0] = 1.0
    return -np.log10(rgb / 255.0)


def _tissue_mask(rgb: np.ndarray, beta: float, sat_thresh: float = 0.1) -> np.ndarray:
    """Pixels usable for stain-vector estimation: not near-white background,
    and not near-black/grayscale artifacts (scale-bar text, ink, dust, slide
    edges) that real scanned slides often have baked into the image outside
    the tissue region. Both kinds of artifact are near-achromatic (low HSV
    saturation) while true H&E tissue is reliably pink/purple, so a
    saturation floor catches both at once — the OD threshold alone only
    catches the white-background case (near-black pixels have very *high*
    OD, so they pass an `od > beta` filter and would otherwise corrupt the
    stain-vector fit).
    """
    od = _rgb_to_od(rgb).reshape(-1, 3)
    non_background = np.all(od > beta, axis=1)

    hsv = np.array(Image.fromarray(rgb).convert("HSV")).reshape(-1, 3).astype(np.float64)
    saturation = hsv[:, 1] / 255.0
    colorful = saturation > sat_thresh

    return non_background & colorful


class MacenkoNormalizer:
    """fit(target).transform(image) — mirrors tiatoolbox's stainnorm API shape."""

    def __init__(self, alpha: float = 1.0, beta: float = 0.15):
        self.alpha = alpha  # percentile cut (alpha / 100-alpha) for the two stain angles
        self.beta = beta  # OD threshold below which a pixel is treated as background
        self.stain_matrix_target: np.ndarray | None = None
        self.max_conc_target: np.ndarray | None = None

    def _estimate_stain_matrix(self, rgb: np.ndarray) -> np.ndarray:
        od = _rgb_to_od(rgb).reshape(-1, 3)
        mask = _tissue_mask(rgb, self.beta)
        tissue = od[mask]
        if tissue.shape[0] < 100:
            tissue = od[np.all(od > self.beta, axis=1)]  # fall back: drop only the saturation filter
        if tissue.shape[0] < 100:
            tissue = od  # near-blank patch: fall back to all pixels rather than erroring

        # Plane spanned by the two largest-variance OD directions.
        eigvals, eigvecs = np.linalg.eigh(np.cov(tissue.T))
        top2 = eigvecs[:, [-1, -2]]

        proj = tissue @ top2
        angles = np.arctan2(proj[:, 1], proj[:, 0])
        min_angle = np.percentile(angles, self.alpha)
        max_angle = np.percentile(angles, 100 - self.alpha)
        v_min = top2 @ np.array([np.cos(min_angle), np.sin(min_angle)])
        v_max = top2 @ np.array([np.cos(max_angle), np.sin(max_angle)])

        # Convention: hematoxylin (more blue/purple) has a smaller red OD
        # component than eosin (more pink) in most H&E stain-vector fits.
        if v_min[0] > v_max[0]:
            stain_matrix = np.stack([v_min, v_max])
        else:
            stain_matrix = np.stack([v_max, v_min])
        return stain_matrix / np.linalg.norm(stain_matrix, axis=1, keepdims=True)

    @staticmethod
    def _get_concentrations(rgb: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
        od = _rgb_to_od(rgb).reshape(-1, 3)
        conc, *_ = np.linalg.lstsq(stain_matrix.T, od.T, rcond=None)
        conc = conc.T  # (N_pixels, 2): [H concentration, E concentration]
        # Dye concentration can't be negative; lstsq can still produce small
        # negative values for pixels that don't project cleanly onto the
        # fitted H/E axes (artifacts, red blood cells, necrotic debris).
        # Left unclipped these dominate the 99th-percentile scale estimate
        # below and blow up the normalization.
        return np.clip(conc, 0, None)

    def fit(self, target_image) -> "MacenkoNormalizer":
        rgb = _to_array(target_image)
        self.stain_matrix_target = self._estimate_stain_matrix(rgb)
        conc = self._get_concentrations(rgb, self.stain_matrix_target)
        self.max_conc_target = np.percentile(conc, 99, axis=0)
        return self

    def transform(self, image, reestimate_stain_matrix: bool = False) -> Image.Image:
        """Recolor `image` to match the fitted target's staining.

        Args:
            image: Source image to normalize.
            reestimate_stain_matrix: If True, fit a source-specific stain
                matrix (textbook Macenko — corrects for a source that uses
                genuinely different dye hues). Default False: deconvolve the
                source with the *target's* stain matrix instead, only
                rescaling concentration. This is more robust on real query
                crops with necrosis/hemorrhage/artifacts, where re-fitting
                percentile-based angles on the source's own pixels is
                unstable (see scripts/select_average_patch.py candidate #10
                for a case where even the *fit* target image itself had a
                near-degenerate H/E split) — and it matches this dataset's
                actual failure mode, where slide-to-slide differences are
                mostly a staining-intensity/batch effect rather than a
                different pair of dye hues per slide.
        """
        if self.stain_matrix_target is None:
            raise RuntimeError("call fit(target_image) before transform()")
        rgb = _to_array(image)
        h, w = rgb.shape[:2]

        stain_matrix_src = self._estimate_stain_matrix(rgb) if reestimate_stain_matrix else self.stain_matrix_target
        conc_src = self._get_concentrations(rgb, stain_matrix_src)
        max_conc_src = np.percentile(conc_src, 99, axis=0)
        max_conc_src[max_conc_src == 0] = 1e-6

        conc_norm = conc_src * (self.max_conc_target / max_conc_src)
        od_norm = conc_norm @ self.stain_matrix_target
        rgb_norm = 255.0 * np.power(10.0, -od_norm)
        rgb_norm = np.clip(rgb_norm, 0, 255).reshape(h, w, 3).astype(np.uint8)
        return Image.fromarray(rgb_norm)


def normalize_to_reference(image, reference_image) -> Image.Image:
    """One-shot convenience wrapper: fit on reference_image, transform image."""
    return MacenkoNormalizer().fit(reference_image).transform(image)
