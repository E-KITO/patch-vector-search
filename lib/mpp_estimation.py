"""Estimate an arbitrary query image's apparent magnification relative to
the indexed corpus, and rescale it to match, before tiling/embedding it.

The corpus was extracted at a fixed physical scale (target_magnification=20,
patch_size_level0=224 — see data/trident_processed/summary.md). An external
reference image (e.g. an NTP atlas figure) carries no MPP metadata and can be
photographed/scanned at any magnification, so a naive crop can show tissue at
a completely different physical scale than the corpus (measured: the necrosis
atlas set mixed ~4-10x whole-lobule overview shots with ~20x close-ups).

**Current recommendation: don't use this module by default.** The only
working method here — nearest-centroid matching in UNI embedding space
(`build_scale_reference_centroids` + `estimate_relative_scale_fm_tiled` /
`auto_scale_image_fm_tiled`, used by
`lib.query_embedding.embed_image_tiles_auto_scale`) — was validated visually
(it stops the tile-blurring bugs described below) but then measured against
ground truth (7 NTP-atlas categories with confirmed corpus labels, see
`scripts/validate_against_ground_truth.py`) to make search recall *worse*
than no correction in most categories. Looking cleaner is not the same as
retrieving better. `experiments/0003_20260808_query_demo` defaults
`--auto_scale` to off for this reason. Use this module only if you've
re-validated it helps your specific case.

Two approaches were tried; both are documented here so they aren't
re-attempted blindly:

1. **Red blood cells as a "histologic ruler" — abandoned, not implemented
   here anymore.** RBCs are a near-constant ~7.5um biconcave disk regardless
   of surrounding pathology, unlike hepatocyte nuclei (whose apparent size is
   confounded by the very pathologies this project queries for — hypertrophy,
   karyomegaly, mitosis all change nuclear size). Three absolute/relative
   color-threshold schemes were tried (percentile eosin+H:E ratio, fixed HSV
   hue/saturation calibrated from one real vessel sample, the same plus a
   brightness floor after Macenko-normalizing) and each failed differently on
   different atlas images — one flagged ~20% of any image regardless of RBC
   content, another matched hepatocyte nuclei instead of RBCs, the third
   matched cell-membrane highlight edges. The NTP atlas JPEGs' color grading
   is inconsistent enough (mixed scan/photo/print pipelines, no shared
   calibration) that simple global color rules don't transfer between images.
   Don't re-attempt this without a fundamentally different (non-global-color)
   approach.

2. **Nearest-centroid matching in UNI embedding space — implemented below,
   works but doesn't help retrieval.** Synthesize training examples of known
   relative scale directly from the corpus (crop a `224*k` px window from a
   real slide at native resolution and resize to 224x224 — this is exactly
   what capturing that same tissue at 1/k the magnification would look like),
   embed them, and average into one centroid per scale k
   (`build_scale_reference_centroids`). An unknown image's scale is the k
   whose centroid its own embedding is closest to.

   A first version (`estimate_relative_scale_fm`, since removed) embedded the
   *whole* query image resized to 224x224 to make this decision — for any
   source much larger than 224px (every NTP atlas figure is 1000-1800px+),
   that whole-image downsize throws away most of the image's actual
   resolution, so the resulting thumbnail looks "coarse" for the same reason
   any large photo looks coarse when shrunk to a postage stamp — not because
   the tissue was captured at low magnification. Measured on a
   25-category/94-image batch: every category's estimated scale landed in
   the 2.5x-8.0x range (median ~5x, no interior similarity peak — a symptom
   of measuring "detail lost by the downsize", not true magnification), and
   applying that correction blurred every query tile into indistinct color
   blobs versus clearly resolved nuclei/RBCs at native scale.

   The fix, kept below as `estimate_relative_scale_fm_tiled` /
   `auto_scale_image_fm_tiled`: vote across every native-resolution tile
   (via `lib.query_embedding.embed_image_tiles`) instead of one whole-image
   downsize, then take the median. Magnification is a property of the whole
   photographed image, not of any one diagnostic region — unlike
   embed_image_tiles's *search* step (which needs whichever tile shows the
   lesion), the scale vote doesn't need a *representative* tile, just tiles
   with real tissue in them; the median is robust to a minority of
   background/label tiles voting badly. Re-run on the same batch, this
   produced a plausible, spread-out distribution (58/94 images voted 0.5x,
   19 voted 4.0x, 14 voted 1.0x, only 3 voted 8.0x) and every
   visually-inspected category showed clearly resolved nuclei/cell
   boundaries after correction. `max_scale` (default 2.0) additionally caps
   the result: every measured over-estimate failure — including one where
   the tiled vote itself was fooled by a fatty-change image's large uniform
   vacuoles reading as "low magnification" the same way a whole-image
   downsize did — has landed at 4x/8x, none at or below 2x.

   A second, independent bug was found and fixed in the same pass:
   `embed_image_tiles_auto_scale` was running the scale vote on
   *stain-normalized* tiles when a `stain_reference` was also passed, but the
   scale centroids were built from *unnormalized* real corpus crops —
   comparing normalized tiles against unnormalized centroids is a mismatched
   reference and was measured to flip a vote's median from 0.5 to 8.0 on the
   same image. Fixed by voting on unnormalized tiles first, then applying
   `stain_reference` only to the final tiles that get embedded for search.

   None of this changed the ground-truth conclusion above: even bug-free,
   this correction was not the single best option in any of 7 tested
   categories (see `lib.query_embedding.embed_image_tiles_auto_scale`'s
   docstring for the full comparison table).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def _load_rgb(image) -> Image.Image:
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    return image.convert("RGB")


# ============================================================
# Nearest-centroid matching in UNI embedding space
# ============================================================

DEFAULT_SCALES = (0.5, 1.0, 2.0, 4.0, 8.0)


def build_scale_reference_centroids(
    raw_wsi_dir: str | Path,
    slide_meta_path: str | Path,
    scales: tuple[float, ...] = DEFAULT_SCALES,
    n_slides: int = 15,
    samples_per_slide: int = 2,
    seed: int = 42,
    device: str | None = None,
) -> dict[float, np.ndarray]:
    """Build one UNI-embedding centroid per relative scale, from synthetic
    examples generated directly from the corpus's own real slides.

    For relative scale k, a `224*k` px window (level-0/native resolution) is
    cropped from a real slide and resized down/up to 224x224 — exactly what
    that same tissue would look like if captured at 1/k the magnification
    (k=2 => captured at half the corpus's magnification, i.e. 2x the corpus's
    mpp; k=0.5 => captured at 2x the corpus's magnification). Embedding these
    with the same UNI encoder used everywhere else in this project turns
    "what apparent scale is this image at" into a nearest-neighbor lookup in
    a space we already trust, instead of a bespoke color/shape heuristic.

    Args:
        raw_wsi_dir: Directory containing {slide_id}.svs (data/raw_wsi).
        slide_meta_path: slide_meta.parquet (has level0_width/height per slide).
        scales: Relative scales to build centroids for. 1.0 = corpus's own
            native scale.
        n_slides: Number of slides to sample.
        samples_per_slide: Crops per slide per scale (total examples per
            scale = n_slides * samples_per_slide).
        seed: RNG seed for slide/crop selection.
        device: "cuda" or "cpu" for embedding (see lib.query_embedding.embed_image).

    Returns:
        {scale: (1024,) L2-normalized centroid embedding}.
    """
    import openslide

    from lib.query_embedding import embed_image

    slide_meta = pd.read_parquet(slide_meta_path)
    rng = np.random.default_rng(seed)
    sample = slide_meta.sample(n=min(n_slides, len(slide_meta)), random_state=seed)

    centroids = {}
    for scale in scales:
        crop_size = max(1, round(224 * scale))
        embeddings = []
        for row in sample.itertuples():
            slide_path = Path(raw_wsi_dir) / f"{row.slide_id}.svs"
            try:
                with openslide.OpenSlide(str(slide_path)) as slide:
                    for _ in range(samples_per_slide):
                        max_x = max(1, row.level0_width - crop_size)
                        max_y = max(1, row.level0_height - crop_size)
                        cx = int(rng.integers(0, max_x))
                        cy = int(rng.integers(0, max_y))
                        region = slide.read_region((cx, cy), 0, (crop_size, crop_size)).convert("RGB")
                        region = region.resize((224, 224), Image.LANCZOS)
                        embeddings.append(embed_image(region, device=device))
            except Exception:
                continue
        centroid = np.mean(embeddings, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        centroids[scale] = centroid.astype(np.float32)

    return centroids


def save_scale_centroids(centroids: dict[float, np.ndarray], path: str | Path) -> None:
    """Persist build_scale_reference_centroids's output so callers don't have
    to rebuild it (openslide reads + ~150 encoder forward passes) on every
    process that wants to auto-scale a query image."""
    scales = np.array(sorted(centroids.keys()), dtype=np.float64)
    vectors = np.stack([centroids[s] for s in scales]).astype(np.float32)
    np.savez(path, scales=scales, vectors=vectors)


def load_scale_centroids(path: str | Path) -> dict[float, np.ndarray]:
    data = np.load(path)
    return {float(s): v for s, v in zip(data["scales"], data["vectors"])}


def estimate_relative_scale_fm_tiled(
    image,
    centroids: dict[float, np.ndarray],
    device: str | None = None,
    tile_size: int = 224,
    min_margin: float = 0.0,
    max_scale: float | None = 2.0,
) -> tuple[float, dict]:
    """Vote across every native-resolution tile of `image` to estimate its
    scale relative to the corpus (see the module docstring for why this
    replaced an earlier whole-image-downsize version, and why the whole
    feature is off by default despite working as designed).

    Magnification is a property of the whole photographed/scanned image, not
    of any one diagnostic region — unlike embed_image_tiles's search step
    (which needs whichever tile actually shows the lesion), this doesn't
    need a *representative* tile. Any tile with real tissue in it gives a
    valid vote, and taking the median across all of them is robust to a
    minority of background/label/artifact tiles voting badly — no need to
    identify which tiles those are in advance.

    Args:
        image: Path/Image, or an already-embedded (n_tiles, 1024) array
            (pass lib.query_embedding.embed_image_tiles's own output
            directly to avoid embedding the image twice).
        min_margin: Drop tiles whose best-vs-second-best centroid similarity
            margin is below this before taking the median (filters
            near-blank/ambiguous tiles that don't clearly favor any scale).
            0.0 keeps every tile.
        max_scale: Clip the final median vote to this ceiling. Every
            measured failure of this estimator (whole-image-downsize bias,
            and — even with that fixed and voting per-tile — large uniform
            pathological structures like fatty-change vacuoles reading as
            "low magnification" the same way a real downsize does) has been
            an *over*-estimate; none has been an under-estimate. Capping the
            upper end is a blunt but evidence-backed safety margin: on the
            25-category NTP atlas batch, 0.5x-2.0x tiles were visually clean
            in every spot check, while 4x/8x tiles included the fatty-change
            failure (native tiles already showed well-resolved, properly
            dense steatosis; 4x correction left 1-2 giant vacuoles filling
            the frame). Pass None to disable clipping.

    Returns:
        (median_scale, debug_info) — debug_info["votes"]/["margins"] are
        per-tile arrays, debug_info["n_kept"] is how many survived min_margin
        filtering, debug_info["median_scale_unclipped"] is the value before
        max_scale was applied, for sanity-checking a noisy or bimodal vote.
    """
    if isinstance(image, np.ndarray):
        tile_vecs = image
    else:
        from lib.query_embedding import embed_image_tiles

        tile_vecs = embed_image_tiles(image, device=device, tile_size=tile_size)

    scales = np.array(sorted(centroids.keys()))
    centroid_matrix = np.stack([centroids[s] for s in scales])  # (n_scales, 1024)
    sims = tile_vecs @ centroid_matrix.T  # (n_tiles, n_scales)

    order = np.argsort(sims, axis=1)
    best_idx, second_idx = order[:, -1], order[:, -2]
    rows = np.arange(len(sims))
    margins = sims[rows, best_idx] - sims[rows, second_idx]
    votes = scales[best_idx]

    keep = margins >= min_margin
    if not keep.any():
        keep = np.ones_like(keep)  # every tile ambiguous: fall back to using them all rather than erroring

    median_scale = float(np.median(votes[keep]))
    clipped_scale = median_scale if max_scale is None else min(median_scale, max_scale)
    debug = {
        "votes": votes.tolist(),
        "margins": margins.tolist(),
        "n_tiles": int(len(tile_vecs)),
        "n_kept": int(keep.sum()),
        "median_scale_unclipped": median_scale,
    }
    return clipped_scale, debug


def auto_scale_image_fm_tiled(
    image,
    centroids: dict[float, np.ndarray],
    device: str | None = None,
    tile_size: int = 224,
    min_margin: float = 0.0,
    max_scale: float | None = 2.0,
) -> tuple[Image.Image, float, dict]:
    """Rescale `image` to the corpus's apparent scale, using
    estimate_relative_scale_fm_tiled's per-tile vote.

    max_scale (default 2.0) caps the applied correction — see
    estimate_relative_scale_fm_tiled's docstring for why: every measured
    over-estimate failure has landed at 4x/8x, none at or below 2x.

    Returns (rescaled_image, scale_factor, debug_info). Pass the rescaled
    image into lib.query_embedding.embed_image / embed_image_tiles as usual.
    """
    image = _load_rgb(image)
    scale, debug = estimate_relative_scale_fm_tiled(
        image, centroids, device=device, tile_size=tile_size, min_margin=min_margin, max_scale=max_scale
    )
    w, h = image.size
    resized = image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    return resized, scale, debug
