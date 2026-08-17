"""Embed an arbitrary image into the same UNI embedding space used to build a patch index.

Uses TRIDENT's encoder_factory, pinned in pyproject.toml to the exact commit
used to generate data/trident_processed/.../features_uni_v1/*.h5
(00-utils/wsi_preprocess uses the same pin) — a different TRIDENT version
could preprocess images differently and desync the embedding space from the
already-indexed corpus.

Supports two corpora/encoders — pass the matching `encoder_name` for whichever
index you're querying, they are NOT interchangeable (different embedding
spaces, different dimensionality):
- "uni_v1" (default): data/trident_processed/20x_224px_0px_overlap/features_uni_v1,
  1024-dim, corpus patches extracted natively at 224px.
- "uni_v2": data/trident_processed_macenko/.../20x_256px_0px_overlap/features_uni_v2,
  1536-dim (ViT-giant vs v1's ViT-large), corpus patches extracted natively at
  256px then resized to 224 for the encoder. This corpus was also produced
  from a different (Macenko-normalized, exact method not recorded in TRIDENT's
  own config files) preprocessing of the raw WSIs, so any v1-vs-v2 comparison
  conflates encoder + patch-size + normalization differences, not normalization
  alone — see patch-vector-search-project memory for the ground-truth comparison
  this was run through.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


_ENCODER_CACHE: dict[tuple[str, str], tuple] = {}


def _load_encoder(device: str | None, encoder_name: str = "uni_v1"):
    import torch
    from trident.patch_encoder_models import encoder_factory

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Loading UNI's weights from disk is expensive (seconds) — callers that
    # embed many images in a loop (e.g. lib.mpp_estimation's centroid
    # builder, which embeds hundreds of synthetic crops) would otherwise pay
    # that cost on every single call. Safe to cache: the encoder is a frozen,
    # stateless eval-mode module for a given (device, encoder_name).
    cache_key = (device, encoder_name)
    if cache_key not in _ENCODER_CACHE:
        encoder = encoder_factory(encoder_name).to(device).eval()
        # encoder.precision is float16/bfloat16, tuned for GPU inference; these
        # have patchy CPU kernel support in PyTorch, so force float32 on CPU instead.
        dtype = encoder.precision if device.startswith("cuda") else torch.float32
        encoder = encoder.to(dtype)
        _ENCODER_CACHE[cache_key] = (encoder, dtype)

    encoder, dtype = _ENCODER_CACHE[cache_key]
    return encoder, device, dtype


def _load_rgb(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    # Always normalize to RGB, even for an already-loaded Image: eval_transforms
    # ends in Normalize(mean=(3,), std=(3,)), which errors on RGBA/grayscale/palette
    # input (e.g. openslide.read_region() returns RGBA by default).
    return image.convert("RGB")


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def _is_blank_tile(crop: Image.Image, mean_threshold: float = 240.0, std_threshold: float = 8.0) -> bool:
    """A tile is near-uniform white slide background (not tissue) if it's
    both bright and has almost no pixel variation. Measured necessary: a
    grid tiling of a large image inevitably includes margin/background
    tiles (e.g. the corners of an NTP atlas figure with white padding), and
    a blank query tile can score deceptively *high* approximate similarity
    against blank background regions in the corpus (real slide background
    is a small, tight, easy-to-quantize region of UNI embedding space) —
    on one query this let 3/54 blank tiles dominate the top-4 reranked
    candidates and produced an all-background 'match' result with
    similarity ~0.8, indistinguishable from a real hit without visual
    inspection."""
    arr = np.asarray(crop, dtype=np.float32)
    return bool(arr.mean() > mean_threshold and arr.std() < std_threshold)


def embed_image(
    image: str | Path | Image.Image,
    device: str | None = None,
    encoder_name: str = "uni_v1",
    stain_reference: str | Path | Image.Image | None = None,
    resize_mode: str = "centercrop",
) -> np.ndarray:
    """Embed a single image with TRIDENT's patch encoder.

    Args:
        image: Path to an image file, or an already-loaded PIL Image.
        device: "cuda" or "cpu". Defaults to cuda if available.
        encoder_name: "uni_v1" (1024-dim, matches the original corpus) or
            "uni_v2" (1536-dim, matches data/trident_processed_macenko's
            corpus) — must match whichever PatchIndex you're querying, the
            two are different embedding spaces. See this module's docstring.
        stain_reference: Optional path/Image to Macenko-normalize `image`
            against before embedding (see lib.stain_normalize). Use this to
            recolor a query crop from a different scanner/lab toward this
            corpus's typical staining before it's embedded and searched —
            corpus staining/batch differences are otherwise a major axis of
            variation in this UNI feature space (see the wsi-ad batch-effect
            finding), so an unnormalized off-distribution query can search
            more on "which lab scanned this" than tissue content. **Caveat**
            (measured in scripts/compare_query_normalization.py): this helps
            queries with a real domain gap (textbook/atlas scans) but can
            badly hurt an already-in-distribution query (a real same-corpus-
            style patch) — e.g. a cellular-infiltration query that ranked
            all 7 ground-truth slides in the top 66/998 raw dropped to 0/7
            found at all once stain-normalized. Not a safe default; compare
            raw vs. normalized per query rather than always applying it.
        resize_mode: How to fit `image` into the encoder's native 224x224
            input.
            - "centercrop" (default): TRIDENT's own `eval_transforms`
              (`Resize(shorter=224)` + `CenterCrop(224,224)`) — preserves
              aspect ratio but discards content on the long axis outside the
              center square. Matches how the indexed corpus itself was
              produced.
            - "stretch": resize straight to (224,224), distorting aspect
              ratio but keeping every source pixel. Measured to shift the
              embedding only modestly (cosine ~0.95-0.96 vs. centercrop on
              a near-square crop) but can meaningfully change which slides
              rank highest for a WSI reverse-lookup — see the necrosis_cut
              comparison in this project's memory/plan notes.

    Returns:
        L2-normalized float32 embedding of shape (dim,), where dim depends on
        encoder_name (1024 for uni_v1, 1536 for uni_v2).
    """
    import torch

    encoder, device, dtype = _load_encoder(device, encoder_name)
    image = _load_rgb(image)

    if stain_reference is not None:
        from lib.stain_normalize import MacenkoNormalizer

        image = MacenkoNormalizer().fit(stain_reference).transform(image)

    if resize_mode == "centercrop":
        tensor = encoder.eval_transforms(image)
    elif resize_mode == "stretch":
        tensor = _stretch_transform(image)
    else:
        raise ValueError(f"unknown resize_mode: {resize_mode!r} (expected 'centercrop' or 'stretch')")

    tensor = tensor.unsqueeze(0).to(device=device, dtype=dtype)

    with torch.no_grad():
        embedding = encoder(tensor)

    embedding = embedding.float().cpu().numpy()[0].astype(np.float32)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


def _stretch_transform(image: Image.Image, tile_size: int = 224):
    from torchvision import transforms

    resize_flat = transforms.Compose([
        transforms.Resize((tile_size, tile_size), interpolation=Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return resize_flat(image)


def embed_image_tiles(
    image: str | Path | Image.Image,
    device: str | None = None,
    encoder_name: str = "uni_v1",
    tile_size: int = 224,
    native_tile_size: int | None = None,
    stride: int | None = None,
    stain_reference: str | Path | Image.Image | None = None,
) -> np.ndarray:
    """Split a large reference image into a grid of crops and embed each one,
    instead of committing to a single crop.

    Useful when the diagnostic feature in a large image (e.g. a low-detail
    NTP atlas figure) isn't centered or its extent isn't known in advance —
    rather than guessing one crop and risking cropping the feature out
    (measured to happen with both centercrop and naive "avoid dark
    annotation ink" heuristics on this project's atlas images), embed every
    tile and let the caller search+aggregate across all of them via
    lib.search.PatchIndex.search_top_slides_multi.

    Args:
        image: Path to an image file, or an already-loaded PIL Image.
        device: "cuda" or "cpu". Defaults to cuda if available.
        encoder_name: "uni_v1" (1024-dim) or "uni_v2" (1536-dim) — must match
            whichever PatchIndex you're querying. See this module's docstring.
        tile_size: Side length the encoder actually receives (224 = its
            native input size).
        native_tile_size: Side length to crop the grid at *before* resizing
            down/up to tile_size — set this when the corpus you're matching
            extracted patches at a native size other than its encoder's input
            size (e.g. data/trident_processed_macenko's uni_v2 corpus crops
            256px native patches, then resizes to 224 for the encoder — see
            lib.raw_patch.crop_patch, which does the same for real corpus
            patches). Defaults to tile_size (crop and encoder input are the
            same, no resize) when None. Leaving this at the encoder's input
            size when the true corpus field-of-view is larger is the same
            kind of magnification mismatch this project spent a lot of effort
            fixing for the auto-scale feature (lib.mpp_estimation) — except
            here the fix is just "crop at the right native size", since the
            corpus's native size is already known instead of needing to be
            estimated.
        stride: Step between tile origins, in native_tile_size units.
            Defaults to native_tile_size (non-overlapping grid); pass e.g.
            native_tile_size // 2 for 50% overlap.
        stain_reference: Optional path/Image to Macenko-normalize `image`
            against before tiling (see embed_image's docstring for the same
            caveat: helps domain-shifted queries but can hurt already
            in-distribution ones — not a safe default).

    Returns:
        (n_tiles, 1024) float32 array, each row L2-normalized. Near-blank
        background tiles (see _is_blank_tile) are dropped before embedding —
        n_tiles can be smaller than the full grid size, and is never zero
        (falls back to the full grid if every tile is blank).
    """
    import torch

    encoder, device, dtype = _load_encoder(device, encoder_name)
    image = _load_rgb(image)

    if stain_reference is not None:
        from lib.stain_normalize import MacenkoNormalizer

        image = MacenkoNormalizer().fit(stain_reference).transform(image)

    crop_size = native_tile_size or tile_size

    # Upscale first if the image is smaller than one tile on either side, so
    # there's always at least a 1x1 grid.
    w, h = image.size
    scale = max(crop_size / w, crop_size / h, 1.0)
    if scale > 1.0:
        image = image.resize((max(crop_size, round(w * scale)), max(crop_size, round(h * scale))), Image.LANCZOS)
        w, h = image.size

    stride = stride or crop_size
    xs = sorted(set(list(range(0, w - crop_size + 1, stride)) + [w - crop_size]))
    ys = sorted(set(list(range(0, h - crop_size + 1, stride)) + [h - crop_size]))

    from torchvision import transforms

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    crops = [image.crop((x, y, x + crop_size, y + crop_size)) for y in ys for x in xs]
    non_blank = [c for c in crops if not _is_blank_tile(c)]
    if non_blank:  # only filter if it wouldn't empty the tile set entirely
        crops = non_blank
    if crop_size != tile_size:
        crops = [c.resize((tile_size, tile_size), Image.LANCZOS) for c in crops]

    tiles = [to_tensor(c) for c in crops]
    batch = torch.stack(tiles).to(device=device, dtype=dtype)

    with torch.no_grad():
        embeddings = encoder(batch)

    embeddings = embeddings.float().cpu().numpy().astype(np.float32)
    return _normalize_rows(embeddings)


def embed_image_tiles_auto_scale(
    image: str | Path | Image.Image,
    centroids: dict[float, np.ndarray],
    device: str | None = None,
    tile_size: int = 224,
    stride: int | None = None,
    stain_reference: str | Path | Image.Image | None = None,
    max_scale: float | None = 2.0,
) -> tuple[np.ndarray, float, dict]:
    """embed_image_tiles, but first rescale `image` to the corpus's apparent
    magnification, estimated by voting across `image`'s own native-resolution
    tiles against lib.mpp_estimation's FM scale centroids (see
    lib.mpp_estimation.estimate_relative_scale_fm_tiled).

    Query images (e.g. NTP atlas figures) are captured at inconsistent,
    unlabeled magnifications — mixing ~4-10x whole-lobule overviews with
    ~20x close-ups even within the same finding category — so tiling at the
    image's native pixel scale can search at the wrong physical scale
    relative to the corpus's fixed 20x/224px patches. This rescales first so
    each tile covers roughly the same tissue extent as an indexed patch.

    An earlier version estimated scale from a single whole-image downsize to
    224x224 instead of a tile vote — measured to systematically
    over-magnify (median ~5x on a 25-category/94-image batch, similarities
    monotonically increasing with scale with no interior peak) and visibly
    blur every query tile into indistinct color blobs, because a whole-image
    downsize looks "coarse" for any source much larger than 224px regardless
    of the tissue's true magnification. The per-tile vote fixes this: unlike
    embed_image_tiles's search step (which needs the *diagnostic* tile),
    magnification is a property of the whole image, so any tile with real
    tissue content gives a valid vote and the median is robust to a handful
    of background/label tiles. Re-run on the same batch, every
    visually-inspected category showed clearly resolved nuclei/cell
    boundaries after correction. See lib.mpp_estimation's module docstring
    for the full history, including a second stain-normalization-ordering
    bug found and fixed in the same pass.

    A later fatty-change case showed the per-tile vote still over-estimates
    when the image's real content (large uniform vacuoles) looks texturally
    "coarse" the same way a genuinely low-magnification crop does — native
    tiles were already well-resolved 20x-equivalent steatosis, but voted
    4.0x anyway. `max_scale` (default 2.0, see
    lib.mpp_estimation.estimate_relative_scale_fm_tiled's docstring) caps
    this: every measured over-estimate failure has landed at 4x/8x, none at
    or below 2x.

    **Validated against ground truth after all image-quality bugs above were
    fixed — and found to HURT search recall.** Comparing this function
    (+ stain_reference) against plain embed_image_tiles across 7 categories
    with confirmed corpus ground truth: this combination was the single best
    option in 0 of 7 categories. Used alone (no stain_reference) it won 2/7;
    plain embed_image_tiles (no correction at all) won 3/7; stain_reference
    alone (no scale correction) won 2/7. The two corrections interact
    negatively when combined — e.g. one category's best_rank went from 16
    (no correction) to 2 (stain_reference alone) to 9 (both together), and
    another's best_rank went from 14 (no correction) to 45 (both together,
    the worst of all four options tested). Looking visually cleaner does not
    imply better retrieval. Do not enable `--auto_scale` or
    `--stain_reference` by default in `experiments/0003_20260808_query_demo`
    or elsewhere based on this function alone — if a specific query seems to
    need one, A/B it against no correction on that query first.

    Args:
        centroids: {scale: (1024,) centroid} from
            lib.mpp_estimation.build_scale_reference_centroids /
            load_scale_centroids (see data/scale_centroids.npz).
        max_scale: Upper bound applied to the scale vote before rescaling.
            Pass None to disable.
        (other args: see embed_image_tiles)

    Returns:
        (tiles, scale_factor, debug_info) — tiles is (n_tiles, 1024) float32,
        same as embed_image_tiles; scale_factor and debug_info come from
        lib.mpp_estimation.estimate_relative_scale_fm_tiled (debug_info["votes"]/
        ["margins"] are per-tile, for sanity-checking a noisy vote).
    """
    from lib.mpp_estimation import estimate_relative_scale_fm_tiled

    image = _load_rgb(image)
    # The scale vote must run on unnormalized tiles: build_scale_reference_centroids
    # built its centroids from raw corpus crops (no stain_reference applied), so
    # voting with normalized tiles compares against a mismatched reference and
    # was measured to flip the outcome (e.g. median 0.5 -> 8.0 for one image) —
    # stain_reference is applied only to the final tiles returned below.
    vote_tiles = embed_image_tiles(image, device=device, tile_size=tile_size, stride=stride)
    scale, debug = estimate_relative_scale_fm_tiled(
        vote_tiles, centroids, device=device, tile_size=tile_size, max_scale=max_scale
    )

    w, h = image.size
    scaled_image = image if scale == 1.0 else image.resize(
        (max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS
    )
    tiles = embed_image_tiles(
        scaled_image, device=device, tile_size=tile_size, stride=stride, stain_reference=stain_reference
    )
    return tiles, scale, debug


def remove_annotation_arrows(
    image: str | Path | Image.Image,
    dark_threshold: int = 60,
    min_line_length: int = 40,
    max_line_gap: int = 4,
    hough_threshold: int = 40,
    stroke_width: int = 10,
) -> Image.Image:
    """Detect and inpaint solid-black arrow/line annotations common in NTP
    atlas / textbook histology figures, without erasing genuinely dark
    tissue features (mitotic figures, hematopoietic cell clusters,
    inflammatory infiltrates).

    Two approaches were tried; this is the one that actually works on this
    dataset's arrows. A first attempt classified whole connected dark
    components by bounding-box elongation ("thin & long => arrow"), but the
    arrowhead in these figures is drawn touching its target (a mitotic
    figure, a cell cluster), so the shaft+head+target merge into one
    connected component whose axis-aligned bbox is closer to square than a
    line's — bbox elongation systematically failed to flag them (measured:
    the 4 arrows in the increased-mitosis atlas image scored bbox
    elongation 1.0-2.6, all below a min_elongation=2.5 cutoff that should
    have caught them). Rotation-invariant PCA elongation of the component's
    pixel coordinates *did* separate them cleanly (~3.8 for the 4 arrows vs.
    <2.9 for other dark specks) but still couldn't locate *where within* the
    merged blob the shaft was, so masking the whole component would also
    erase the touching target.

    `cv2.HoughLinesP` sidesteps both problems: it finds straight-line
    segments directly (arrows are drawn as straight/angled lines; organic
    chromatin clumps and cell clusters are not), regardless of what they're
    touching or connected to. Only the shaft is masked (thickened by
    `stroke_width` to cover the stroke's actual width and anti-aliasing),
    not the whole merged component — the touching target is left intact.
    The arrowhead itself (a small filled triangle, not a line) is mostly
    left alone; in practice it's a small fraction of the ink and its
    residual presence doesn't materially affect embedding search results
    (see scripts/compare_query_normalization.py-style validation).

    Args:
        image: Path to an image file, or an already-loaded PIL Image.
        dark_threshold: Mean-RGB brightness below which a pixel is candidate ink.
        min_line_length: Minimum straight segment length (px) for cv2.HoughLinesP.
        max_line_gap: Max gap (px) to bridge between collinear segments.
        hough_threshold: Hough accumulator vote threshold (higher = fewer, more confident lines).
        stroke_width: Thickness (px) to draw each detected line at when building the inpaint mask.

    Returns:
        A copy of `image` (RGB) with detected line strokes inpainted from surrounding pixels.
    """
    import cv2

    image = _load_rgb(image)
    arr = np.array(image)
    gray = arr.mean(axis=2)
    dark_mask = (gray < dark_threshold).astype(np.uint8) * 255

    lines = cv2.HoughLinesP(
        dark_mask, 1, np.pi / 180, threshold=hough_threshold,
        minLineLength=min_line_length, maxLineGap=max_line_gap,
    )
    if lines is None:
        return image

    line_mask = np.zeros_like(dark_mask)
    for x1, y1, x2, y2 in lines[:, 0]:
        cv2.line(line_mask, (x1, y1), (x2, y2), color=255, thickness=stroke_width)
    # Stay conservative: only inpaint pixels that are both on a detected
    # line AND actually dark, so a thick line drawn near (but not over) a
    # lighter tissue edge doesn't eat into real content.
    arrow_mask = cv2.bitwise_and(line_mask, dark_mask)
    arrow_mask = cv2.dilate(arrow_mask, np.ones((3, 3), np.uint8), iterations=1)

    if arrow_mask.sum() == 0:
        return image

    inpainted = cv2.inpaint(arr, arrow_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    return Image.fromarray(inpainted)
