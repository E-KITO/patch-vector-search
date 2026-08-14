"""Crop real patch pixels from raw WSI files (data/raw_slide/*.svs).

Not required by the core search pipeline (lib/manifest.py, lib/faiss_index.py,
lib/search.py all operate on precomputed UNI features only) — this is a
visual QC helper for looking at what a search hit actually looks like,
usable once raw_slide/*.svs is present for the slide in question.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image


def crop_patch(
    slide_id: str,
    coord_x: int,
    coord_y: int,
    raw_slide_dir: str | Path,
    patch_size_level0: int,
    target_size: int = 224,
) -> Image.Image:
    """Read the exact pixels TRIDENT would have extracted for this patch.

    (coord_x, coord_y) and patch_size_level0 are level-0 (native resolution)
    pixel coordinates, matching lib.manifest's manifest.parquet /
    slide_meta.parquet columns — read_region always takes level-0
    coordinates for `location` regardless of which pyramid level is read.

    Args:
        slide_id: Slide to read from data/raw_slide/{slide_id}.svs.
        coord_x: Level-0 x coordinate (top-left of the patch).
        coord_y: Level-0 y coordinate (top-left of the patch).
        raw_slide_dir: Directory containing {slide_id}.svs.
        patch_size_level0: Patch side length in level-0 pixels (from
            slide_meta.parquet; varies per slide if native magnification
            differs from the target magnification used at extraction time).
        target_size: Resize the crop to this many pixels per side (224 to
            match the size UNI was fed at extraction time).

    Returns:
        RGB PIL Image, target_size x target_size.
    """
    import openslide

    slide_path = Path(raw_slide_dir) / f"{slide_id}.svs"
    with openslide.OpenSlide(str(slide_path)) as slide:
        region = slide.read_region(
            (int(coord_x), int(coord_y)), 0, (patch_size_level0, patch_size_level0)
        ).convert("RGB")
    if region.size != (target_size, target_size):
        region = region.resize((target_size, target_size), Image.LANCZOS)
    return region
