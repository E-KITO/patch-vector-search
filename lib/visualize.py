"""Visualize search hits on a slide's low-resolution thumbnail.

There is no raw WSI pixel access in this project (no openslide-readable
.svs files) — data/trident_processed/thumbnails/{slide_id}.jpg is the only
per-slide visual asset available, so hits are shown as markers over it
rather than as cropped patch images.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def plot_slide_hits_on_thumbnail(
    slide_id: str,
    hits_df: pd.DataFrame,
    thumbnails_dir: str | Path,
    slide_meta: pd.DataFrame,
    marker_size: int = 40,
) -> plt.Figure:
    """Overlay hit patch locations on a slide's thumbnail JPEG.

    Args:
        slide_id: Slide to visualize.
        hits_df: Rows for this slide only, with level-0 pixel coord_x/coord_y columns
            (e.g. lib.search.PatchIndex.search_similar_patches output, filtered to slide_id).
        thumbnails_dir: Directory containing {slide_id}.jpg thumbnails.
        slide_meta: Per-slide metadata indexed by slide_id (lib.search.PatchIndex.slide_meta),
            must contain level0_width/level0_height.
        marker_size: Scatter marker size.

    Returns:
        matplotlib Figure with the thumbnail and hit markers.
    """
    thumbnail = Image.open(Path(thumbnails_dir) / f"{slide_id}.jpg")
    thumb_w, thumb_h = thumbnail.size

    meta = slide_meta.loc[slide_id]
    scale_x = thumb_w / meta["level0_width"]
    scale_y = thumb_h / meta["level0_height"]

    fig, ax = plt.subplots(figsize=(thumb_w / 100, thumb_h / 100))
    ax.imshow(thumbnail)
    ax.scatter(
        hits_df["coord_x"] * scale_x,
        hits_df["coord_y"] * scale_y,
        s=marker_size,
        facecolors="none",
        edgecolors="red",
        linewidths=1.5,
    )
    ax.set_title(f"{slide_id} — {len(hits_df)} similar patches")
    ax.axis("off")
    fig.tight_layout()
    return fig
