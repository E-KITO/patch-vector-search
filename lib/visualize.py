"""Visualize search hits: as markers on a low-resolution thumbnail, or as
real-resolution patch crops.

plot_slide_hits_on_thumbnail predates raw WSI pixel access in this project
and only shows hit *locations* as markers over
data/trident_processed/thumbnails/{slide_id}.jpg. plot_hit_patch_gallery
shows what a hit actually looks like, using data/moo_collected_tggate_wsi/
raw_wsi/{slide_id}.svs (now available for the full uni_v1 corpus) via
lib.raw_patch.crop_patch.
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


def plot_hit_patch_gallery(
    hits_df: pd.DataFrame,
    raw_slide_dir: str | Path,
    slide_meta: pd.DataFrame,
    n_cols: int = 5,
) -> plt.Figure:
    """Render each hit as its actual real-resolution WSI patch (via
    lib.raw_patch.crop_patch), arranged in a grid.

    plot_slide_hits_on_thumbnail only shows *where* a hit is, as a dot on a
    low-resolution thumbnail — it can't show what the hit actually looks
    like, since (per its docstring) this project originally had no raw WSI
    pixel access. That's no longer true: data/moo_collected_tggate_wsi/
    raw_wsi/{slide_id}.svs now covers every slide in the uni_v1 corpus, so
    hits can be shown as real patch pixels instead of markers — letting a
    human directly check whether a hit visually looks like the intended
    finding.

    Args:
        hits_df: Rows with slide_id, coord_x, coord_y, similarity (e.g.
            lib.search.PatchIndex.search_similar_patches[_multi] output).
        raw_slide_dir: Directory containing {slide_id}.svs
            (data/moo_collected_tggate_wsi/raw_wsi for the uni_v1 corpus).
        slide_meta: Per-slide metadata indexed by slide_id
            (lib.search.PatchIndex.slide_meta), must contain
            patch_size_level0.
        n_cols: Number of grid columns.

    Returns:
        matplotlib Figure with one real-resolution patch per hit, each
        titled with its slide_id and similarity score.
    """
    from lib.raw_patch import crop_patch

    n = len(hits_df)
    n_rows = -(-n // n_cols)  # ceil division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.4), squeeze=False)
    axes = axes.flatten()

    for ax, row in zip(axes, hits_df.itertuples()):
        patch_size_level0 = int(slide_meta.loc[row.slide_id, "patch_size_level0"])
        patch = crop_patch(row.slide_id, row.coord_x, row.coord_y, raw_slide_dir, patch_size_level0)
        ax.imshow(patch)
        ax.set_title(f"{row.slide_id}\nsim={row.similarity:.3f}", fontsize=8)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    return fig
