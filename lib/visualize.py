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


def plot_query_tile_scores(
    image: str | Path | Image.Image,
    patch_index,
    tile_size: int = 224,
    native_tile_size: int | None = None,
    stride: int | None = None,
    nprobe: int = 32,
    max_tiles_reranked: int | None = None,
    device: str | None = None,
) -> plt.Figure:
    """Overlay every grid tile of a query image with its approximate FAISS
    top-1 score, as a heatmap.

    lib.search.PatchIndex.search_similar_patches_multi ranks a query
    image's tiles by this exact score and only exact-reranks the top
    `max_tiles_reranked` of them (see its docstring) — every other tile,
    however diagnostic its content, is silently dropped before that point
    and never contributes to similar_patches/hit_pool at all. This makes
    that selection visible: measured on a real query (see the "necrosis
    imgi_11" case in this project's conversation notes — a 60/40 split of
    normal/necrotic tissue where none of the 4 confirmed ground-truth
    necrosis slides made top_slides at all), a distinctive but locally
    uncommon finding can lose this ranking to "generic normal tissue"
    tiles, which score deceptively high simply because near-duplicates of
    normal tissue are abundant everywhere in the corpus. A tile with a low
    score here could not have made it into search_similar_patches_multi's
    shortlist regardless of what it actually shows.

    Reimplements lib.query_embedding.embed_image_tiles's own grid/
    blank-tile logic exactly (same tile_size/native_tile_size/stride
    handling, same _is_blank_tile filter) rather than calling it directly,
    because that function doesn't return each tile's grid position, which
    this needs to draw the overlay. Keep the tiling arguments identical to
    whatever embed_image_tiles call the real search used, or the tiles
    shown here won't correspond to what was actually searched.

    Args:
        image: Path/Image, the query image to tile.
        patch_index: lib.search.PatchIndex to score tiles against (must be
            the index actually queried, so scores match).
        tile_size / native_tile_size / stride: see embed_image_tiles.
        nprobe: IVF clusters probed — matches the nprobe actually used for
            search, since the approximate score depends on it.
        max_tiles_reranked: if given, the top-N tiles by score (i.e. the
            ones search_similar_patches_multi would actually exact-rerank)
            get a red outline.
        device: "cuda"/"cpu" for the UNI forward pass.

    Returns:
        matplotlib Figure: the query image with a semi-transparent
        per-tile score heatmap overlaid (blank tiles left unshaded — they
        never reach scoring, same as in the real search), a colorbar, and
        (if max_tiles_reranked is given) red outlines on the tiles that
        actually get reranked.
    """
    import matplotlib.patches as mpatches
    import torch
    from matplotlib.colors import Normalize
    from torchvision import transforms

    import numpy as np

    from lib.query_embedding import _is_blank_tile, _load_encoder, _load_rgb

    encoder, device, dtype = _load_encoder(device)
    pil_image = _load_rgb(image)

    crop_size = native_tile_size or tile_size
    w, h = pil_image.size
    scale = max(crop_size / w, crop_size / h, 1.0)
    if scale > 1.0:
        pil_image = pil_image.resize(
            (max(crop_size, round(w * scale)), max(crop_size, round(h * scale))), Image.LANCZOS
        )
        w, h = pil_image.size

    stride = stride or crop_size
    xs = sorted(set(list(range(0, w - crop_size + 1, stride)) + [w - crop_size]))
    ys = sorted(set(list(range(0, h - crop_size + 1, stride)) + [h - crop_size]))
    origins = [(x, y) for y in ys for x in xs]
    crops = [pil_image.crop((x, y, x + crop_size, y + crop_size)) for x, y in origins]

    blank = [_is_blank_tile(c) for c in crops]
    if all(blank):  # mirror embed_image_tiles: don't filter if it would empty the set
        blank = [False] * len(blank)

    scored_origins = [o for o, b in zip(origins, blank) if not b]
    scored_crops = [c for c, b in zip(crops, blank) if not b]
    if crop_size != tile_size:
        scored_crops = [c.resize((tile_size, tile_size), Image.LANCZOS) for c in scored_crops]

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    tensors = torch.stack([to_tensor(c) for c in scored_crops]).to(device=device, dtype=dtype)
    with torch.no_grad():
        embeddings = encoder(tensors).float().cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    patch_index._set_nprobe(nprobe)
    scores = patch_index.index.search(embeddings, 1)[0][:, 0]

    top_idx = set()
    if max_tiles_reranked is not None:
        top_idx = set(np.argsort(scores)[::-1][:max_tiles_reranked].tolist())

    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=float(scores.min()), vmax=float(scores.max()))

    fig, ax = plt.subplots(figsize=(w / 100, h / 100))
    ax.imshow(pil_image)
    for i, ((x, y), score) in enumerate(zip(scored_origins, scores)):
        ax.add_patch(mpatches.Rectangle(
            (x, y), crop_size, crop_size, facecolor=cmap(norm(score)), alpha=0.5,
            edgecolor="red" if i in top_idx else "none", linewidth=2.5,
        ))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="approx. top-1 FAISS score", shrink=0.7)

    n_blank = sum(blank)
    title = "Per-tile approximate FAISS score"
    if max_tiles_reranked is not None:
        title += f" (red outline = top {max_tiles_reranked} reranked)"
    if n_blank:
        title += f"\n{n_blank} blank tile(s) excluded (unshaded)"
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    fig.tight_layout()
    return fig
