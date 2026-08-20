"""One-off: crop a single 224x224 tile from squarely inside the necrotic
zone of data/query/necrosis_imgi11_a30016.jpg (itself a copy of the NNL
atlas's "Liver - Necrosis" imgi_11_figure-006-a30016_large.jpg, 1800x1171px).

Purpose: bypass the tile-selection problem entirely (see this project's
lib.visualize.plot_query_tile_scores heatmap diagnostic -- the necrotic
region's tiles score lower than plain normal-tissue tiles on FAISS's
approximate top-1 metric and never make search_similar_patches_multi's
top-`max_tiles_reranked` shortlist) by handing the necrotic tile to the
search directly as a single 224x224 query image, instead of relying on
tiling+ranking to find it inside the full 1800x1171 figure.

Crop box (x=150..374, y=400..624) chosen to sit well inside the necrotic
zone (left ~0-700px of the source image) and away from its boundary with
normal tissue (~700-750px) and the image edges.

No GPU/UNI/torch needed -- pure PIL, runs in well under a second. Not a
Slurm job.

Usage:
    .venv/bin/python3 scripts/crop_necrosis_query_tile.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

SOURCE = Path("data/query/necrosis_imgi11_a30016.jpg")
CROP_BOX = (150, 400, 374, 624)  # (left, top, right, bottom), 224x224
OUT_PATH = Path("data/query/necrosis_imgi11_crop_x150_y400.jpg")


def main() -> None:
    image = Image.open(SOURCE).convert("RGB")
    crop = image.crop(CROP_BOX)
    assert crop.size == (224, 224), f"unexpected crop size: {crop.size}"
    crop.save(OUT_PATH)
    print(f"wrote {OUT_PATH} ({crop.size[0]}x{crop.size[1]})")


if __name__ == "__main__":
    main()
