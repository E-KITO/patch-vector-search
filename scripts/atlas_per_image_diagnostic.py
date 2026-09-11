"""atlas GT 比較を図版1枚単位に分解して比較する診断スクリプト。

`validate_against_ground_truth.py` はカテゴリ内の全図版タイルを1回のマルチタイル
クエリに束ねてしまうため、カテゴリ単位の best_rank の悪化が「特定の外れ値図版1枚が
足を引っ張っている」のか「そのカテゴリの図版すべてに一様に効いている」のか区別できない。
experiments/0025 で whiten 索引が atlas GT を悪化させた原因の切り分け(ドメインギャップ
起因の外れ値なのか、変換自体が構造的に atlas クエリと相性が悪いのか)のために、図版
1枚ずつ個別に `search_top_slides_multi` を呼び、複数索引(例: baseline / whiten)を
並べて比較する。

Usage:
    .venv/bin/python3 scripts/atlas_per_image_diagnostic.py \
        --index-dirs baseline=outputs/0018_20260909_build_faiss_index_deblank/default,whiten=outputs/0025_20260910_whiten_index_rebuild/default/whiten_v1 \
        --out outputs/atlas_per_image_diagnostic.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from lib.atlas_figures import query_images
from lib.query_embedding import embed_image_tiles
from lib.search import PatchIndex
from scripts.validate_against_ground_truth import ATLAS_DIR, CATEGORIES, GT_CSV, rank_stats

FEATURES_DIR = Path("data/trident_processed/20x_224px_0px_overlap/features_uni_v1")


def load_index(exp_dir: str | Path) -> PatchIndex:
    exp_dir = Path(exp_dir)
    return PatchIndex.load(
        index_path=exp_dir / "index.faiss",
        manifest_path=exp_dir / "manifest.parquet",
        slide_meta_path=exp_dir / "slide_meta.parquet",
        features_dir=FEATURES_DIR,
    )


def parse_index_dirs(spec: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for part in spec.split(","):
        name, _, path = part.partition("=")
        if not name or not path:
            raise SystemExit(f"bad --index-dirs entry: {part!r} (want name=path)")
        out[name.strip()] = Path(path.strip())
    return out


def run(index_dirs: dict[str, Path], nprobe: int = 64) -> pd.DataFrame:
    print(f"loading {len(index_dirs)} indices: {list(index_dirs)}", flush=True)
    indices = {name: load_index(p) for name, p in index_dirs.items()}
    any_index = next(iter(indices.values()))
    corpus_slide_ids = set(any_index.slide_meta.index.astype(str))
    n_slides = len(corpus_slide_ids)

    gt = pd.read_csv(GT_CSV)
    gt["slide_id"] = gt["image_id"].str.replace(".svs", "", regex=False)

    rows = []
    for cat_dir_name, finding_type in CATEGORIES.items():
        cat_dir = ATLAS_DIR / cat_dir_name
        images = query_images(cat_dir)
        if not images:
            print(f"WARNING: no images found for {cat_dir_name!r}, skipping")
            continue
        gt_slides = set(gt.loc[gt.FINDING_TYPE == finding_type, "slide_id"]) & corpus_slide_ids
        if not gt_slides:
            print(f"WARNING: no in-corpus GT slides for {finding_type!r}, skipping")
            continue

        for img in images:
            tiles = embed_image_tiles(str(img), tile_size=224)
            row = {"finding": finding_type, "image": img.name, "n_gt": len(gt_slides)}
            for name, pi in indices.items():
                ranked = pi.search_top_slides_multi(
                    tiles, k_candidates=8000, nprobe=nprobe, top_n_slides=n_slides
                )
                s = rank_stats(ranked, gt_slides)
                row[f"{name}_found"] = s["found"]
                row[f"{name}_best"] = s["best_rank"]
                row[f"{name}_mean"] = s["mean_rank"]
            rows.append(row)
            print(row, flush=True)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--index-dirs",
        required=True,
        help="comma-separated name=path pairs, e.g. "
        "'baseline=outputs/0018_.../default,whiten=outputs/0025_.../default/whiten_v1'",
    )
    ap.add_argument("--out", type=Path, default=Path("outputs/atlas_per_image_diagnostic.csv"))
    ap.add_argument("--nprobe", type=int, default=64)
    args = ap.parse_args()

    df = run(parse_index_dirs(args.index_dirs), nprobe=args.nprobe)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
