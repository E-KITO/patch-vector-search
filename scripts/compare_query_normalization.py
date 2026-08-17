"""Does stain-normalizing a query patch (toward the chosen 'average' reference)
before embedding actually help search quality?

Ad-hoc validation (not a numbered experiment / no Slurm needed — GPU + UNI
weights are already available locally): embeds each data/query/* image both
raw and Macenko-normalized (lib.stain_normalize, target = candidate #01),
runs search_top_slides for each, and checks where the ground-truth slides
for that finding (data/processed_csv/single_finding_liver.csv) land in each
ranking. Mirrors the ground-truth check already done for this project (see
patch-vector-search-project memory).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.query_embedding import embed_image
from lib.search import PatchIndex

INDEX_DIR = Path("outputs/0002_20260808_build_faiss_index/default")
FEATURES_DIR = Path("data/trident_processed/20x_224px_0px_overlap/features_uni_v1")
STAIN_REFERENCE = "outputs/average_patch_candidates/63958_x38976_y7616.png"  # candidate #01

# (query image, FINDING_TYPE substring to match, case-sensitive)
QUERIES = [
    ("data/query/necrosis/focul_necrosis.jpg", "necrosis"),
    ("data/query/necrosis/focul_necrosis_CUT.jpg", "necrosis"),
    ("data/query/degeneration,fatty/26513_x22272_y1024.png", "fatty"),
    ("data/query/celluar,infiltration/27753_x9728_y36608.png", "infiltration"),
]


def ground_truth_slides(keyword: str, corpus_ids: set[str]) -> set[str]:
    df = pd.read_csv("data/processed_csv/single_finding_liver.csv")
    mask = df["FINDING_TYPE"].str.contains(keyword, case=False, na=False)
    slides = set(df.loc[mask, "image_id"].str.replace(".svs", "", regex=False))
    return slides & corpus_ids


def rank_report(top_slides: pd.DataFrame, gt_slides: set[str], label: str) -> None:
    top_slides = top_slides.reset_index(drop=True)
    top_slides["rank"] = top_slides.index + 1
    hits = top_slides[top_slides["slide_id"].astype(str).isin(gt_slides)]
    print(f"\n  [{label}] ground-truth hits: {len(hits)}/{len(gt_slides)}")
    if len(hits):
        print("   " + hits[["rank", "slide_id", "n_hits", "n_hits_ratio", "max_similarity"]]
              .to_string(index=False).replace("\n", "\n   "))
        print(f"   best rank: {hits['rank'].min()}  mean rank: {hits['rank'].mean():.1f}")
    else:
        print("   (none in pool)")


def main():
    patch_index = PatchIndex.load(
        index_path=INDEX_DIR / "index.faiss",
        manifest_path=INDEX_DIR / "manifest.parquet",
        slide_meta_path=INDEX_DIR / "slide_meta.parquet",
        features_dir=FEATURES_DIR,
    )
    corpus_ids = set(patch_index.slide_meta.index.astype(str))

    for image_path, keyword in QUERIES:
        gt_slides = ground_truth_slides(keyword, corpus_ids)
        print(f"\n{'=' * 70}\n{image_path}  (finding keyword: '{keyword}', "
              f"{len(gt_slides)} ground-truth slides in corpus)\n{'=' * 70}")

        raw_vec = embed_image(image_path)
        norm_vec = embed_image(image_path, stain_reference=STAIN_REFERENCE)
        print("cosine sim between raw and normalized query embeddings:", float(raw_vec @ norm_vec))

        raw_top = patch_index.search_top_slides(raw_vec, k_candidates=8000, nprobe=64, top_n_slides=998)
        norm_top = patch_index.search_top_slides(norm_vec, k_candidates=8000, nprobe=64, top_n_slides=998)

        overlap = len(set(raw_top["slide_id"].head(20)) & set(norm_top["slide_id"].head(20)))
        print(f"top-20 slide overlap (raw vs normalized): {overlap}/20")

        if gt_slides:
            rank_report(raw_top, gt_slides, "raw")
            rank_report(norm_top, gt_slides, "normalized")
        else:
            print("  no ground-truth slides for this finding inside the 998-slide corpus "
                  "(known limitation, see patch-vector-search-project memory) — "
                  "reporting embedding/top-slide shift only.")


if __name__ == "__main__":
    main()
