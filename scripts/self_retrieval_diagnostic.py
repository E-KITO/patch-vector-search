"""Measure the retrieval ceiling of the uni_v1 corpus + index itself, with the
NTP-atlas query images (and their domain gap / unknown ROI / unknown
magnification) taken out of the picture entirely.

Motivation: scripts/validate_against_ground_truth.py compares pipelines using
NTP-atlas figures as queries, but that conflates (a) whether UNI embeddings
place same-finding patches near each other in this corpus with (b) the
atlas->TG-GATEs domain gap, the fact that an atlas plate is mostly
non-finding tissue, unknown magnification, no ROI. experiments/0013 showed the
hard categories (necrosis, Kupffer) fail upstream of any query preprocessing,
so before investing in uni_v2 (a full GPU re-embed) or a retrieval-algorithm
change we need to know the model/corpus ceiling.

Method: leave-one-out, entirely inside the corpus. For each FINDING_TYPE with
>= MIN_SLIDES_PER_FINDING confirmed single-finding slides in the corpus
(data/processed_csv/single_finding_liver.csv), take each such slide in turn as
the query: read a random sample of its own patch feature vectors straight from
its .h5 (NO re-embedding, NO GPU — these are the exact vectors already in the
index), search the index, drop the query slide from the results, and record
where the *other* same-finding slides land in the ranking. Aggregate per
finding and compare against the random-chance baseline.

Two knobs that matter for interpretation:
  - ranking key: n_hits_ratio (what search_top_slides_multi sorts by) vs
    max_similarity. experiments/0013 finding 2 was that these diverge; this
    script reports rank under both so the divergence is quantified per finding.
  - same-group exclusion: two slides from the same EXP_ID+GROUP_ID share
    compound/dose/timepoint and will look alike for batch reasons, not just
    the finding. Metrics are reported both with all same-finding targets and
    with same-group targets dropped.

Output: outputs/gt_validations/self_retrieval_diagnostic.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.search import PatchIndex

GT_CSV = Path("data/processed_csv/single_finding_liver.csv")
FEATURES_DIR = Path("data/trident_processed/20x_224px_0px_overlap/features_uni_v1")
INDEX_EXP_DIR = Path("outputs/0002_20260808_build_faiss_index/default")
OUT_PATH = Path("outputs/gt_validations/self_retrieval_diagnostic.csv")

SEED = 42
MIN_SLIDES_PER_FINDING = 2       # need >=1 query + >=1 target
MAX_QUERY_SLIDES_PER_FINDING = 10  # cap LOO queries for big findings (targets always = all others)
N_QUERY_PATCHES = 1200           # random patch vectors sampled per query slide
NPROBE = 64
K_CANDIDATES = 8000
HIT_KS = (10, 50)


def load_v1_index() -> PatchIndex:
    return PatchIndex.load(
        index_path=INDEX_EXP_DIR / "index.faiss",
        manifest_path=INDEX_EXP_DIR / "manifest.parquet",
        slide_meta_path=INDEX_EXP_DIR / "slide_meta.parquet",
        features_dir=FEATURES_DIR,
    )


def load_query_vecs(slide_id: str, rng: np.random.Generator) -> np.ndarray:
    """Random sample of a slide's own patch feature vectors, L2-normalized.

    These are read straight from the corpus .h5 — the same vectors that are in
    the index — so this is a pure retrieval test with no encoder in the loop.
    """
    with h5py.File(FEATURES_DIR / f"{slide_id}.h5", "r") as f:
        n = f["features"].shape[0]
        if n <= N_QUERY_PATCHES:
            vecs = f["features"][:].astype(np.float32)
        else:
            rows = np.sort(rng.choice(n, size=N_QUERY_PATCHES, replace=False))
            vecs = f["features"][rows].astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def ranks_of(ranked_slides: list[str], targets: set[str]) -> dict:
    """Positions (1-indexed) of `targets` within an ordered slide list."""
    pos = [i + 1 for i, s in enumerate(ranked_slides) if s in targets]
    n_found = len(pos)
    out = {
        "found": n_found,
        "best_rank": min(pos) if pos else None,
        "mean_rank": round(float(np.mean(pos)), 1) if pos else None,
    }
    for k in HIT_KS:
        out[f"hit@{k}"] = int(any(p <= k for p in pos))
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    pi = load_v1_index()
    n_corpus = len(pi.slide_meta)
    corpus_slides = set(pi.slide_meta.index.astype(str))

    gt = pd.read_csv(GT_CSV)
    gt["slide_id"] = gt["image_id"].str.replace(".svs", "", regex=False)
    gt = gt[gt["slide_id"].isin(corpus_slides)].copy()
    gt["group_key"] = gt["EXP_ID"].astype(str) + "_" + gt["GROUP_ID"].astype(str)
    group_of = dict(zip(gt["slide_id"], gt["group_key"]))

    finding_to_slides = (
        gt.groupby("FINDING_TYPE")["slide_id"].agg(lambda s: sorted(set(s))).to_dict()
    )
    findings = sorted(
        (f for f, s in finding_to_slides.items() if len(s) >= MIN_SLIDES_PER_FINDING),
        key=lambda f: -len(finding_to_slides[f]),
    )

    print(f"corpus slides: {n_corpus}")
    print(f"single-finding corpus slides: {gt['slide_id'].nunique()}")
    print(f"findings with >= {MIN_SLIDES_PER_FINDING} corpus slides: {len(findings)}\n")

    rows = []
    t_all = time.time()
    for finding in findings:
        slides = finding_to_slides[finding]
        query_slides = slides
        if len(slides) > MAX_QUERY_SLIDES_PER_FINDING:
            idx = rng.choice(len(slides), size=MAX_QUERY_SLIDES_PER_FINDING, replace=False)
            query_slides = [slides[i] for i in sorted(idx)]

        per_query = {key: [] for key in ("nhr_all", "sim_all", "nhr_grp", "sim_grp")}
        for q in query_slides:
            q_vecs = load_query_vecs(q, rng)
            ranked = pi.search_top_slides_multi(
                q_vecs, k_candidates=K_CANDIDATES, nprobe=NPROBE, top_n_slides=n_corpus
            )
            ranked = ranked[ranked["slide_id"].astype(str) != q]

            by_nhr = ranked["slide_id"].astype(str).tolist()  # already n_hits_ratio-sorted
            by_sim = (
                ranked.sort_values("max_similarity", ascending=False)["slide_id"].astype(str).tolist()
            )

            targets_all = set(slides) - {q}
            targets_grp = {s for s in targets_all if group_of.get(s) != group_of.get(q)}

            per_query["nhr_all"].append(ranks_of(by_nhr, targets_all))
            per_query["sim_all"].append(ranks_of(by_sim, targets_all))
            if targets_grp:
                per_query["nhr_grp"].append(ranks_of(by_nhr, targets_grp))
                per_query["sim_grp"].append(ranks_of(by_sim, targets_grp))

        def agg(key: str, metric: str):
            vals = [d[metric] for d in per_query[key] if d[metric] is not None]
            if not vals:
                return None
            return round(float(np.median(vals)), 1)

        def hit_rate(key: str, k: int):
            vals = [d[f"hit@{k}"] for d in per_query[key]]
            return round(float(np.mean(vals)), 2) if vals else None

        n_targets_typical = len(slides) - 1
        row = {
            "finding": finding,
            "n_corpus_slides": len(slides),
            "n_query_slides": len(query_slides),
            "random_best_rank": round(n_corpus / (n_targets_typical + 1), 1),
            # n_hits_ratio ranking (what the product currently uses)
            "nhr_best_rank_med": agg("nhr_all", "best_rank"),
            "nhr_mean_rank_med": agg("nhr_all", "mean_rank"),
            "nhr_hit@10": hit_rate("nhr_all", 10),
            "nhr_hit@50": hit_rate("nhr_all", 50),
            # max_similarity ranking (see experiments/0013 finding 2)
            "sim_best_rank_med": agg("sim_all", "best_rank"),
            "sim_mean_rank_med": agg("sim_all", "mean_rank"),
            "sim_hit@10": hit_rate("sim_all", 10),
            "sim_hit@50": hit_rate("sim_all", 50),
            # same-group targets dropped (finding-similarity, not batch-similarity)
            "nhr_best_rank_med_nogrp": agg("nhr_grp", "best_rank"),
            "sim_best_rank_med_nogrp": agg("sim_grp", "best_rank"),
        }
        rows.append(row)
        print(
            f"{finding:38s} n={len(slides):2d}  "
            f"nhr_best={row['nhr_best_rank_med']}  sim_best={row['sim_best_rank_med']}  "
            f"random={row['random_best_rank']}  "
            f"nhr_hit@50={row['nhr_hit@50']}  sim_hit@50={row['sim_hit@50']}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nTOTAL: {time.time() - t_all:.1f}s")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
