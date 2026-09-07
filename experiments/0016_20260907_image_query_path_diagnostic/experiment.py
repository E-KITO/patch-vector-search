"""Isolate the query-embedding path as a confound in the 0014 atlas conclusion.

0014 concluded "atlas figures as queries can't retrieve the finding's GT slides
-> atlas->TG-GATEs domain gap is the #1 blocker". But the atlas arm and the
TG-GATEs arm (0015 / scripts/self_retrieval_diagnostic.py) differ on two axes at
once: (1) content/domain, and (2) the query path itself -- pixels through
lib.query_embedding.embed_image(_tiles) (tiling, resize, blank filter) vs a
direct read of the stored h5 feature vectors.

This experiment measures axis (2) alone: take patches that are *already in the
corpus* (so the domain gap is zero by construction) and check whether querying
with them via the image path retrieves as well as querying with their stored
vectors.

  Tier 1 "roundtrip": re-embed corpus GT patches from raw WSI pixels, compare to
    the stored h5 vector (cos_self), then compare retrieval (own patch / own
    slide / other same-finding GT slides) between the re-embedded vector and the
    stored vector, paired on the same physical patches.
  Tier 2 "patchset": run the 0015 validate pipeline twice on the *same* seed
    patch set -- once seeding from stored h5 vectors, once from image-path
    re-embeddings -- and compare hold-out GT recall.
  Tier 3 "region": crop a large level-0 region (N x N patches) from a GT slide
    and query with embed_image_tiles, reproducing the atlas plate's ROI-less
    multi-tile structure but in-domain and at matched magnification. Compared
    against Tier 1b (individual exact patches) this isolates the cost of "tile a
    big image and aggregate" from the cost of the domain gap.

If image-path in-domain retrieval holds up (Tier 1b/2 image arm ~= feature arm,
Tier 3 only mildly worse), 0014's atlas failure is cleanly attributable to the
atlas figures themselves. If it degrades materially, part of what 0014 attributed
to "domain gap" is really the query pipeline.
"""
import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import yaml

import numpy as np
import pandas as pd

HIT_KS = (10, 50)


def _get_project_root() -> Path:
    project_root = os.environ.get("PROJECT_ROOT")
    if not project_root:
        print("Error: PROJECT_ROOT is not set. Run via run_slurm.sh.", file=sys.stderr)
        sys.exit(1)
    return Path(project_root)


def _staged_or(project_root: Path, config_rel: str, env_var: str) -> Path:
    """A data path, preferring a local-SSD staged copy (env_var, set by
    run_slurm.sh's PRE_NATIVE_COMMAND) over the NFS original. The FAISS index and
    per-slide .h5 feature files are read at high frequency during exact
    re-ranking (Tier 1c, Tier 2), which must not run against NFS — see the
    storage policy in run_slurm.sh."""
    staged = os.environ.get(env_var)
    if staged and Path(staged).exists():
        print(f"{env_var}: using staged copy {staged}")
        return Path(staged)
    return project_root / config_rel


def setup_logger(run_dir: Path, name: str = "experiment") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(run_dir / "experiment.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def load_config(exp_dir: Path) -> dict:
    config_path = exp_dir / "config.yml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="config.yml")
    return parser.parse_args()


def slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()


def _l2norm_rows(v: np.ndarray) -> np.ndarray:
    v = np.atleast_2d(v).astype(np.float32)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


def corpus_gt_slides(gt: pd.DataFrame, finding: str, corpus_slides: set[str]) -> list[str]:
    return sorted(set(gt.loc[gt["FINDING_TYPE"] == finding, "slide_id"]) & corpus_slides)


def sample_slide_patch_rows(
    manifest: pd.DataFrame, slide_id: str, n: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Random rows (local_idx, coord_x, coord_y) of one slide's patches, sorted
    by local_idx (h5 fancy-indexing needs increasing indices)."""
    rows = manifest.loc[manifest["slide_id"] == slide_id, ["local_idx", "coord_x", "coord_y"]]
    if len(rows) > n:
        rows = rows.iloc[np.sort(rng.choice(len(rows), size=n, replace=False))]
    return rows.sort_values("local_idx").reset_index(drop=True)


def stored_vecs(features_dir: Path, slide_id: str, local_idx: np.ndarray) -> np.ndarray:
    """The exact index vectors for these patches, straight from the corpus h5."""
    import h5py

    order = np.argsort(local_idx)
    with h5py.File(features_dir / f"{slide_id}.h5", "r") as f:
        v = f["features"][np.asarray(local_idx)[order]].astype(np.float32)
    out = np.empty_like(v)
    out[order] = v
    return _l2norm_rows(out)


def image_vecs(
    raw_slide_dir: Path, slide_id: str, rows: pd.DataFrame, patch_size_level0: int,
    encoder_name: str, logger: logging.Logger,
) -> np.ndarray:
    """Re-embed each patch from raw WSI pixels via the query image path."""
    from lib.query_embedding import embed_image
    from lib.raw_patch import crop_patch

    out = []
    for r in rows.itertuples():
        patch = crop_patch(slide_id, int(r.coord_x), int(r.coord_y), raw_slide_dir, patch_size_level0)
        out.append(embed_image(patch, encoder_name=encoder_name, resize_mode="centercrop"))
    return _l2norm_rows(np.stack(out))


def ranks_of(ranked_slides: list[str], targets: set[str]) -> dict:
    pos = [i + 1 for i, s in enumerate(ranked_slides) if s in targets]
    out = {"n_targets": len(targets), "found": len(pos),
           "best_rank": min(pos) if pos else None,
           "mean_rank": round(float(np.mean(pos)), 1) if pos else None}
    for k in HIT_KS:
        out[f"hit@{k}"] = int(any(p <= k for p in pos))
    return out


def _clean(xs: list) -> list[float]:
    """Drop None and NaN (pandas turns None into NaN inside a numeric column, so
    an `is not None` filter alone lets NaN through and poisons np.median/mean)."""
    return [float(x) for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]


def _med(xs: list) -> float | None:
    xs = _clean(xs)
    return round(float(np.median(xs)), 2) if xs else None


def _rate(xs: list) -> float | None:
    xs = _clean(xs)
    return round(float(np.mean(xs)), 3) if xs else None


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1: roundtrip
# ─────────────────────────────────────────────────────────────────────────────
def tier_roundtrip(pi, cfg, findings, gt, corpus_slides, features_dir, raw_slide_dir,
                   encoder_name, nprobe, out_dir, rng, logger) -> dict:
    t = cfg["tier1"]
    n_patches = int(t["n_patches_per_slide"])
    n_selfrank = int(t["n_patches_selfrank"])
    selfrank_rerank_pool = int(t["selfrank_rerank_pool"])
    k_candidates_slide = int(t["k_candidates_slide"])

    slide_meta = pi.slide_meta.copy()
    slide_meta.index = slide_meta.index.astype(str)
    n_corpus = len(slide_meta)

    fidelity_rows = []   # per patch
    selfrank_rows = []    # per patch (subsample)
    slide_rows = []       # per (finding, query slide, arm)

    for finding in findings:
        gt_slides = corpus_gt_slides(gt, finding, corpus_slides)
        if len(gt_slides) < 2:
            logger.info(f"[tier1] {finding!r}: only {len(gt_slides)} corpus GT slide(s), skipping")
            continue
        logger.info(f"[tier1] {finding!r}: {len(gt_slides)} corpus GT slides {gt_slides}")

        for q in gt_slides:
            psl = int(slide_meta.loc[q, "patch_size_level0"])
            rows = sample_slide_patch_rows(pi.manifest, q, n_patches, rng)
            local_idx = rows["local_idx"].to_numpy()

            v_feat = stored_vecs(features_dir, q, local_idx)
            v_img = image_vecs(raw_slide_dir, q, rows, psl, encoder_name, logger)

            # 1a: embedding fidelity, paired per patch
            cos_self = np.sum(v_feat * v_img, axis=1)
            for r, c in zip(rows.itertuples(), cos_self):
                fidelity_rows.append({
                    "finding": finding, "slide_id": q,
                    "local_idx": int(r.local_idx), "coord_x": int(r.coord_x),
                    "coord_y": int(r.coord_y), "cos_self": round(float(c), 5),
                })

            # 1c: does the re-embedded vector still retrieve its own patch?
            sub = np.sort(rng.choice(len(rows), size=min(n_selfrank, len(rows)), replace=False))
            for arm, V in (("feature", v_feat), ("image", v_img)):
                for j in sub:
                    r = rows.iloc[j]
                    hits = pi.search_similar_patches(
                        V[j], k=selfrank_rerank_pool, nprobe=nprobe,
                        rerank_pool=selfrank_rerank_pool,
                    )
                    hits["slide_id"] = hits["slide_id"].astype(str)
                    match = hits.index[
                        (hits["slide_id"] == q)
                        & (hits["coord_x"] == int(r["coord_x"]))
                        & (hits["coord_y"] == int(r["coord_y"]))
                    ]
                    patch_rank = int(match[0]) + 1 if len(match) else None
                    slide_match = hits.index[hits["slide_id"] == q]
                    selfrank_rows.append({
                        "finding": finding, "slide_id": q, "arm": arm,
                        "local_idx": int(r["local_idx"]),
                        "patch_self_rank": patch_rank,
                        "own_slide_first_rank": int(slide_match[0]) + 1 if len(slide_match) else None,
                        "top1_similarity": round(float(hits["similarity"].iloc[0]), 4),
                    })

            # 1b: slide-level self-retrieval, paired (mirrors self_retrieval_diagnostic)
            targets = set(gt_slides) - {q}
            for arm, V in (("feature", v_feat), ("image", v_img)):
                ranked = pi.search_top_slides_multi(
                    V, k_candidates=k_candidates_slide, nprobe=nprobe, top_n_slides=n_corpus,
                )
                ranked_ids = ranked["slide_id"].astype(str).tolist()
                self_rank = ranked_ids.index(q) + 1 if q in ranked_ids else None
                tgt = ranks_of([s for s in ranked_ids if s != q], targets)
                slide_rows.append({
                    "finding": finding, "query_slide": q, "arm": arm,
                    "self_rank": self_rank, **tgt,
                })

    fidelity = pd.DataFrame(fidelity_rows)
    selfrank = pd.DataFrame(selfrank_rows)
    slides = pd.DataFrame(slide_rows)
    fidelity.to_csv(out_dir / "tier1_fidelity.csv", index=False)
    selfrank.to_csv(out_dir / "tier1_patch_selfrank.csv", index=False)
    slides.to_csv(out_dir / "tier1_slide_retrieval.csv", index=False)

    summary = {"per_finding": {}}
    for finding in slides["finding"].unique() if len(slides) else []:
        fp = fidelity[fidelity["finding"] == finding]["cos_self"]
        sr = selfrank[selfrank["finding"] == finding]
        sl = slides[slides["finding"] == finding]
        row = {
            "n_patches": int(len(fp)),
            "cos_self_median": round(float(fp.median()), 5) if len(fp) else None,
            "cos_self_p05": round(float(fp.quantile(0.05)), 5) if len(fp) else None,
            "cos_self_min": round(float(fp.min()), 5) if len(fp) else None,
        }
        for arm in ("feature", "image"):
            a_sr = sr[sr["arm"] == arm]
            a_sl = sl[sl["arm"] == arm]
            psr = a_sr["patch_self_rank"]
            ssr = a_sl["self_rank"]
            row[arm] = {
                "n_selfrank_patches": int(len(psr)),
                "patch_self_rank_median": _med(psr.tolist()),
                "patch_self_rank_is_1_rate": round(float((psr == 1).mean()), 3) if len(psr) else None,
                "patch_self_rank_found_rate": round(float(psr.notna().mean()), 3) if len(psr) else None,
                "slide_self_rank_median": _med(ssr.tolist()),
                "slide_self_rank_is_1_rate": round(float((ssr == 1).mean()), 3) if len(ssr) else None,
                "slide_self_rank_found_rate": round(float(ssr.notna().mean()), 3) if len(ssr) else None,
                "target_best_rank_median": _med(a_sl["best_rank"].tolist()),
                **{f"target_hit@{k}_rate": _rate(a_sl[f"hit@{k}"].tolist()) for k in HIT_KS},
            }
        summary["per_finding"][finding] = row
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2: patchset (0015 validate, feature-seed vs image-seed)
# ─────────────────────────────────────────────────────────────────────────────
def tier_patchset(pi, cfg, findings, gt, corpus_slides, features_dir, raw_slide_dir,
                  encoder_name, nprobe, seed, out_dir, logger) -> dict:
    from lib.patch_set import build_patch_set

    t = cfg["tier2"]
    slide_meta = pi.slide_meta.copy()
    slide_meta.index = slide_meta.index.astype(str)

    summary = {"per_finding": {}}
    for finding in findings:
        gt_slides = corpus_gt_slides(gt, finding, corpus_slides)
        if len(gt_slides) < 2:
            logger.info(f"[tier2] {finding!r}: only {len(gt_slides)} corpus GT slide(s), skipping")
            continue

        # Seed / hold-out split -- identical rule to experiments/0015 validate.
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(gt_slides))
        n_seed = max(1, min(len(gt_slides) - 1,
                            round(float(t["seed_fraction"]) * len(gt_slides))))
        seed_pool = [gt_slides[i] for i in perm[:n_seed]]
        seed_slides = sorted(seed_pool[: int(t["max_seed_slides"])])
        holdout_slides = sorted(set(gt_slides) - set(seed_slides))
        exclude_slides = set(seed_slides)
        gt_positive_slides = set(holdout_slides)
        logger.info(
            f"[tier2] {finding!r}: seed {seed_slides}  hold-out {holdout_slides}"
        )

        # Sample the seed patches ONCE per seed slide, then build both query
        # vector sets from the same physical patches (paired comparison).
        n_per = int(t["n_seed_patches_per_slide"])
        feat_parts, img_parts = [], []
        for sid in seed_slides:
            psl = int(slide_meta.loc[sid, "patch_size_level0"])
            rows = sample_slide_patch_rows(pi.manifest, sid, n_per, rng)
            feat_parts.append(stored_vecs(features_dir, sid, rows["local_idx"].to_numpy()))
            img_parts.append(image_vecs(raw_slide_dir, sid, rows, psl, encoder_name, logger))
        arms = {
            "feature": _l2norm_rows(np.concatenate(feat_parts)),
            "image": _l2norm_rows(np.concatenate(img_parts)),
        }

        finding_slug = slugify(finding)
        per_arm = {}
        for arm, query_vecs in arms.items():
            stats = build_patch_set(
                pi, query_vecs,
                out_dir=out_dir / f"{finding_slug}__{arm}",
                raw_slide_dir=raw_slide_dir,
                exclude_slides=exclude_slides,
                gt_positive_slides=gt_positive_slides,
                nprobe=nprobe,
                k_candidate_patches=int(t["k_candidate_patches"]),
                rerank_pool=int(t["rerank_pool"]),
                max_tiles_reranked=t["max_tiles_reranked"],
                sim_floor=float(t["sim_floor"]),
                nms_radius_patches=float(t["nms_radius_patches"]),
                max_per_slide=int(t["max_per_slide"]),
                target_n_patches=int(t["target_n_patches"]),
            )
            per_arm[arm] = stats
            logger.info(
                f"[tier2] {finding!r} {arm}: {stats['n_final_patches']} patches, "
                f"{stats['n_contributing_slides_heldout_gt']}/{len(holdout_slides)} hold-out GT "
                f"slides recovered, frac_gt={stats['frac_final_patches_from_heldout_gt_slide']}"
            )

        f_gt = set(per_arm["feature"]["heldout_gt_contributing_slides"])
        i_gt = set(per_arm["image"]["heldout_gt_contributing_slides"])
        summary["per_finding"][finding] = {
            "seed_slides": seed_slides,
            "holdout_slides": holdout_slides,
            "n_query_vectors": {k: int(v.shape[0]) for k, v in arms.items()},
            "feature": per_arm["feature"],
            "image": per_arm["image"],
            "heldout_gt_recovered": {
                "feature": sorted(f_gt), "image": sorted(i_gt),
                "shared": sorted(f_gt & i_gt),
                "feature_only": sorted(f_gt - i_gt),
                "image_only": sorted(i_gt - f_gt),
            },
        }
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3: region (atlas-plate-like ROI-less tiling, in-domain)
# ─────────────────────────────────────────────────────────────────────────────
def tier_region(pi, cfg, findings, gt, corpus_slides, raw_slide_dir, encoder_name,
                nprobe, out_dir, rng, logger) -> dict:
    import openslide

    from lib.query_embedding import embed_image_tiles

    t = cfg["tier3"]
    region_tiles = int(t["region_tiles"])
    n_regions = int(t["n_regions_per_slide"])
    k_candidates_slide = int(t["k_candidates_slide"])

    slide_meta = pi.slide_meta.copy()
    slide_meta.index = slide_meta.index.astype(str)
    n_corpus = len(slide_meta)

    region_rows = []
    for finding in findings:
        gt_slides = corpus_gt_slides(gt, finding, corpus_slides)
        if len(gt_slides) < 2:
            logger.info(f"[tier3] {finding!r}: only {len(gt_slides)} corpus GT slide(s), skipping")
            continue
        logger.info(f"[tier3] {finding!r}: {len(gt_slides)} corpus GT slides")

        for q in gt_slides:
            psl = int(slide_meta.loc[q, "patch_size_level0"])
            side = region_tiles * psl
            w = int(slide_meta.loc[q, "level0_width"])
            h = int(slide_meta.loc[q, "level0_height"])
            # Centre each region on a real tissue patch so it lands on tissue,
            # not slide background.
            centres = sample_slide_patch_rows(pi.manifest, q, n_regions, rng)
            slide_path = Path(raw_slide_dir) / f"{q}.svs"
            targets = set(gt_slides) - {q}
            for ci, c in enumerate(centres.itertuples()):
                x0 = int(np.clip(int(c.coord_x) + psl // 2 - side // 2, 0, max(0, w - side)))
                y0 = int(np.clip(int(c.coord_y) + psl // 2 - side // 2, 0, max(0, h - side)))
                with openslide.OpenSlide(str(slide_path)) as s:
                    region = s.read_region((x0, y0), 0, (side, side)).convert("RGB")
                tiles = embed_image_tiles(
                    region, tile_size=224, native_tile_size=psl, encoder_name=encoder_name,
                )
                ranked = pi.search_top_slides_multi(
                    tiles, k_candidates=k_candidates_slide, nprobe=nprobe, top_n_slides=n_corpus,
                )
                ranked_ids = ranked["slide_id"].astype(str).tolist()
                self_rank = ranked_ids.index(q) + 1 if q in ranked_ids else None
                tgt = ranks_of([s for s in ranked_ids if s != q], targets)
                region_rows.append({
                    "finding": finding, "query_slide": q, "region_idx": ci,
                    "x0": x0, "y0": y0, "side_px": side, "n_tiles": int(tiles.shape[0]),
                    "self_rank": self_rank, **tgt,
                })
                logger.info(
                    f"[tier3] {finding!r} {q} region {ci}: {tiles.shape[0]} tiles, "
                    f"self_rank={self_rank}, targets found {tgt['found']}/{tgt['n_targets']} "
                    f"(best {tgt['best_rank']})"
                )

    regions = pd.DataFrame(region_rows)
    regions.to_csv(out_dir / "tier3_region_retrieval.csv", index=False)

    summary = {"per_finding": {}}
    for finding in regions["finding"].unique() if len(regions) else []:
        r = regions[regions["finding"] == finding]
        srr = r["self_rank"]
        summary["per_finding"][finding] = {
            "n_regions": int(len(r)),
            "n_tiles_median": _med(r["n_tiles"].tolist()),
            "self_rank_median": _med(srr.tolist()),
            "self_rank_is_1_rate": round(float((srr == 1).mean()), 3) if len(srr) else None,
            "self_rank_found_rate": round(float(srr.notna().mean()), 3) if len(srr) else None,
            "target_best_rank_median": _med(r["best_rank"].tolist()),
            **{f"target_hit@{k}_rate": _rate(r[f"hit@{k}"].tolist()) for k in HIT_KS},
        }
    return summary


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.search import PatchIndex

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()
    config = load_config(Path(__file__).parent)

    variant_key = "default"
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)

    seed = int(config.get("seed", 42))
    index_exp_dir = _staged_or(project_root, config["index_exp_dir"], "PVS_INDEX_DIR")
    features_dir = _staged_or(project_root, config["features_dir"], "PVS_FEATURES_DIR")
    # raw WSI (~634 GB) is not staged; Tier 1/2/3 open a few thousand read_region
    # calls across ~10-50 .svs over the whole run — bounded, same as 0014/0015.
    raw_slide_dir = _staged_or(project_root, config["raw_slide_dir"], "PVS_RAW_SLIDE_DIR")
    gt_csv = project_root / config["gt_csv"]
    encoder_name = config.get("encoder_name", "uni_v1")
    findings = list(config["findings"])
    tiers = list(config["tiers"])
    nprobe = int(config.get("nprobe", 64))

    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key,
                       findings=findings, tiers=tiers)

    logger.info(f"Starting: {exp_name} / {variant_key}")
    logger.info(f"findings: {findings}")
    logger.info(f"tiers:    {tiers}")

    pi = PatchIndex.load(
        index_path=index_exp_dir / "index.faiss",
        manifest_path=index_exp_dir / "manifest.parquet",
        slide_meta_path=index_exp_dir / "slide_meta.parquet",
        features_dir=features_dir,
    )
    corpus_slides = set(pi.slide_meta.index.astype(str))
    pi.manifest["slide_id"] = pi.manifest["slide_id"].astype(str)

    gt = pd.read_csv(gt_csv)
    gt["slide_id"] = gt["image_id"].str.replace(".svs", "", regex=False)

    results: dict = {
        "findings": findings,
        "encoder_name": encoder_name,
        "index_exp_dir": config["index_exp_dir"],
        "params": {"seed": seed, "nprobe": nprobe, **{k: config[k] for k in ("tier1", "tier2", "tier3")}},
    }

    if "roundtrip" in tiers:
        logger.info("=== Tier 1: roundtrip ===")
        results["tier1_roundtrip"] = tier_roundtrip(
            pi, config, findings, gt, corpus_slides, features_dir, raw_slide_dir,
            encoder_name, nprobe, run_dir, np.random.default_rng(seed), logger,
        )
    if "patchset" in tiers:
        logger.info("=== Tier 2: patchset ===")
        results["tier2_patchset"] = tier_patchset(
            pi, config, findings, gt, corpus_slides, features_dir, raw_slide_dir,
            encoder_name, nprobe, seed, run_dir / "patchset", logger,
        )
    if "region" in tiers:
        logger.info("=== Tier 3: region ===")
        results["tier3_region"] = tier_region(
            pi, config, findings, gt, corpus_slides, raw_slide_dir, encoder_name,
            nprobe, run_dir, np.random.default_rng(seed + 1), logger,
        )

    (run_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    complete_run(run_dir)
    logger.info(f"Done. {json.dumps(results, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
