import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import yaml

import h5py
import numpy as np
import pandas as pd


# Findings the 2026-09-04 self-retrieval diagnostic showed the corpus
# represents well enough to seed from. Keep this list conservative — the point
# of 0015 is to test the curation pipeline on a seed that is known to work, not
# to re-litigate which findings retrieve.
SUPPORTED_FINDINGS = ("Deposit, glycogen", "Increased mitosis")


def _get_project_root() -> Path:
    project_root = os.environ.get("PROJECT_ROOT")
    if not project_root:
        print("Error: PROJECT_ROOT is not set. Run via run_slurm.sh.", file=sys.stderr)
        sys.exit(1)
    return Path(project_root)


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
    parser = argparse.ArgumentParser(
        description="Build a representative-patch set for one finding, seeding the "
        "query from that finding's confirmed TG-GATEs slides (not NNL atlas figures). "
        "GT slides are split seed/hold-out; the seed slides' patch vectors are the "
        "query, seed slides are excluded from results, and the gt_positive flag "
        "marks hold-out GT slides so it reads as recall."
    )
    parser.add_argument("--config", type=str, default="config.yml")
    parser.add_argument(
        "--finding", required=True,
        help="FINDING_TYPE (spaces may be written as '_'). One of: "
        f"{sorted(SUPPORTED_FINDINGS)}",
    )
    args = parser.parse_args()
    args.finding = args.finding.replace("_", " ")
    return args


def slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()


def load_slide_query_vecs(
    features_dir: Path, slide_ids: list[str], n_per_slide: int, rng: np.random.Generator
) -> np.ndarray:
    """Concatenated random sample of the seed slides' own patch feature vectors,
    L2-normalized. Read straight from the corpus h5 — no encoder."""
    parts = []
    for sid in slide_ids:
        with h5py.File(features_dir / f"{sid}.h5", "r") as f:
            n = f["features"].shape[0]
            if n <= n_per_slide:
                v = f["features"][:].astype(np.float32)
            else:
                rows = np.sort(rng.choice(n, size=n_per_slide, replace=False))
                v = f["features"][rows].astype(np.float32)
        parts.append(v)
    vecs = np.concatenate(parts, axis=0)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.patch_set import build_patch_set
    from lib.search import PatchIndex

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()
    config = load_config(Path(__file__).parent)

    if args.finding not in SUPPORTED_FINDINGS:
        raise SystemExit(f"--finding {args.finding!r} not in {sorted(SUPPORTED_FINDINGS)}")

    finding_slug = slugify(args.finding)
    run_dir = get_run_dir(project_root, __file__, finding_slug, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)

    seed = int(config.get("seed", 42))
    index_exp_dir = project_root / config["index_exp_dir"]
    features_dir = project_root / config["features_dir"]
    raw_slide_dir = project_root / config["raw_slide_dir"]
    gt_csv = project_root / config["gt_csv"]
    seed_fraction = float(config.get("seed_fraction", 0.5))
    n_query_patches_per_seed_slide = int(config.get("n_query_patches_per_seed_slide", 400))
    nprobe = int(config.get("nprobe", 64))
    k_candidate_patches = int(config.get("k_candidate_patches", 5000))
    rerank_pool = int(config.get("rerank_pool", 1000))
    max_tiles_reranked = config.get("max_tiles_reranked", None)
    sim_floor = float(config.get("sim_floor", 0.40))
    nms_radius_patches = float(config.get("nms_radius_patches", 1.5))
    max_per_slide = int(config.get("max_per_slide", 15))
    target_n_patches = int(config.get("target_n_patches", 150))

    write_run_metadata(run_dir, exp_name=exp_name, variant_key=finding_slug, finding=args.finding)

    rng = np.random.default_rng(seed)

    pi = PatchIndex.load(
        index_path=index_exp_dir / "index.faiss",
        manifest_path=index_exp_dir / "manifest.parquet",
        slide_meta_path=index_exp_dir / "slide_meta.parquet",
        features_dir=features_dir,
    )
    corpus_slides = set(pi.slide_meta.index.astype(str))

    gt = pd.read_csv(gt_csv)
    gt["slide_id"] = gt["image_id"].str.replace(".svs", "", regex=False)
    finding_slides = sorted(
        set(gt.loc[gt["FINDING_TYPE"] == args.finding, "slide_id"]) & corpus_slides
    )
    if len(finding_slides) < 2:
        raise SystemExit(
            f"{args.finding!r} has only {len(finding_slides)} corpus slide(s) — cannot split seed/hold-out"
        )

    perm = rng.permutation(len(finding_slides))
    n_seed = max(1, min(len(finding_slides) - 1, round(seed_fraction * len(finding_slides))))
    seed_slides = sorted(finding_slides[i] for i in perm[:n_seed])
    holdout_slides = sorted(finding_slides[i] for i in perm[n_seed:])

    logger.info(f"Starting: {exp_name} / {finding_slug}")
    logger.info(f"finding:        {args.finding}")
    logger.info(f"corpus GT slides ({len(finding_slides)}): {finding_slides}")
    logger.info(f"seed slides ({len(seed_slides)}):    {seed_slides}")
    logger.info(f"hold-out slides ({len(holdout_slides)}): {holdout_slides}")

    query_vecs = load_slide_query_vecs(
        features_dir, seed_slides, n_query_patches_per_seed_slide, rng
    )
    logger.info(f"query vectors from seed slides: {query_vecs.shape[0]}")

    out_dir = run_dir / finding_slug
    stats = build_patch_set(
        pi, query_vecs,
        out_dir=out_dir, raw_slide_dir=raw_slide_dir,
        exclude_slides=set(seed_slides),
        gt_positive_slides=set(holdout_slides),
        nprobe=nprobe, k_candidate_patches=k_candidate_patches,
        rerank_pool=rerank_pool, max_tiles_reranked=max_tiles_reranked,
        sim_floor=sim_floor, nms_radius_patches=nms_radius_patches,
        max_per_slide=max_per_slide, target_n_patches=target_n_patches,
    )

    results = {
        "finding": args.finding,
        "corpus_gt_slides": finding_slides,
        "seed_slides": seed_slides,
        "holdout_slides": holdout_slides,
        "n_query_vectors": int(query_vecs.shape[0]),
        **stats,
        "params": {
            "seed": seed, "seed_fraction": seed_fraction,
            "n_query_patches_per_seed_slide": n_query_patches_per_seed_slide,
            "nprobe": nprobe, "k_candidate_patches": k_candidate_patches,
            "rerank_pool": rerank_pool, "max_tiles_reranked": max_tiles_reranked,
            "sim_floor": sim_floor, "nms_radius_patches": nms_radius_patches,
            "max_per_slide": max_per_slide, "target_n_patches": target_n_patches,
        },
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

    complete_run(run_dir)
    logger.info(f"Done. {json.dumps(results, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
