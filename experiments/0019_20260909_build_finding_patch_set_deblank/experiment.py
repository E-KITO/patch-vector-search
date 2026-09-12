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
# represents well enough to seed from, AND that are whole-patch texture — the
# one class of the three in README "所見の3クラス分け" that patch-level
# retrieval can serve. Keep this list conservative: a finding earns its place
# by a random-control check (scripts/random_patch_baseline.py), not by looking
# like it should work.
#
# Dropped after being disproven:
#   "Increased mitosis" — sub-patch signal; a mitotic figure does not change
#     how a 224px patch looks (job 9675).
#   "Hypertrophy" — a *relative* judgement ("larger than normal") with no
#     reference tissue inside a 224px crop. Its GT recall looked passable
#     (4/19, 8.7x chance) but a blind random control could not tell its patch
#     set from uniformly sampled liver (job 9937: 59% vs a 61% baseline, while
#     glycogen scored 91% under the same judge). Not reachable by deeper search.
SUPPORTED_FINDINGS = (
    "Deposit, glycogen",
    "Ground glass appearance",
    "Degeneration, granular, eosinophilic",
)

# validate: split the finding's GT slides seed/hold-out, exclude only the seed
#   slides, and mark hold-out GT slides so the gt_positive fraction reads as
#   recall — for a finding whose seed->retrieval leg is not yet proven.
# deliver: seed from all the finding's GT slides, exclude all of them, and
#   output the patches discovered in *unlabelled* slides — the actual
#   representative-patch set, for a finding already validated.
MODES = ("validate", "deliver")


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
        "query from that finding's confirmed TG-GATEs slides (not NNL atlas figures)."
    )
    parser.add_argument("--config", type=str, default="config.yml")
    parser.add_argument(
        "--task", required=True,
        help="'<FINDING_TYPE>@<mode>', e.g. 'Deposit,_glycogen@deliver' or "
        f"'Hypertrophy@validate'. Spaces in the finding may be written as '_'. "
        f"Findings: {sorted(SUPPORTED_FINDINGS)}. Modes: {list(MODES)}.",
    )
    args = parser.parse_args()
    finding, _, mode = args.task.partition("@")
    args.finding = finding.replace("_", " ")
    args.mode = mode
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

    from lib.finding_routing import load_index_for_finding
    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.patch_set import build_patch_set

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()
    config = load_config(Path(__file__).parent)

    if args.finding not in SUPPORTED_FINDINGS:
        raise SystemExit(f"finding {args.finding!r} not in {sorted(SUPPORTED_FINDINGS)}")
    if args.mode not in MODES:
        raise SystemExit(f"mode {args.mode!r} not in {list(MODES)} (task='<finding>@<mode>')")

    finding_slug = slugify(args.finding)
    variant_key = f"{finding_slug}__{args.mode}"
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)

    seed = int(config.get("seed", 42))
    features_dir = project_root / config["features_dir"]
    raw_slide_dir = project_root / config["raw_slide_dir"]
    gt_csv = project_root / config["gt_csv"]
    seed_fraction = float(config.get("seed_fraction", 0.5))
    max_seed_slides = int(config.get("max_seed_slides", 6))
    n_query_patches_per_seed_slide = int(config.get("n_query_patches_per_seed_slide", 120))
    nprobe = int(config.get("nprobe", 64))
    k_candidate_patches = int(config.get("k_candidate_patches", 5000))
    rerank_pool = int(config.get("rerank_pool", 5000))
    max_tiles_reranked = config.get("max_tiles_reranked", None)
    sim_floor = float(config.get("sim_floor", 0.65))
    nms_radius_patches = float(config.get("nms_radius_patches", 1.5))
    max_per_slide = int(config.get("max_per_slide", 15))
    target_n_patches = int(config.get("target_n_patches", 150))

    write_run_metadata(
        run_dir, exp_name=exp_name, variant_key=variant_key,
        finding=args.finding, mode=args.mode,
    )

    rng = np.random.default_rng(seed)

    # lib.finding_routing: 所見ラベルごとに baseline/whiten/macenko 索引を出し
    # 分ける(README「所見ごとのルーティング表」参照)。現状の SUPPORTED_FINDINGS
    # はいずれも whiten/macenko 対象外なので、このリストが変わらない限り
    # baseline と同じ結果になる。macenko ルーティング対象の所見をこのトラックに
    # 追加する場合、クエリも corpus 自身のスライドから直接 h5 を読む方式なので
    # features_dir を MACENKO_FEATURES_DIR に切り替える対応が別途必要になる
    # (lib.finding_routing.load_index_for_finding のdocstring参照、現状未対応)。
    pi = load_index_for_finding(args.finding)
    corpus_slides = set(pi.slide_meta.index.astype(str))

    gt = pd.read_csv(gt_csv)
    gt["slide_id"] = gt["image_id"].str.replace(".svs", "", regex=False)
    finding_slides = sorted(
        set(gt.loc[gt["FINDING_TYPE"] == args.finding, "slide_id"]) & corpus_slides
    )
    if len(finding_slides) < 2:
        raise SystemExit(
            f"{args.finding!r} has only {len(finding_slides)} corpus slide(s)"
        )

    # Seed / hold-out split depends on mode.
    perm = rng.permutation(len(finding_slides))
    if args.mode == "deliver":
        # Seed from (a sample of) all the finding's GT slides; exclude every GT
        # slide so the output is patches discovered in unlabelled slides.
        seed_pool = [finding_slides[i] for i in perm]
        holdout_slides: list[str] = []
        exclude_slides = set(finding_slides)
        gt_positive_slides: set[str] = set()
    else:  # validate
        n_seed = max(1, min(len(finding_slides) - 1, round(seed_fraction * len(finding_slides))))
        seed_pool = [finding_slides[i] for i in perm[:n_seed]]

    seed_slides = sorted(seed_pool[:max_seed_slides])
    if args.mode == "validate":
        # Hold-out is *every* GT slide not actually seeded, so it stays the
        # complement of seed_slides no matter which of seed_fraction /
        # max_seed_slides binds. Deriving it from seed_pool instead left the
        # slides that max_seed_slides trimmed off in neither set: not queried,
        # not excluded, and not flagged gt_positive, so patches retrieved from
        # them scored as non-GT and recall read low (job 9701 Hypertrophy
        # retrieved 4 GT slides but counted 1).
        holdout_slides = sorted(set(finding_slides) - set(seed_slides))
        exclude_slides = set(seed_slides)
        gt_positive_slides = set(holdout_slides)

    logger.info(f"Starting: {exp_name} / {variant_key}  (mode={args.mode})")
    logger.info(f"finding:        {args.finding}")
    logger.info(f"corpus GT slides ({len(finding_slides)}): {finding_slides}")
    logger.info(f"seed slides ({len(seed_slides)}, capped at {max_seed_slides}): {seed_slides}")
    logger.info(f"hold-out slides ({len(holdout_slides)}): {holdout_slides}")
    logger.info(f"excluding {len(exclude_slides)} slide(s) from results")

    query_vecs = load_slide_query_vecs(
        features_dir, seed_slides, n_query_patches_per_seed_slide, rng
    )
    logger.info(f"query vectors from seed slides: {query_vecs.shape[0]}")

    out_dir = run_dir / variant_key
    stats = build_patch_set(
        pi, query_vecs,
        out_dir=out_dir, raw_slide_dir=raw_slide_dir,
        exclude_slides=exclude_slides,
        gt_positive_slides=gt_positive_slides,
        nprobe=nprobe, k_candidate_patches=k_candidate_patches,
        rerank_pool=rerank_pool, max_tiles_reranked=max_tiles_reranked,
        sim_floor=sim_floor, nms_radius_patches=nms_radius_patches,
        max_per_slide=max_per_slide, target_n_patches=target_n_patches,
    )

    results = {
        "finding": args.finding,
        "mode": args.mode,
        "corpus_gt_slides": finding_slides,
        "seed_slides": seed_slides,
        "holdout_slides": holdout_slides,
        "n_query_vectors": int(query_vecs.shape[0]),
        **stats,
        "params": {
            "seed": seed, "seed_fraction": seed_fraction, "max_seed_slides": max_seed_slides,
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
