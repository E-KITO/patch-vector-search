import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import yaml


def _get_project_root() -> Path:
    project_root = os.environ.get("PROJECT_ROOT")
    if not project_root:
        print("Error: PROJECT_ROOT is not set. Run via run_slurm.sh.", file=sys.stderr)
        sys.exit(1)
    return Path(project_root)


def setup_logger(run_dir: Path, name: str = "experiment") -> logging.Logger:
    """Set up a logger writing to both console and run_dir/experiment.log."""
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
    """Load config.yml from the experiment directory."""
    config_path = exp_dir / "config.yml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yml")
    return parser.parse_args()


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.faiss_index import build_faiss_index
    from lib.output_utils import complete_run, get_run_dir, write_run_metadata

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")

    parse_args()

    variant_key = "default"
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)

    config = load_config(Path(__file__).parent)
    seed: int = config.get("seed", 42)
    features_dir = project_root / config["features_dir"]
    manifest_exp_dir = project_root / config["manifest_exp_dir"]
    nlist: int = config.get("nlist", 4096)
    pq_m: int = config.get("pq_m", 64)
    pq_nbits: int = config.get("pq_nbits", 8)
    opq_niter: int = config.get("opq_niter", 10)

    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key)

    logger.info(f"Starting: {exp_name} / {variant_key}")
    logger.info(f"run_dir:          {run_dir}")
    logger.info(f"features_dir:     {features_dir}")
    logger.info(f"manifest_exp_dir: {manifest_exp_dir}")
    logger.info(f"nlist={nlist} pq_m={pq_m} pq_nbits={pq_nbits} seed={seed}")

    manifest_path = manifest_exp_dir / "manifest.parquet"
    slide_meta_path = manifest_exp_dir / "slide_meta.parquet"
    training_sample_path = manifest_exp_dir / "training_sample.npy"
    for p in (manifest_path, slide_meta_path, training_sample_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found — run experiments/0001_..._build_patch_manifest first"
            )

    index_path = run_dir / "index.faiss"

    # ── Experiment logic ──────────────────────────────────────────────────────
    index = build_faiss_index(
        training_sample_path=training_sample_path,
        features_dir=features_dir,
        manifest_path=manifest_path,
        index_path=index_path,
        nlist=nlist,
        pq_m=pq_m,
        pq_nbits=pq_nbits,
        opq_niter=opq_niter,
    )

    # Copy manifest/slide_meta alongside the index so downstream consumers
    # (experiments/0003_..._query_demo, notebooks/) only need to point at
    # this one run_dir instead of also reaching back into 0001's outputs.
    shutil.copy2(manifest_path, run_dir / "manifest.parquet")
    shutil.copy2(slide_meta_path, run_dir / "slide_meta.parquet")

    results = {
        "index_path": str(index_path),
        "manifest_path": str(run_dir / "manifest.parquet"),
        "slide_meta_path": str(run_dir / "slide_meta.parquet"),
        "n_vectors": int(index.ntotal),
        "nlist": nlist,
        "pq_m": pq_m,
        "pq_nbits": pq_nbits,
        "opq_niter": opq_niter,
    }

    # ── Save results ──────────────────────────────────────────────────────────
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    complete_run(run_dir)
    logger.info(f"Done. {results}")


if __name__ == "__main__":
    main()
