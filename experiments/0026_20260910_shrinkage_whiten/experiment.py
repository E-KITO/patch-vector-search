"""experiments/0026: 収縮白色化(shrinkage whitening)の α スイープ。

experiments/0025 の full ZCA 白色化(α=1)が効きすぎて common 所見と atlas クエリを
悪化させたのを受け、固有値を平均へ収縮してから白色化する α ∈ {0.25,0.5,0.75} を
本番 OPQ+IVF+PQ 索引にベイクしてフル測定する。両端(baseline / full whiten)は
0025 の CSV を summary で参照。

出力(outputs/0026_.../):
  transforms/whiten_a{alpha}.npz
  a{tag}_v1/index.faiss + manifest.parquet + slide_meta.parquet
  self_retrieval_a{tag}.csv / atlas_gt_a{tag}.csv
  summary.md / results.json
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _get_project_root() -> Path:
    r = os.environ.get("PROJECT_ROOT")
    if not r:
        print("Error: PROJECT_ROOT is not set. Run via run_slurm.sh.", file=sys.stderr)
        sys.exit(1)
    return Path(r)


def load_config(exp_dir: Path) -> dict:
    with open(exp_dir / "config.yml") as f:
        return yaml.safe_load(f) or {}


def _tag(alpha: float) -> str:
    return f"{alpha:.2f}".replace(".", "")  # 0.25 -> "025", 0.50 -> "050"


def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("exp0026")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "experiment.log")):
        h.setFormatter(fmt)
        logger.addHandler(h)
    # lib.faiss_index も同じハンドラで拾う(0025 で進捗ログが出なかった対策)。
    fl = logging.getLogger("lib.faiss_index")
    fl.setLevel(logging.INFO)
    fl.handlers = logger.handlers
    return logger


def _staged_or(project_root: Path, config_rel: str, env_var: str) -> Path:
    staged = os.environ.get(env_var)
    if staged and Path(staged).exists():
        return Path(staged)
    return project_root / config_rel


# ============================================================
# Stage 1: fit
# ============================================================

def fit_transforms(project_root: Path, cfg: dict, run_dir: Path, logger: logging.Logger) -> dict:
    import faiss

    from lib.embedding_transform import fit_whiten_shrink, load_transform, save_transform

    tdir = run_dir / "transforms"
    tdir.mkdir(exist_ok=True)
    alphas = [float(a) for a in cfg["alphas"]]
    need = [a for a in alphas if not (tdir / f"whiten_a{_tag(a)}.npz").exists()]
    if need:
        sp = project_root / cfg["manifest_exp_dir"] / "training_sample.npy"
        logger.info("Stage 1: loading training sample %s", sp)
        X = np.load(sp).astype(np.float32)
        faiss.normalize_L2(X)
        for a in need:
            p = fit_whiten_shrink(X, alpha=a, eps=float(cfg["whiten_eps"]),
                                  fit_sample=int(cfg["fit_sample"]), seed=int(cfg["seed"]))
            save_transform(p, tdir / f"whiten_a{_tag(a)}.npz")
            logger.info("  fit alpha=%.2f", a)
        del X
    return {a: load_transform(tdir / f"whiten_a{_tag(a)}.npz") for a in alphas}


# ============================================================
# Stage 2: build indexes
# ============================================================

def build_indexes(project_root: Path, cfg: dict, transforms: dict, run_dir: Path,
                  logger: logging.Logger) -> None:
    from lib.embedding_transform import make_faiss_pretransform
    from lib.faiss_index import build_faiss_index

    manifest_dir = project_root / cfg["manifest_exp_dir"]
    features_dir = _staged_or(project_root, cfg["features_dir"], "PVS_FEATURES_DIR")

    for a in [float(x) for x in cfg["alphas"]]:
        vdir = run_dir / f"a{_tag(a)}_v1"
        vdir.mkdir(exist_ok=True)
        idx_path = vdir / "index.faiss"
        if idx_path.exists():
            logger.info("Stage 2: alpha=%.2f index exists — skipping", a)
            continue
        logger.info("Stage 2: building alpha=%.2f index -> %s", a, idx_path)
        lt, nt = make_faiss_pretransform(transforms[a])
        build_faiss_index(
            training_sample_path=manifest_dir / "training_sample.npy",
            features_dir=features_dir,
            manifest_path=manifest_dir / "manifest.parquet",
            index_path=idx_path,
            nlist=int(cfg["nlist"]), pq_m=int(cfg["pq_m"]),
            pq_nbits=int(cfg["pq_nbits"]), opq_niter=int(cfg["opq_niter"]),
            pretransform=[lt, nt],
        )
        shutil.copy2(manifest_dir / "manifest.parquet", vdir / "manifest.parquet")
        shutil.copy2(manifest_dir / "slide_meta.parquet", vdir / "slide_meta.parquet")
        logger.info("Stage 2: alpha=%.2f done", a)


# ============================================================
# Stage 3/4: diagnostics
# ============================================================

def run_diagnostics(project_root: Path, cfg: dict, run_dir: Path, logger: logging.Logger) -> None:
    py = sys.executable
    env = dict(os.environ)

    def sh(cmd, tag):
        logger.info("Stage: %s :: %s", tag, " ".join(cmd))
        r = subprocess.run(cmd, cwd=str(project_root), env=env, capture_output=True, text=True)
        (run_dir / f"log_{tag}.txt").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
        if r.returncode != 0:
            logger.error("%s FAILED rc=%d\n%s", tag, r.returncode, r.stderr[-2000:])
            raise RuntimeError(f"{tag} failed")
        logger.info("%s ok", tag)

    for a in [float(x) for x in cfg["alphas"]]:
        t = _tag(a)
        vdir = run_dir / f"a{t}_v1"
        sr = run_dir / f"self_retrieval_a{t}.csv"
        if not sr.exists():
            sh([py, "scripts/self_retrieval_diagnostic.py", "--index-dir", str(vdir),
                "--out", str(sr)], f"selfret_a{t}")
        gt = run_dir / f"atlas_gt_a{t}.csv"
        if not gt.exists():
            sh([py, "scripts/validate_against_ground_truth.py", "--index-dir", str(vdir),
                "--out", str(gt), "--nprobe", str(cfg["nprobe"])], f"atlasgt_a{t}")


# ============================================================
# Stage 5: summary(0025 の両端 CSV を取り込む)
# ============================================================

def write_summary(project_root: Path, cfg: dict, run_dir: Path, logger: logging.Logger) -> None:
    prior = project_root / cfg["prior_run_dir"]
    alphas = [float(x) for x in cfg["alphas"]]

    sr = {}
    p = prior / "self_retrieval_baseline.csv"
    if p.exists():
        sr["a0.00 (baseline)"] = pd.read_csv(p)
    for a in alphas:
        f = run_dir / f"self_retrieval_a{_tag(a)}.csv"
        if f.exists():
            sr[f"a{a:.2f}"] = pd.read_csv(f)
    p = prior / "self_retrieval_whiten.csv"
    if p.exists():
        sr["a1.00 (full)"] = pd.read_csv(p)

    gt = {}
    p = prior / "atlas_gt_baseline.csv"
    if p.exists():
        gt["a0.00"] = pd.read_csv(p)
    for a in alphas:
        f = run_dir / f"atlas_gt_a{_tag(a)}.csv"
        if f.exists():
            gt[f"a{a:.2f}"] = pd.read_csv(f)
    p = prior / "atlas_gt_whiten.csv"
    if p.exists():
        gt["a1.00"] = pd.read_csv(p)

    L = ["# experiments/0026: 収縮白色化の α スイープ", ""]
    L.append("固有値を平均へ収縮してから白色化(α=0→baseline 相当、α=1→0025 の full whiten)。"
             "両端は 0025 の CSV を参照。索引ハイパラは 0018/0025 と同一。")
    L.append("")

    if sr:
        base = next(iter(sr.values()))
        key = "finding"
        for metric in ("nhr_best_rank_med_noexp", "nhr_best_rank_med", "sim_best_rank_med"):
            if metric not in base.columns:
                continue
            L.append(f"## self_retrieval :: {metric}(低いほど良い)")
            L.append("")
            L.append("| finding | " + " | ".join(sr.keys()) + " |")
            L.append("|" + "---|" * (len(sr) + 1))
            for f in list(base[key]):
                row = [str(f)[:30]]
                for df in sr.values():
                    m = df.loc[df[key] == f, metric]
                    row.append(str(m.iloc[0]) if len(m) else "—")
                L.append("| " + " | ".join(row) + " |")
            L.append("")

    if gt:
        anydf = next(iter(gt.values()))
        cats = list(anydf["category"])
        L.append("## atlas GT best_rank(7 カテゴリ)")
        L.append("")
        L.append("| category | " + " | ".join(gt.keys()) + " |")
        L.append("|" + "---|" * (len(gt) + 1))
        for c in cats:
            row = [str(c)[:24]]
            for df in gt.values():
                bc = next((cc for cc in df.columns if cc.endswith("_best")), None)
                m = df.loc[df["category"] == c, bc] if bc else []
                row.append(str(m.iloc[0]) if len(m) else "—")
            L.append("| " + " | ".join(row) + " |")
        L.append("")

    L.append("## 読み方")
    L.append("")
    L.append("- **α を上げるほど白色化が強い**。0025 で α=1 は self_retrieval の難所見を "
             "改善したが Change eos / Cellular infiltration と atlas を悪化させた。")
    L.append("- 探すのは「**難所見の改善を保ちつつ common 所見と atlas の悪化が消える α**」。")
    L.append("  無ければ 収縮白色化は筋が悪く、成果物トラック専用索引(atlas クエリ無し)に "
             "full whiten を使う案へ。")
    (run_dir / "summary.md").write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    cfg = load_config(Path(__file__).parent)
    exp_name = os.environ["EXP_NAME"]

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata

    run_dir = get_run_dir(project_root, __file__, "default", output_root=os.environ.get("OUTPUT_ROOT"))
    write_run_metadata(run_dir, exp_name=exp_name, alphas=str(cfg["alphas"]))
    logger = setup_logger(run_dir)
    logger.info("run_dir: %s", run_dir)

    transforms = fit_transforms(project_root, cfg, run_dir, logger)
    build_indexes(project_root, cfg, transforms, run_dir, logger)
    run_diagnostics(project_root, cfg, run_dir, logger)
    write_summary(project_root, cfg, run_dir, logger)

    (run_dir / "results.json").write_text(json.dumps(
        {"alphas": cfg["alphas"],
         "indexes": {_tag(float(a)): str(run_dir / f"a{_tag(float(a))}_v1") for a in cfg["alphas"]}},
        indent=2, ensure_ascii=False))
    complete_run(run_dir)
    logger.info("done -> %s", run_dir)


if __name__ == "__main__":
    main()
