"""experiments/0025: 異方性除去変換を FAISS 索引にベイクして再構築 + フル測定。

experiments/0024 で有望だった ZCA 白色化 / ABTT を、コーパス部分サンプルの厳密行列
ではなく本番の OPQ+IVF+PQ 索引に組み込んでフル 18M パッチで再構築し、
self_retrieval_diagnostic(フル FAISS)と atlas GT で baseline(0018)と A/B する。
設計は config.yml のヘッダ参照。

出力(outputs/0025_.../):
  transforms/{whiten,abtt4}.npz          fit した変換パラメータ
  {whiten,abtt4}_v1/index.faiss + manifest.parquet + slide_meta.parquet
  self_retrieval_{whiten,abtt4}.csv       Stage 3
  atlas_gt_{baseline,whiten,abtt4}.csv    Stage 4
  summary.md / results.json               Stage 5
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default="config.yml")
    return p.parse_args()


def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("exp0025")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "experiment.log")):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def _staged_or(project_root: Path, config_rel: str, env_var: str) -> Path:
    staged = os.environ.get(env_var)
    if staged and Path(staged).exists():
        return Path(staged)
    return project_root / config_rel


# ============================================================
# Stage 1: 変換を fit
# ============================================================

def fit_transforms(project_root: Path, cfg: dict, run_dir: Path, logger: logging.Logger) -> dict:
    import faiss

    from lib.embedding_transform import fit_abtt, fit_whiten, load_transform, save_transform

    tdir = run_dir / "transforms"
    tdir.mkdir(exist_ok=True)
    out = {}
    need = [v for v in cfg["variants"] if not (tdir / f"{v}.npz").exists()]
    if need:
        sample_path = project_root / cfg["manifest_exp_dir"] / "training_sample.npy"
        logger.info("Stage 1: loading training sample %s", sample_path)
        X = np.load(sample_path).astype(np.float32)
        faiss.normalize_L2(X)
        logger.info("  training sample: %s", X.shape)
        for v in need:
            if v == "whiten":
                p = fit_whiten(X, eps=float(cfg["whiten_eps"]),
                               fit_sample=int(cfg["fit_sample"]), seed=int(cfg["seed"]))
            elif v == "abtt4":
                p = fit_abtt(X, d=int(cfg["abtt_d"]),
                             fit_sample=int(cfg["fit_sample"]), seed=int(cfg["seed"]))
            else:
                raise ValueError(f"unknown variant: {v}")
            save_transform(p, tdir / f"{v}.npz")
            logger.info("  fit %s: explained_var_ratio[:8]=%s", v,
                        np.round(p["explained_var_ratio"][:8], 4).tolist())
        del X
    for v in cfg["variants"]:
        out[v] = load_transform(tdir / f"{v}.npz")
    return out


# ============================================================
# Stage 2: 索引を再構築
# ============================================================

def build_indexes(project_root: Path, cfg: dict, transforms: dict, run_dir: Path,
                  logger: logging.Logger) -> None:
    from lib.embedding_transform import make_faiss_pretransform
    from lib.faiss_index import build_faiss_index

    manifest_dir = project_root / cfg["manifest_exp_dir"]
    features_dir = _staged_or(project_root, cfg["features_dir"], "PVS_FEATURES_DIR")

    for v in cfg["variants"]:
        vdir = run_dir / f"{v}_v1"
        vdir.mkdir(exist_ok=True)
        idx_path = vdir / "index.faiss"
        if idx_path.exists():
            logger.info("Stage 2: %s index exists — skipping", v)
            continue
        logger.info("Stage 2: building %s index -> %s", v, idx_path)
        lt, nt = make_faiss_pretransform(transforms[v])
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
        logger.info("Stage 2: %s done", v)


# ============================================================
# Stage 3/4: 診断スクリプトを --index-dir で回す
# ============================================================

def run_diagnostics(project_root: Path, cfg: dict, run_dir: Path, logger: logging.Logger) -> None:
    env = dict(os.environ)
    py = sys.executable

    def sh(cmd: list[str], tag: str):
        logger.info("Stage: %s :: %s", tag, " ".join(cmd))
        r = subprocess.run(cmd, cwd=str(project_root), env=env,
                           capture_output=True, text=True)
        (run_dir / f"log_{tag}.txt").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
        if r.returncode != 0:
            logger.error("%s FAILED (rc=%d), tail:\n%s", tag, r.returncode, r.stderr[-2000:])
            raise RuntimeError(f"{tag} failed")
        logger.info("%s ok", tag)

    targets = [("baseline", project_root / cfg["baseline_index_dir"])]
    targets += [(v, run_dir / f"{v}_v1") for v in cfg["variants"]]

    for name, idir in targets:
        sr_out = run_dir / f"self_retrieval_{name}.csv"
        if not sr_out.exists():
            sh([py, "scripts/self_retrieval_diagnostic.py",
                "--index-dir", str(idir), "--out", str(sr_out)], f"selfret_{name}")
        gt_out = run_dir / f"atlas_gt_{name}.csv"
        if not gt_out.exists():
            sh([py, "scripts/validate_against_ground_truth.py",
                "--index-dir", str(idir), "--out", str(gt_out),
                "--nprobe", str(cfg["nprobe"])], f"atlasgt_{name}")


# ============================================================
# Stage 5: summary
# ============================================================

def _load_selfret(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def write_summary(project_root: Path, cfg: dict, run_dir: Path, logger: logging.Logger) -> None:
    L = ["# experiments/0025: 異方性除去変換を索引にベイクして再構築 + フル測定", ""]
    L.append(f"variants = {cfg['variants']}、baseline = {cfg['baseline_index_dir']}。")
    L.append("変換は FAISS 索引に prepend(LinearTransform + NormalizationTransform)、"
             "add/search 双方で自動適用。ハイパラ(nlist/pq_m/pq_nbits/opq_niter)は 0018 と同一。")
    L.append("")

    # --- self_retrieval(フル FAISS)---
    L.append("## self_retrieval_diagnostic(フル FAISS、nhr_best_rank_med / sim_best_rank_med)")
    L.append("")
    srs = {}
    for name in (["baseline"] + list(cfg["variants"])):
        srs[name] = _load_selfret(run_dir / f"self_retrieval_{name}.csv")
    base_sr = srs.get("baseline")
    key = "finding" if (base_sr is not None and "finding" in base_sr.columns) else (
        base_sr.columns[0] if base_sr is not None else "finding")
    cols = [c for c in ("nhr_best_rank_med", "nhr_best_rank_med_noexp", "sim_best_rank_med")
            if base_sr is not None and c in base_sr.columns]
    if base_sr is not None and cols:
        findings = list(base_sr[key])
        for metric in cols:
            L.append(f"### {metric}")
            L.append("")
            L.append("| finding | " + " | ".join(srs.keys()) + " |")
            L.append("|" + "---|" * (len(srs) + 1))
            for f in findings:
                row = [f[:32]]
                for name, df in srs.items():
                    if df is None or f not in set(df[key]):
                        row.append("—")
                    else:
                        row.append(str(df.loc[df[key] == f, metric].iloc[0]))
                L.append("| " + " | ".join(row) + " |")
            L.append("")
    else:
        L.append("(baseline self_retrieval CSV が見つからない、または列名不一致 — "
                 "self_retrieval_{whiten,abtt4}.csv を直接参照)")
        L.append("")

    # --- atlas GT ---
    L.append("## atlas GT best_rank(7 カテゴリ、n_hits_ratio ランキング)")
    L.append("")
    gts = {}
    for name in (["baseline"] + list(cfg["variants"])):
        p = run_dir / f"atlas_gt_{name}.csv"
        if p.exists():
            gts[name] = pd.read_csv(p)
    if gts:
        any_df = next(iter(gts.values()))
        # validate script は列 <pipeline>_best を出す。--index-dir 経由なら pipeline名 = "index"。
        catcol = "category"
        cats = list(any_df[catcol])
        L.append("| category | " + " | ".join(gts.keys()) + " |")
        L.append("|" + "---|" * (len(gts) + 1))
        for c in cats:
            row = [str(c)[:26]]
            for name, df in gts.items():
                bestcol = next((cc for cc in df.columns if cc.endswith("_best")), None)
                if bestcol is None or c not in set(df[catcol]):
                    row.append("—")
                else:
                    row.append(str(df.loc[df[catcol] == c, bestcol].iloc[0]))
            L.append("| " + " | ".join(row) + " |")
        L.append("")

    L.append("## 読み方")
    L.append("")
    L.append("- **self_retrieval の nhr_best_rank_med** が baseline を下回れば、0024 の "
             "改善がフル FAISS でも再現(部分サンプル・proxy でなく本番指標)。")
    L.append("- **atlas GT** は de-emphasize 済み・±5 位ノイズ。ここでの悪化は "
             "「変換が外れクエリを壊す」サインだが、self_retrieval が主。")
    L.append("- 勝てば `lib.faiss_index` の pretransform を既定索引の再構築に使い、"
             "`lib.search._exact_similarity`(パッチ再ランク)側にも同じ変換を通す(現状は "
             "生 h5 = 未変換のまま。0025 の測定は slide ランキングのみなので影響なし)。")
    (run_dir / "summary.md").write_text("\n".join(L), encoding="utf-8")


# ============================================================
# main
# ============================================================

def main() -> None:
    parse_args()
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    exp_dir = Path(__file__).parent
    cfg = load_config(exp_dir)
    exp_name = os.environ["EXP_NAME"]

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata

    run_dir = get_run_dir(project_root, __file__, "default", output_root=os.environ.get("OUTPUT_ROOT"))
    write_run_metadata(run_dir, exp_name=exp_name, variants=",".join(cfg["variants"]))
    logger = setup_logger(run_dir)
    logger.info("run_dir: %s", run_dir)

    transforms = fit_transforms(project_root, cfg, run_dir, logger)
    build_indexes(project_root, cfg, transforms, run_dir, logger)
    run_diagnostics(project_root, cfg, run_dir, logger)
    write_summary(project_root, cfg, run_dir, logger)

    (run_dir / "results.json").write_text(json.dumps({
        "variants": cfg["variants"],
        "indexes": {v: str(run_dir / f"{v}_v1") for v in cfg["variants"]},
    }, indent=2, ensure_ascii=False))
    complete_run(run_dir)
    logger.info("done -> %s", run_dir)


if __name__ == "__main__":
    main()
