"""experiments/0027: atlas GT 悪化の図版単位切り分け診断。

experiments/0025 の whiten 索引が atlas GT(7カテゴリ)を悪化させた件について、
悪化が特定の外れ値図版に集中しているのか、カテゴリ内の図版全体に一様なのかを
scripts/atlas_per_image_diagnostic.py で切り分ける。索引の再構築はせず、既存の
baseline(0018)/ whiten(0025 full whiten)索引を図版1枚ずつ個別クエリで比較する。

出力(outputs/0027_.../default/):
  atlas_per_image.csv   finding, image, n_gt, {name}_found/{name}_best/{name}_mean
  summary.md            悪化幅でソートした一覧 + 集中度の要約
  results.json
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

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
    logger = logging.getLogger("exp0027")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "experiment.log")):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


# ============================================================
# Stage 1: 図版単位の診断を回す
# ============================================================

def run_diagnostic(project_root: Path, cfg: dict, run_dir: Path, logger: logging.Logger) -> Path:
    out = run_dir / "atlas_per_image.csv"
    if out.exists():
        logger.info("Stage 1: %s exists — skipping", out)
        return out

    index_dirs_spec = ",".join(f"{name}={path}" for name, path in cfg["index_dirs"].items())
    cmd = [
        sys.executable, "scripts/atlas_per_image_diagnostic.py",
        "--index-dirs", index_dirs_spec,
        "--out", str(out),
        "--nprobe", str(cfg["nprobe"]),
    ]
    logger.info("Stage 1: %s", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(project_root), env=dict(os.environ), capture_output=True, text=True)
    (run_dir / "log_diagnostic.txt").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    if r.returncode != 0:
        logger.error("Stage 1 FAILED (rc=%d), tail:\n%s", r.returncode, r.stderr[-2000:])
        raise RuntimeError("atlas_per_image_diagnostic failed")
    logger.info("Stage 1 ok")
    return out


# ============================================================
# Stage 2: summary — 悪化幅でソートし、外れ値集中 vs 一様を判定しやすくする
# ============================================================

def write_summary(cfg: dict, csv_path: Path, run_dir: Path, logger: logging.Logger) -> None:
    df = pd.read_csv(csv_path)
    names = list(cfg["index_dirs"].keys())
    base, other = names[0], names[1]

    df["delta_best"] = df[f"{other}_best"] - df[f"{base}_best"]
    df_sorted = df.sort_values("delta_best", ascending=False)

    L = ["# experiments/0027: atlas GT 悪化の図版単位切り分け診断", ""]
    L.append(f"baseline = `{cfg['index_dirs'][base]}`、比較対象 = `{other}` "
             f"(`{cfg['index_dirs'][other]}`)。図版1枚ずつ個別に "
             "search_top_slides_multi。delta_best = {other}_best - {base}_best "
             "(正で大きいほどその図版で悪化)。")
    L.append("")
    L.append("## 図版ごとの delta_best(悪化幅の降順)")
    L.append("")
    cols = ["finding", "image", "n_gt", f"{base}_best", f"{other}_best", "delta_best",
            f"{base}_found", f"{other}_found"]
    L.append("| " + " | ".join(cols) + " |")
    L.append("|" + "---|" * len(cols))
    for _, row in df_sorted.iterrows():
        L.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    L.append("")

    n = len(df_sorted)
    worsened = df_sorted[df_sorted["delta_best"] > 0]
    n_worse = len(worsened)
    if n_worse:
        top_k = max(1, round(n_worse * 0.2))
        share = worsened["delta_best"].sort_values(ascending=False).head(top_k).sum() / worsened["delta_best"].sum()
        L.append("## 集中度の目安")
        L.append("")
        L.append(f"- 悪化した図版: {n_worse}/{n} 枚")
        L.append(f"- 悪化幅の合計に対する上位{top_k}枚(悪化した図版の上位20%)の寄与率: "
                 f"{share:.0%}")
        L.append("- 目安: この寄与率が高ければ少数の外れ値図版が悪化を主導、"
                 "50%前後に近ければカテゴリ全体に広く効いている(ドメインギャップの疑いが強まる)。")
    else:
        L.append("## 集中度の目安")
        L.append("")
        L.append("悪化した図版なし。")
    L.append("")

    (run_dir / "summary.md").write_text("\n".join(L), encoding="utf-8")
    logger.info("wrote %s", run_dir / "summary.md")


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
    write_run_metadata(run_dir, exp_name=exp_name, index_dirs=",".join(cfg["index_dirs"]))
    logger = setup_logger(run_dir)
    logger.info("run_dir: %s", run_dir)

    csv_path = run_diagnostic(project_root, cfg, run_dir, logger)
    write_summary(cfg, csv_path, run_dir, logger)

    (run_dir / "results.json").write_text(json.dumps({
        "index_dirs": cfg["index_dirs"],
        "csv": str(csv_path),
    }, indent=2, ensure_ascii=False))
    complete_run(run_dir)
    logger.info("done -> %s", run_dir)


if __name__ == "__main__":
    main()
