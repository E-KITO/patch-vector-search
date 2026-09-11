"""experiments/0024: 類似度指標の再ランク比較(self-retrieval、FAISS なし)。

現行の「L2 正規化コサイン」に対し、異方性除去(center / all-but-the-top / 白色化)と
hubness 補正(CSLS)が、コーパス内 leave-one-out の検索順位を改善するかを A/B する。
設計と数学的背景は config.yml のヘッダ参照。

出力: outputs/0024_.../rerank/{corpus_sub.npz, results.csv, summary.md, curve.json}
"""

import argparse
import json
import logging
import os
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
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("exp0024")
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


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


# ============================================================
# コーパス部分サンプル
# ============================================================

def _local_idx_by_slide(index_dir: Path) -> dict[str, np.ndarray]:
    """manifest.parquet を一度だけ読み、{slide_id: 背景除外後の local_idx 配列}。"""
    m = pd.read_parquet(index_dir / "manifest.parquet", columns=["slide_id", "local_idx"])
    m["slide_id"] = m["slide_id"].astype(str)
    return {sid: g["local_idx"].to_numpy() for sid, g in m.groupby("slide_id")}


def build_corpus_subsample(project_root: Path, index_dir: Path, features_dir: Path,
                           loc_by_slide: dict, cfg: dict, run_dir: Path, overwrite: bool,
                           logger: logging.Logger):
    import h5py

    cache = run_dir / "corpus_sub.npz"
    if cache.exists() and not overwrite:
        d = np.load(cache, allow_pickle=True)
        logger.info("corpus subsample loaded from cache: %s (%d vecs)", cache, len(d["X"]))
        return d["X"], d["slide_ids"].astype(str), d["sub_counts"].item()

    n_sub = int(cfg["n_corpus_subsample"])
    rng = np.random.default_rng(int(cfg["seed"]))

    Xs, sids = [], []
    sub_counts: dict[str, int] = {}
    slides = sorted(loc_by_slide.keys())
    for i, sid in enumerate(slides):
        loc = loc_by_slide[sid]
        take = np.sort(rng.choice(loc, size=min(n_sub, len(loc)), replace=False))
        with h5py.File(features_dir / f"{sid}.h5", "r") as f:
            v = f["features"][take].astype(np.float32)
        Xs.append(v)
        sids.append(np.full(len(v), sid))
        sub_counts[sid] = len(v)
        if (i + 1) % 100 == 0:
            logger.info("  corpus subsample: %d/%d slides", i + 1, len(slides))

    X = _l2(np.concatenate(Xs, axis=0))
    slide_ids = np.concatenate(sids)
    np.savez(cache, X=X, slide_ids=slide_ids, sub_counts=np.array(sub_counts, dtype=object))
    logger.info("corpus subsample built: %d vecs across %d slides -> %s", len(X), len(slides), cache)
    return X, slide_ids.astype(str), sub_counts


# ============================================================
# クエリ層化(self_retrieval_diagnostic と同じ)
# ============================================================

def load_findings(project_root: Path, cfg: dict, corpus_slides: set, rng: np.random.Generator,
                  logger: logging.Logger):
    gt = pd.read_csv(project_root / cfg["gt_csv"])
    gt["slide_id"] = gt["image_id"].str.replace(".svs", "", regex=False)
    gt = gt[gt["slide_id"].isin(corpus_slides)].copy()
    exp_of = dict(zip(gt["slide_id"], gt["EXP_ID"].astype(str)))
    f2s = gt.groupby("FINDING_TYPE")["slide_id"].agg(lambda s: sorted(set(s))).to_dict()
    mn = int(cfg["min_slides_per_finding"])
    mx = int(cfg["max_query_slides_per_finding"])
    findings = sorted((f for f, s in f2s.items() if len(s) >= mn), key=lambda f: -len(f2s[f]))
    plan = {}
    for f in findings:
        slides = f2s[f]
        q = slides if len(slides) <= mx else [slides[i] for i in sorted(rng.choice(len(slides), mx, replace=False))]
        plan[f] = {"targets": set(slides), "queries": q}
    logger.info("findings with >=%d corpus slides: %d", mn, len(findings))
    return plan, exp_of


def load_query_vecs(features_dir: Path, loc_by_slide: dict, sid: str,
                    n: int, rng: np.random.Generator) -> np.ndarray:
    import h5py

    loc = loc_by_slide[sid]
    take = np.sort(rng.choice(loc, size=min(n, len(loc)), replace=False))
    with h5py.File(features_dir / f"{sid}.h5", "r") as f:
        v = f["features"][take].astype(np.float32)
    return _l2(v)


# ============================================================
# 変換空間
# ============================================================

def fit_space(X: np.ndarray, cfg: dict, rng: np.random.Generator, logger: logging.Logger) -> dict:
    mu = X.mean(axis=0)
    m = int(cfg["pca_fit_sample"])
    idx = rng.choice(len(X), size=min(m, len(X)), replace=False)
    Xc = X[idx] - mu
    # SVD で主成分と特異値。Xc = U S Vt, 列 Vt.T が主方向。
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    eig = (s ** 2) / len(Xc)  # 共分散の固有値
    eps = float(cfg["whiten_eps"])
    W = (vt.T * (1.0 / np.sqrt(eig + eps))) @ vt  # ZCA: V (Λ+εI)^{-1/2} V^T
    logger.info("fit_space: top-8 explained-variance ratio = %s",
                np.round(eig[:8] / eig.sum(), 4).tolist())
    return {"mu": mu.astype(np.float32), "V": vt.astype(np.float32), "W": W.astype(np.float32)}


def transform(name: str, V: np.ndarray, sp: dict) -> np.ndarray:
    """name の変換空間へ写した単位ベクトルを返す。CSLS は base 空間のみここで扱う。"""
    if name in ("cosine", "csls"):
        return _l2(V)  # csls も base は素のコサイン空間(補正はスコアリングで)
    Vc = V - sp["mu"]
    if name == "center":
        return _l2(Vc)
    if name.startswith("abtt") or name == "csls_abtt2":
        d = 2 if name == "csls_abtt2" else int(name[4:])
        U = sp["V"][:d]                       # (d, 1024)
        proj = (Vc @ U.T) @ U                 # 上位 d 主成分への射影
        return _l2(Vc - proj)
    if name == "whiten":
        return _l2(Vc @ sp["W"].T)
    raise ValueError(f"unknown arm: {name!r}")


def _base_space(name: str) -> str:
    if name == "csls":
        return "cosine"
    if name == "csls_abtt2":
        return "abtt2"
    if name == "csls_whiten":
        return "whiten"
    return name


# ============================================================
# CSLS の r_T(c)(コーパス内 kNN 平均、チャンク matmul)
# ============================================================

def compute_rT(Xc_t: np.ndarray, k: int, logger: logging.Logger) -> np.ndarray:
    n = len(Xc_t)
    rT = np.empty(n, dtype=np.float32)
    step = 4000
    for a in range(0, n, step):
        b = min(a + step, n)
        blk = Xc_t[a:b] @ Xc_t.T          # (rows, n)
        for r in range(b - a):
            blk[r, a + r] = -np.inf       # 自分を除外
        part = np.partition(blk, -k, axis=1)[:, -k:]
        rT[a:b] = part.mean(axis=1)
        if (a // step) % 5 == 0:
            logger.info("  rT: %d/%d", b, n)
    return rT


# ============================================================
# スコアリング
# ============================================================

def _best_rank(ranked_slides: list, targets: set):
    pos = [i + 1 for i, s in enumerate(ranked_slides) if s in targets]
    return min(pos) if pos else None


def score_query(Xc_t: np.ndarray, slide_ids: np.ndarray, sub_counts: dict,
                q_t: np.ndarray, is_csls: bool, rT: np.ndarray | None,
                csls_k: int, hit_k: int, uniq_slides: np.ndarray):
    S = q_t @ Xc_t.T  # (nq, Nsub)
    if is_csls:
        rS = np.partition(S, -csls_k, axis=1)[:, -csls_k:].mean(axis=1)
        S = 2.0 * S - rT[None, :] - rS[:, None]

    # --- sim 集約: 各コーパスパッチの「全クエリベクトル最大」→ スライド最大 ---
    patch_best = S.max(axis=0)  # (Nsub,)
    sim_by_slide = pd.Series(patch_best).groupby(slide_ids).max()
    sim_ranked = sim_by_slide.sort_values(ascending=False).index.tolist()

    # --- nhr 集約: 各クエリベクトルの上位 hit_k パッチ → distinct hit をスライド別に数え、
    #     そのスライドの部分サンプル数で割る ---
    top = np.argpartition(S, -hit_k, axis=1)[:, -hit_k:]
    hit_idx = np.unique(top)
    hit_slides = slide_ids[hit_idx]
    cnt = pd.Series(hit_slides).value_counts()
    nhr = (cnt / cnt.index.map(lambda s: sub_counts[s])).sort_values(ascending=False)
    nhr_ranked = nhr.index.tolist()
    # nhr に出てこないスライドは末尾(順位 = 全スライド数扱い)
    return sim_ranked, nhr_ranked


# ============================================================
# main
# ============================================================

def main() -> None:
    args = parse_args()
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    exp_dir = Path(__file__).parent
    cfg = load_config(exp_dir)
    exp_name = os.environ["EXP_NAME"]

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata

    run_dir = get_run_dir(project_root, __file__, "rerank", output_root=os.environ.get("OUTPUT_ROOT"))
    write_run_metadata(run_dir, exp_name=exp_name, arms=",".join(cfg["arms"]))
    logger = setup_logger(run_dir)
    logger.info("run_dir: %s", run_dir)

    index_dir = _staged_or(project_root, cfg["index_dir"], "PVS_INDEX_DIR")
    features_dir = _staged_or(project_root, cfg["features_dir"], "PVS_FEATURES_DIR")
    logger.info("index_dir=%s  features_dir=%s", index_dir, features_dir)

    loc_by_slide = _local_idx_by_slide(index_dir)
    X, slide_ids, sub_counts = build_corpus_subsample(
        project_root, index_dir, features_dir, loc_by_slide, cfg, run_dir, args.overwrite, logger)
    n_corpus_slides = len(sub_counts)
    corpus_slides = set(sub_counts.keys())
    uniq_slides = np.array(sorted(corpus_slides))

    rng = np.random.default_rng(int(cfg["seed"]))
    plan, exp_of = load_findings(project_root, cfg, corpus_slides, rng, logger)

    sp = fit_space(X, cfg, np.random.default_rng(int(cfg["seed"]) + 7), logger)

    results_csv = run_dir / "results.csv"
    rows = []
    done_arms = set()
    if results_csv.exists() and not args.overwrite:
        prev = pd.read_csv(results_csv)
        rows = prev.to_dict("records")
        done_arms = set(prev["arm"].unique())
        logger.info("resuming: arms done = %s", sorted(done_arms))

    # クエリベクトルは全アーム共通(seed 固定で再現)。読み込んでキャッシュ。
    qcache = run_dir / "query_vecs.npz"
    if qcache.exists() and not args.overwrite:
        qz = np.load(qcache)
        qvecs = {k: qz[k] for k in qz.files}
        logger.info("query vecs loaded from cache (%d slides)", len(qvecs))
    else:
        qvecs = {}
        qrng = np.random.default_rng(int(cfg["seed"]) + 99)
        for f, d in plan.items():
            for sid in d["queries"]:
                if sid not in qvecs:
                    qvecs[sid] = load_query_vecs(features_dir, loc_by_slide, sid,
                                                 int(cfg["n_query_vecs"]), qrng)
        np.savez(qcache, **qvecs)
        logger.info("loaded + cached query vecs for %d slides", len(qvecs))

    base_cache: dict[str, np.ndarray] = {}
    for arm in cfg["arms"]:
        if arm in done_arms:
            continue
        logger.info("=== arm: %s ===", arm)
        base = _base_space(arm)
        if base not in base_cache:
            base_cache[base] = transform(base, X, sp)
        Xc_t = base_cache[base]
        is_csls = arm.startswith("csls")
        rT = compute_rT(Xc_t, int(cfg["csls_k"]), logger) if is_csls else None

        for f, d in plan.items():
            tgt_all = d["targets"]
            for sid in d["queries"]:
                q_t = transform(base, qvecs[sid], sp)
                sim_ranked, nhr_ranked = score_query(
                    Xc_t, slide_ids, sub_counts, q_t, is_csls, rT,
                    int(cfg["csls_k"]), int(cfg["hit_k"]), uniq_slides)
                sim_ranked = [s for s in sim_ranked if s != sid]
                nhr_ranked = [s for s in nhr_ranked if s != sid]
                t_all = tgt_all - {sid}
                t_noexp = {s for s in t_all if exp_of.get(s) != exp_of.get(sid)}
                rows.append({
                    "arm": arm, "finding": f, "query_slide": sid,
                    "n_targets_all": len(t_all), "n_targets_noexp": len(t_noexp),
                    "sim_best_all": _best_rank(sim_ranked, t_all),
                    "sim_best_noexp": _best_rank(sim_ranked, t_noexp),
                    "nhr_best_all": _best_rank(nhr_ranked, t_all),
                    "nhr_best_noexp": _best_rank(nhr_ranked, t_noexp),
                })
        pd.DataFrame(rows).to_csv(results_csv, index=False)
        logger.info("arm %s done", arm)

    df = pd.DataFrame(rows)
    df.to_csv(results_csv, index=False)
    summarize(df, cfg, n_corpus_slides, run_dir)
    complete_run(run_dir)
    logger.info("done -> %s", run_dir)


def summarize(df: pd.DataFrame, cfg: dict, n_corpus_slides: int, run_dir: Path) -> None:
    arms = list(cfg["arms"])
    cap = n_corpus_slides

    def med(sub, col):
        v = sub[col].to_numpy(dtype=float)
        v = np.where(np.isnan(v), cap, v)
        return float(np.median(v)) if len(v) else float("nan")

    per = []
    for arm in arms:
        a = df[df["arm"] == arm]
        for metric in ("sim_best_noexp", "nhr_best_noexp"):
            per.append({"arm": arm, "metric": metric,
                        "median_over_findings": float(np.median(
                            [med(a[a["finding"] == f], metric) for f in a["finding"].unique()]))
                        if len(a) else float("nan")})

    L = ["# experiments/0024: 類似度指標の再ランク比較", ""]
    L.append(f"self-retrieval(コーパス内 leave-one-out、FAISS なし、"
             f"スライドあたり {cfg['n_corpus_subsample']} パッチの厳密行列)。")
    L.append(f"順位は同一 FINDING_TYPE の別スライド(同一 EXP_ID 除外 = _noexp)の best_rank。"
             f"未 top-{cap} は {cap} として集計。所見横断の median。")
    L.append("")
    L.append("## 所見横断 median best_rank(低いほど良い)")
    L.append("")
    L.append("| arm | sim (max_similarity) | nhr (n_hits_ratio proxy) |")
    L.append("|---|---|---|")
    for arm in arms:
        s = next(x["median_over_findings"] for x in per if x["arm"] == arm and x["metric"] == "sim_best_noexp")
        n = next(x["median_over_findings"] for x in per if x["arm"] == arm and x["metric"] == "nhr_best_noexp")
        L.append(f"| {arm} | {s:.1f} | {n:.1f} |")
    L.append("")
    L.append("## 所見別 nhr best_rank(_noexp、median)")
    L.append("")
    findings = sorted(df["finding"].unique())
    L.append("| finding | " + " | ".join(arms) + " |")
    L.append("|" + "---|" * (len(arms) + 1))
    for f in findings:
        cells = []
        for arm in arms:
            sub = df[(df["arm"] == arm) & (df["finding"] == f)]
            cells.append(f"{med(sub, 'nhr_best_noexp'):.0f}" if len(sub) else "—")
        L.append(f"| {f[:34]} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## 読み方")
    L.append("")
    L.append("- **baseline = `cosine`**。他アームの所見横断 median がこれを下回れば、"
             "その変換が検索を改善している。")
    L.append("- **`csls` / `whiten` / `abtt*` が効くのは融合壊死・髄外造血・granular** "
             "(hub な正常組織にスコアを奪われていた所見)であるはず。common な所見は "
             "元々 hub の影響が小さいので横ばい〜微減が想定。")
    L.append("- **`csls` が sim より nhr で効く**なら、r_S(q) 項(クエリ密度補正)が "
             "タイル間比較に効いているサイン。")
    L.append("- 改善が確認できたら、勝ったアームを `lib.search` の再ランク段に実装し "
             "(コーパス変換は一度、クエリは都度)、atlas GT でも測る。")
    (run_dir / "summary.md").write_text("\n".join(L), encoding="utf-8")
    (run_dir / "curve.json").write_text(json.dumps({"per_arm_metric": per}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
