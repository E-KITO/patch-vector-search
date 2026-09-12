"""所見ラベルに対する One-vs-Rest 線形分類器で、クエリタイルを「その所見らしいか」で
再スコアする(experiments/0035)。

背景(README「検索結果の可視化改善とタイル選択バイアスの発見」): lib.search.PatchIndex
の近似FAISSスコアは「コーパス内で最近傍が何か」を測るため、コーパスにありふれた正常
組織に似たタイルほど高スコアになりやすい——珍しい所見のタイルほど、近似スコアという
選抜基準自体で不利になる。この頻度バイアスを避けるため、所見の確定GTスライド
(data/processed_csv/single_finding_liver.csv)のパッチを正例、コーパス全体から
ランダムサンプルした「その他大半」を負例として、UNI特徴量(既存の埋め込みをそのまま
使う、再学習・再埋め込み不要)上に所見ごとの軽量な線形分類器(ロジスティック回帰)を
学習する。スコアは「コーパス内で目立つか」ではなく「その所見の正例らしいか」になる。

正例はスライド単位のラベルしか無く、パッチ単位のアノテーションではない
(README「評価に使ったデータとその限界」)。同じスライドの全パッチが正例として扱われる
弱ラベルであり、実際には所見と無関係な"平凡な"パッチも正例に混入する。
scripts/self_retrieval_diagnostic.py と同じ流儀で、study(EXP_ID)単位の
leave-one-study-out交差検証を用意し、分類器が「所見」ではなく「化合物・studyという
バッチ」を学習していないかを確認する。
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

GT_CSV = Path("data/processed_csv/single_finding_liver.csv")


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def load_gt_slides_for_finding(
    finding_type: str, corpus_slides: set[str], gt_csv: Path = GT_CSV
) -> pd.DataFrame:
    """finding_type のコーパス内スライド一覧(slide_id, EXP_ID, COMPOUND_NAME)を返す。

    1スライドが複数行(部位・グレード違い)を持つことがあるため slide_id で重複排除する
    (self_retrieval_diagnostic.py と同じ扱い)。
    """
    gt = pd.read_csv(gt_csv)
    gt["slide_id"] = gt["image_id"].str.replace(".svs", "", regex=False)
    gt = gt[gt["slide_id"].isin(corpus_slides)]
    gt = gt[gt["FINDING_TYPE"] == finding_type]
    return (
        gt[["slide_id", "EXP_ID", "COMPOUND_NAME"]]
        .drop_duplicates("slide_id")
        .reset_index(drop=True)
    )


def _read_patch_vectors(slide_id: str, local_idx: np.ndarray, features_dir: Path) -> np.ndarray:
    """h5py はソート済み添字しか受け付けないので、ソートして読み、元の順序に戻す
    (lib.search.PatchIndex._exact_similarity と同じパターン)。"""
    sort_order = np.argsort(local_idx)
    with h5py.File(features_dir / f"{slide_id}.h5", "r") as f:
        vectors = f["features"][local_idx[sort_order]].astype(np.float32)
    out = np.empty_like(vectors)
    out[sort_order] = vectors
    return out


def sample_patch_vectors(
    slide_ids: list[str],
    manifest: pd.DataFrame,
    features_dir: Path,
    max_per_slide: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """指定したスライドそれぞれから最大 max_per_slide 枚のパッチ特徴量をランダム
    サンプルする(1スライドが検索候補プールを埋め尽くさないためのキャップ)。

    Returns:
        (vectors [N, dim] L2正規化済み, slide_id_per_row [N])
    """
    vecs, slides = [], []
    for slide_id in slide_ids:
        rows = manifest[manifest["slide_id"] == slide_id]
        if rows.empty:
            continue
        local_idx = rows["local_idx"].to_numpy()
        if len(local_idx) > max_per_slide:
            local_idx = rng.choice(local_idx, size=max_per_slide, replace=False)
        v = _read_patch_vectors(slide_id, np.sort(local_idx), features_dir)
        vecs.append(v)
        slides.extend([slide_id] * len(v))
    if not vecs:
        return np.empty((0, 0), dtype=np.float32), np.array([])
    return _l2_normalize(np.concatenate(vecs, axis=0)), np.array(slides)


def sample_negative_pool(
    manifest: pd.DataFrame,
    features_dir: Path,
    n: int,
    exclude_slides: set[str],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """正例スライドを除いたコーパス全体からランダムに n パッチをサンプルする
    (scripts/random_patch_baseline.py と同じ発想: 「所見によらないコーパスの地の
    分布」を負例として反映させる——コーパス中の圧倒的多数を占める平凡な正常組織が、
    そのまま負例側の多数派になる)。

    Returns:
        (vectors [N, dim] L2正規化済み, slide_id_per_row [N])
    """
    pool = manifest[~manifest["slide_id"].isin(exclude_slides)]
    n = min(n, len(pool))
    drawn = pool.iloc[np.sort(rng.choice(len(pool), size=n, replace=False))]
    vecs, slides = [], []
    for slide_id, rows in drawn.groupby("slide_id"):
        local_idx = np.sort(rows["local_idx"].to_numpy())
        v = _read_patch_vectors(slide_id, local_idx, features_dir)
        vecs.append(v)
        slides.extend([slide_id] * len(v))
    return _l2_normalize(np.concatenate(vecs, axis=0)), np.array(slides)


def train_logreg(pos_vecs: np.ndarray, neg_vecs: np.ndarray, C: float, seed: int):
    """正例 vs 負例(その他大半)の二値ロジスティック回帰(OvR)。

    class_weight="balanced" は負例が正例よりずっと多い(コーパスの大半が「その他」)
    ことを補正する——補正しないと「常に負例」を出すだけで見かけの精度が高くなる。
    """
    from sklearn.linear_model import LogisticRegression

    X = np.concatenate([pos_vecs, neg_vecs], axis=0)
    y = np.concatenate([np.ones(len(pos_vecs)), np.zeros(len(neg_vecs))])
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed)
    clf.fit(X, y)
    return clf


def loso_auroc_by_study(
    pos_vecs: np.ndarray,
    pos_slide_ids: np.ndarray,
    exp_id_of: dict,
    neg_train_vecs: np.ndarray,
    neg_test_vecs: np.ndarray,
    C: float,
    seed: int,
) -> pd.DataFrame:
    """study(EXP_ID)単位の leave-one-study-out 交差検証。

    負例は study に紐付かないため、train/test で完全に分離した固定プール
    (neg_train_vecs / neg_test_vecs、呼び出し側で一度だけ分割)を全フォールドで
    使い回す——正例側だけでなく負例側でも train/test 漏洩が無いようにするため。

    分類器が本当に「所見」を学習しているなら、held-outのstudy(=分類器が一度も
    見ていない化合物・撮影バッチ)の正例でもAUROCが高いはず。逆に「バッチ」を
    学習しているだけなら、held-out studyでAUROCがchance(0.5)近くまで落ちる
    (scripts/self_retrieval_diagnostic.py の batch_dominates_finding_rate と
    同じ検証の考え方)。

    n_exp_ids(正例が属するユニークstudy数)が1の場合、held-outを作れないため
    空のDataFrameを返す(呼び出し側でその旨を記録すること)。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    studies = sorted({exp_id_of[s] for s in pos_slide_ids})
    rows = []
    for held_out in studies:
        held_mask = np.array([exp_id_of[s] == held_out for s in pos_slide_ids])
        train_pos, test_pos = pos_vecs[~held_mask], pos_vecs[held_mask]
        if len(train_pos) == 0 or len(test_pos) == 0:
            continue
        X = np.concatenate([train_pos, neg_train_vecs], axis=0)
        y = np.concatenate([np.ones(len(train_pos)), np.zeros(len(neg_train_vecs))])
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed)
        clf.fit(X, y)

        X_test = np.concatenate([test_pos, neg_test_vecs], axis=0)
        y_test = np.concatenate([np.ones(len(test_pos)), np.zeros(len(neg_test_vecs))])
        score = clf.decision_function(X_test)
        rows.append({
            "held_out_exp_id": held_out,
            "n_train_pos": int(len(train_pos)),
            "n_test_pos": int(len(test_pos)),
            "auroc": float(roc_auc_score(y_test, score)),
        })
    return pd.DataFrame(rows)
