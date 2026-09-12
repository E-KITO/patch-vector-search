"""所見ラベルに対する One-vs-Rest 線形分類器で、クエリタイルを「その所見らしいか」で
再スコアする(experiments/0035、複数所見での再試行はexperiments/0036)。コーパスGTが
無い所見向けにatlas図版自体を正例にする変種はexperiments/0037
(train_and_evaluate_classifier_from_atlas / loio_auroc_by_image)。

分類器の学習・batch-confound検証に加えて、atlas図版のタイル分割・埋め込み・スコア
ヒートマップ描画・検索腕(unfiltered/ovr_filtered)の実行という、この手法をどの所見に
適用する場合でも共通する処理一式もここに置く(experiments/0035と0036の重複を避ける)。

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


def resolve_atlas_images(project_root: Path, config: dict, target_finding: str) -> list[str]:
    """target_finding にマッピングされる atlas 図版フォルダの画像パスをすべて返す
    (scripts.validate_against_ground_truth.CATEGORIES 経由の解決、experiments/0034
    と同じ)。CATEGORIES に対応フォルダが無い所見は空リストを返す(呼び出し側で
    スキップすること)。"""
    import sys

    sys.path.insert(0, str(project_root))
    from lib.atlas_figures import query_images
    from scripts.validate_against_ground_truth import CATEGORIES

    atlas_root = project_root / config["atlas_root"]
    atlas_csv = project_root / config["atlas_figures_csv"]

    images: list[str] = []
    for folder_name, finding_type in CATEGORIES.items():
        if finding_type != target_finding:
            continue
        cat_dir = atlas_root / folder_name
        images.extend(str(p) for p in query_images(cat_dir, atlas_csv=atlas_csv))
    return images


def list_atlas_folders(atlas_root: Path, atlas_csv: Path) -> list[tuple[str, list[str]]]:
    """atlas_root配下の全所見フォルダを、コーパスGTの有無やCATEGORIESマッピングに
    関係なく列挙する(experiments/0029のresolve_findingsと同じロジック——GTが
    無い所見にも同じ枠組みを適用するexperiments/0037用)。resolve_atlas_images
    はCATEGORIES経由でコーパスfinding_typeに対応するフォルダしか拾えないため、
    25所見全部を対象にする場合はこちらを使う。

    Returns:
        [(folder_name, [image_path, ...]), ...](画像のあるフォルダのみ)
    """
    from lib.atlas_figures import query_images

    out = []
    for sub in sorted(p for p in atlas_root.iterdir() if p.is_dir()):
        imgs = [str(p) for p in query_images(sub, atlas_csv=atlas_csv)]
        if imgs:
            out.append((sub.name, imgs))
    return out


def tile_grid_crops(pil_image, tile_size: int, crop_size: int):
    """lib.query_embedding.embed_image_tiles / lib.visualize.plot_query_tile_scores
    と同一のグリッド分割・空白タイル除外ロジック(タイル原点座標付きで返す必要が
    あるため、plot_query_tile_scores と同じ理由で複製する——そちらのdocstring参照)。

    Returns:
        (pil_image(拡大後), kept_origins, kept_crops, n_blank_excluded)
    """
    from lib.query_embedding import _is_blank_tile
    from PIL import Image

    w, h = pil_image.size
    scale = max(crop_size / w, crop_size / h, 1.0)
    if scale > 1.0:
        pil_image = pil_image.resize(
            (max(crop_size, round(w * scale)), max(crop_size, round(h * scale))), Image.LANCZOS
        )
        w, h = pil_image.size

    xs = sorted(set(list(range(0, w - crop_size + 1, crop_size)) + [w - crop_size]))
    ys = sorted(set(list(range(0, h - crop_size + 1, crop_size)) + [h - crop_size]))
    origins = [(x, y) for y in ys for x in xs]
    crops = [pil_image.crop((x, y, x + crop_size, y + crop_size)) for x, y in origins]

    blank = [_is_blank_tile(c) for c in crops]
    if all(blank):
        blank = [False] * len(blank)

    kept_origins = [o for o, b in zip(origins, blank) if not b]
    kept_crops = [c for c, b in zip(crops, blank) if not b]
    if crop_size != tile_size:
        kept_crops = [c.resize((tile_size, tile_size), Image.LANCZOS) for c in kept_crops]
    return pil_image, kept_origins, kept_crops, sum(blank)


def embed_tile_crops(crops, tile_size: int) -> np.ndarray:
    """タイル(PIL Image)のリストをUNIエンコーダ(uni_v1)で埋め込む
    (lib.query_embedding.embed_image_tiles と同じ前処理・同じエンコーダキャッシュ)。"""
    import torch
    from torchvision import transforms
    from lib.query_embedding import _load_encoder, _normalize_rows

    encoder, device, dtype = _load_encoder(None)
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    batch = torch.stack([to_tensor(c) for c in crops]).to(device=device, dtype=dtype)
    with torch.no_grad():
        embeddings = encoder(batch).float().cpu().numpy().astype(np.float32)
    return _normalize_rows(embeddings)


def plot_classifier_tile_heatmap(image_path: str, origins, scores: np.ndarray, tile_size: int):
    """OvR分類器のdecision_functionスコアを、lib.visualize.plot_query_tile_scores と
    同じ見た目(元画像にタイルごとの半透明カラーオーバーレイ+カラーバー)で可視化する。"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import Normalize
    from lib.query_embedding import _load_rgb

    pil_image = _load_rgb(image_path)
    w, h = pil_image.size
    scale = max(tile_size / w, tile_size / h, 1.0)
    if scale > 1.0:
        from PIL import Image
        pil_image = pil_image.resize((max(tile_size, round(w * scale)), max(tile_size, round(h * scale))), Image.LANCZOS)
        w, h = pil_image.size

    cmap = plt.get_cmap("magma")
    norm = Normalize(vmin=float(scores.min()), vmax=float(scores.max()))

    fig, ax = plt.subplots(figsize=(w / 100, h / 100))
    ax.imshow(pil_image)
    for (x, y), score in zip(origins, scores):
        ax.add_patch(mpatches.Rectangle(
            (x, y), tile_size, tile_size, facecolor=cmap(norm(score)), alpha=0.5, edgecolor="none",
        ))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="OvR classifier decision_function score", shrink=0.7)
    ax.set_title("Per-tile OvR classifier score (higher = more finding-like)", fontsize=9)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    fig.tight_layout()
    return fig


def run_query_arm(
    *, arm_label: str, query_vecs: np.ndarray, patch_index, thumbnails_dir: Path,
    run_dir: Path, params: dict, logger,
) -> dict:
    """1つの腕(unfiltered / ovr_filtered)について検索・可視化を実行する
    (experiments/0034 の process_query_set と同じ構造)。"""
    from lib.visualize import plot_hit_patch_gallery, plot_slide_hits_on_thumbnail
    import matplotlib.pyplot as plt

    arm_dir = run_dir / f"query__{arm_label}"
    arm_dir.mkdir(exist_ok=True)
    gallery_dir = arm_dir / "patch_gallery"
    gallery_dir.mkdir(exist_ok=True)
    plots_dir = arm_dir / "thumbnail_plots"
    plots_dir.mkdir(exist_ok=True)

    top_slides = patch_index.search_top_slides_multi(
        query_vecs, k_candidates=params["k_candidates"], nprobe=params["nprobe"],
        top_n_slides=params["top_n_slides"],
    )
    top_slides.to_csv(arm_dir / "top_slides.csv", index=False)
    logger.info(f"[{arm_label}] top slides:\n{top_slides}")

    hit_pool = patch_index.search_similar_patches_multi(
        query_vecs, k=params["rerank_pool"], nprobe=params["nprobe"],
        rerank_pool=params["rerank_pool"], max_tiles_reranked=params["max_tiles_reranked"],
    )
    n_gal = 0
    for slide_id in top_slides["slide_id"].head(params["top_n_slides_to_plot"]):
        hits = hit_pool[hit_pool["slide_id"] == slide_id]
        if hits.empty:
            continue
        try:
            fig = plot_slide_hits_on_thumbnail(slide_id, hits, thumbnails_dir, patch_index.slide_meta)
            fig.savefig(plots_dir / f"{slide_id}.png", dpi=150)
            plt.close(fig)
        except Exception as e:
            logger.warning(f"[{arm_label}] thumbnail plot failed for slide {slide_id}: {e!r}")
        try:
            fig = plot_hit_patch_gallery(hits, params["raw_slide_dir"], patch_index.slide_meta)
            fig.savefig(gallery_dir / f"{slide_id}.png", dpi=150)
            plt.close(fig)
            n_gal += 1
        except Exception as e:
            logger.warning(f"[{arm_label}] gallery failed for slide {slide_id}: {e!r}")
    logger.info(f"[{arm_label}] galleries written: {n_gal}")
    return {
        "n_tiles": int(query_vecs.shape[0]),
        "n_top_slides": int(len(top_slides)),
        "top_slide_ids": top_slides["slide_id"].astype(str).tolist(),
        "n_galleries": n_gal,
    }


def train_and_evaluate_classifier(project_root, config: dict, manifest, target_finding: str, logger):
    """GT正例 vs コーパス全体ランダム負例で分類器を学習し、study単位LOSO AUROCで
    「所見」と「化合物・studyというバッチ」のどちらを学習しているかを確認する。

    manifest は呼び出し側で slide_id を str にキャストしたコピーを渡すこと
    (patch_index.manifest 本体は検索側の他メソッドが使うため——experiments/0035の
    main() 参照)。target_finding を config から読まず明示引数にしているのは、
    experiments/0036 のように同じ config で複数所見を順に処理する呼び出し元が
    あるため。

    GT正例スライドが1件も無い場合は空のdiagnosticsを返す(raiseしない——
    experiments/0036 のような複数所見スイープが1所見の欠測で全体停止しないため。
    呼び出し側は diagnostics["n_positive_slides"] == 0 で判定すること)。

    Returns:
        (final_classifier | None, loso_df | None, diagnostics: dict)
    """
    rng = np.random.default_rng(config.get("seed", 42))
    features_dir = project_root / config["features_dir"]

    corpus_slides = set(manifest["slide_id"].astype(str).unique())
    gt = load_gt_slides_for_finding(target_finding, corpus_slides, project_root / config["gt_csv"])
    if gt.empty:
        logger.warning(f"no corpus GT slides found for finding_type={target_finding!r} — skipping")
        return None, None, {"target_finding": target_finding, "n_positive_slides": 0}
    logger.info(f"GT slides for {target_finding!r}: {len(gt)} "
                f"(n_compounds={gt['COMPOUND_NAME'].nunique()}, n_exp_ids={gt['EXP_ID'].nunique()})")

    pos_vecs, pos_slide_ids = sample_patch_vectors(
        gt["slide_id"].tolist(), manifest, features_dir,
        max_per_slide=config["max_positive_patches_per_slide"], rng=rng,
    )
    logger.info(f"positive patches sampled: {pos_vecs.shape[0]} from {gt['slide_id'].nunique()} slides")

    neg_vecs, neg_slide_ids = sample_negative_pool(
        manifest, features_dir, n=config["n_negative_patches"],
        exclude_slides=set(gt["slide_id"]), rng=rng,
    )
    logger.info(f"negative patches sampled: {neg_vecs.shape[0]} from {len(set(neg_slide_ids))} slides")

    # 負例をLOSO専用のtrain/testに一度だけ分割(train/test漏洩を避けるため全フォールドで
    # 使い回す。loso_auroc_by_study のdocstring参照)。
    perm = rng.permutation(len(neg_vecs))
    n_test = int(round(len(neg_vecs) * config["neg_test_fraction"]))
    neg_test_vecs = neg_vecs[perm[:n_test]]
    neg_train_vecs = neg_vecs[perm[n_test:]]

    exp_id_of = dict(zip(gt["slide_id"], gt["EXP_ID"].astype(str)))
    n_exp_ids = gt["EXP_ID"].nunique()
    if n_exp_ids >= 2:
        loso_df = loso_auroc_by_study(
            pos_vecs, pos_slide_ids, exp_id_of, neg_train_vecs, neg_test_vecs,
            C=config["classifier_C"], seed=config.get("seed", 42),
        )
        if len(loso_df):
            logger.info(f"LOSO AUROC (median={loso_df['auroc'].median():.3f}, "
                        f"min={loso_df['auroc'].min():.3f}, max={loso_df['auroc'].max():.3f}, "
                        f"n_folds={len(loso_df)}):\n{loso_df}")
        else:
            logger.warning("LOSO produced 0 usable folds (every study had all-or-nothing positives)")
    else:
        loso_df = None
        logger.warning(
            f"only {n_exp_ids} distinct EXP_ID among {target_finding!r} GT slides — "
            "cannot leave-one-study-out; AUROC would be indistinguishable from "
            "memorizing that single study (see README self_retrieval_diagnostic "
            "n_exp_ids caveat). Skipping LOSO."
        )

    final_clf = train_logreg(pos_vecs, np.concatenate([neg_train_vecs, neg_test_vecs]),
                              C=config["classifier_C"], seed=config.get("seed", 42))

    diagnostics = {
        "target_finding": target_finding,
        "n_positive_patches": int(pos_vecs.shape[0]),
        "n_positive_slides": int(gt["slide_id"].nunique()),
        "n_compounds": int(gt["COMPOUND_NAME"].nunique()),
        "n_exp_ids": int(n_exp_ids),
        "n_negative_patches": int(neg_vecs.shape[0]),
        "n_negative_slides": int(len(set(neg_slide_ids))),
        "loso_median_auroc": float(loso_df["auroc"].median()) if loso_df is not None and len(loso_df) else None,
        "loso_min_auroc": float(loso_df["auroc"].min()) if loso_df is not None and len(loso_df) else None,
        "loso_max_auroc": float(loso_df["auroc"].max()) if loso_df is not None and len(loso_df) else None,
        "loso_n_folds": int(len(loso_df)) if loso_df is not None else 0,
    }
    return final_clf, loso_df, diagnostics


def embed_atlas_images_as_positives(images: list[str], tile_size: int) -> tuple[np.ndarray, np.ndarray]:
    """コーパス内GTが無い所見向け: atlas図版自体のタイルを正例として埋め込む
    (experiments/0037)。

    corpus GT スライドのパッチを正例にする train_and_evaluate_classifier と違い、
    ここでの「正例」は図版1枚の全タイル——GTスライドの全パッチを正例とする既存の
    弱ラベル(README「評価に使ったデータとその限界」)よりさらに一段弱い。atlas
    図版は「所見部位を含む図版全体」であり、矢印注釈・周囲の正常組織・余白を含む
    タイルも区別なく正例に混入する(README「アトラス画像1枚は所見部位を含む図版
    全体」)。目視での結果解釈時にこの点を割り引くこと。

    Returns:
        (vectors [N, dim] L2正規化済み, image_path_per_row [N])
    """
    from PIL import Image as PILImage

    vecs, image_ids = [], []
    for image_path in images:
        pil_image = PILImage.open(image_path).convert("RGB")
        _, _, crops, _ = tile_grid_crops(pil_image, tile_size, tile_size)
        v = embed_tile_crops(crops, tile_size)
        vecs.append(v)
        image_ids.extend([image_path] * len(v))
    return np.concatenate(vecs, axis=0), np.array(image_ids)


def loio_auroc_by_image(
    pos_vecs: np.ndarray,
    pos_image_ids: np.ndarray,
    neg_train_vecs: np.ndarray,
    neg_test_vecs: np.ndarray,
    C: float,
    seed: int,
) -> pd.DataFrame:
    """atlas図版単位の leave-one-image-out 交差検証(loso_auroc_by_studyのGT無し版)。

    コーパスGTが無い所見では study(EXP_ID) という単位がそもそも無いため、代わりに
    「同じ所見の他の図版から学習して、見たことのない図版のタイルを当てられるか」を
    見る。これは**自己一貫性チェックであり、コーパス上の実際の正解に紐づいた検証
    ではない**——atlas図版群自体が互いに似ていれば高AUROCになるし、逆に図版ごとに
    見え方が大きく違う所見(倍率不揃い・染色差)では低くなりうるが、どちらの場合も
    「コーパス内でこの所見を正しく引けているか」は保証しない(loso_auroc_by_study
    のdocstring、および README「評価に使ったデータとその限界」参照)。

    図版が1枚しかない所見はheld-outを作れないため空のDataFrameを返す。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    images = sorted(set(pos_image_ids.tolist()))
    rows = []
    for held_out in images:
        held_mask = pos_image_ids == held_out
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
            "held_out_image": Path(held_out).stem,
            "n_train_pos": int(len(train_pos)),
            "n_test_pos": int(len(test_pos)),
            "auroc": float(roc_auc_score(y_test, score)),
        })
    return pd.DataFrame(rows)


def train_and_evaluate_classifier_from_atlas(
    project_root, config: dict, manifest, images: list[str], target_finding: str,
    exclude_slides: set[str], logger,
):
    """train_and_evaluate_classifier のコーパスGT無し版(experiments/0037)。

    正例を GT スライドのパッチではなく images(atlas図版)自体のタイルから作る。
    負例サンプリングは従来通りコーパス全体からだが、CATEGORIES 経由で対応する
    コーパス finding_type が分かっている場合は exclude_slides(呼び出し側が
    load_gt_slides_for_finding で解決)でその既知GTスライドを負例から除外できる
    (コーパスに本物が混じって負例を汚染するのを避けるため)。対応が無ければ
    exclude_slides は空集合でよい。

    Returns:
        (final_classifier, loio_df | None, diagnostics: dict)
    """
    rng = np.random.default_rng(config.get("seed", 42))
    features_dir = project_root / config["features_dir"]
    tile_size = int(config.get("tile_size", 224))

    pos_vecs, pos_image_ids = embed_atlas_images_as_positives(images, tile_size)
    logger.info(f"positive tiles sampled: {pos_vecs.shape[0]} from {len(images)} atlas image(s)")

    neg_vecs, neg_slide_ids = sample_negative_pool(
        manifest, features_dir, n=config["n_negative_patches"],
        exclude_slides=exclude_slides, rng=rng,
    )
    logger.info(f"negative patches sampled: {neg_vecs.shape[0]} from {len(set(neg_slide_ids))} slides")

    perm = rng.permutation(len(neg_vecs))
    n_test = int(round(len(neg_vecs) * config["neg_test_fraction"]))
    neg_test_vecs = neg_vecs[perm[:n_test]]
    neg_train_vecs = neg_vecs[perm[n_test:]]

    n_images = len(set(pos_image_ids.tolist()))
    if n_images >= 2:
        loio_df = loio_auroc_by_image(
            pos_vecs, pos_image_ids, neg_train_vecs, neg_test_vecs,
            C=config["classifier_C"], seed=config.get("seed", 42),
        )
        if len(loio_df):
            logger.info(f"LOIO AUROC (median={loio_df['auroc'].median():.3f}, "
                        f"min={loio_df['auroc'].min():.3f}, max={loio_df['auroc'].max():.3f}, "
                        f"n_folds={len(loio_df)}):\n{loio_df}")
        else:
            logger.warning("LOIO produced 0 usable folds")
    else:
        loio_df = None
        logger.warning(f"only {n_images} atlas image for {target_finding!r} — cannot "
                        "leave-one-image-out. Skipping LOIO.")

    final_clf = train_logreg(pos_vecs, np.concatenate([neg_train_vecs, neg_test_vecs]),
                              C=config["classifier_C"], seed=config.get("seed", 42))

    diagnostics = {
        "target_finding": target_finding,
        "n_positive_tiles": int(pos_vecs.shape[0]),
        "n_atlas_images": n_images,
        "n_negative_patches": int(neg_vecs.shape[0]),
        "n_negative_slides": int(len(set(neg_slide_ids))),
        "loio_median_auroc": float(loio_df["auroc"].median()) if loio_df is not None and len(loio_df) else None,
        "loio_min_auroc": float(loio_df["auroc"].min()) if loio_df is not None and len(loio_df) else None,
        "loio_max_auroc": float(loio_df["auroc"].max()) if loio_df is not None and len(loio_df) else None,
        "loio_n_folds": int(len(loio_df)) if loio_df is not None else 0,
    }
    return final_clf, loio_df, diagnostics


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
