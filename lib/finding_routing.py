"""所見(finding)ラベルに基づいて、検索に使う索引/クエリ経路を選ぶ。

experiments/0025-0031 で、次の2種類の変換がそれぞれ所見によって効果が逆転する
(GT実測で見て有利・不利がはっきり分かれる)ことが分かった:

  - whiten: 白色化(異方性除去)を FAISS 索引にベイクしたもの(experiments/0025)。
    コーパス内検索(self_retrieval)は大きく改善するが、atlas クエリでは所見に
    よって悪化する(README「experiments/0027」)。
  - macenko: Macenko染色正規化コーパス(experiments/0033、背景除去済み)+
    per-tile Macenko正規化クエリ。集計のGT比較では baseline_v1 に僅差で
    勝てず棚上げされていたが(README「クエリ側染色正規化の再検証」)、所見
    ごとには明確な勝敗パターンがある(README「experiments/0031」)。

クエリ画像は常に所見ラベルを伴って渡される前提(ユーザー確認済み — 所見不明の
自由アップロードは想定しない)なので、所見ラベルをキーに baseline/whiten/macenko
を出し分ける。GT実測の無い所見はすべて保守的に baseline(「悪化しないと確認
できるまでは使わない」方針)。

Hypertrophy と Inclusion body, intracytoplasmic は whiten・macenko の両方で
GT改善が確認されたため、GT改善幅がより大きい macenko を採用した(ユーザー確認
済み、README「experiments/0031」の三択問題)。
"""
from __future__ import annotations

from pathlib import Path

PLAIN_FEATURES_DIR = Path("data/trident_processed/20x_224px_0px_overlap/features_uni_v1")

BASELINE_INDEX_DIR = Path("outputs/0018_20260909_build_faiss_index_deblank/default")

WHITEN_INDEX_DIR = Path("outputs/0025_20260910_whiten_index_rebuild/default/whiten_v1")
# whiten 索引が FAISS にベイクした変換と同一のパラメータ。PatchIndex.transform に
# 渡すことで、パッチ単位の厳密re-rank(_exact_similarity、生 h5 読み)も索引の
# 近似候補プールと同じ空間で計算される(load_index_for_finding 経由で必ず対で使う)。
WHITEN_TRANSFORM_PATH = Path("outputs/0025_20260910_whiten_index_rebuild/default/transforms/whiten.npz")

# 背景除去済み(experiments/0032)で作り直した Macenko 索引(experiments/0033)。
# whiten と違い別コーパス(別 h5 特徴量ファイル群)なので、features_dir も
# 専用のものに切り替える必要がある — PLAIN_FEATURES_DIR は使えない。
MACENKO_INDEX_DIR = Path("outputs/0033_20260911_build_faiss_index_macenko_deblank/default")
MACENKO_FEATURES_DIR = Path("data/trident_processed_uni_v1_macenko/20x_224px_0px_overlap/features_uni_v1_macenko")
# クエリ側の per-tile Macenko 正規化基準パッチ。scripts/validate_against_ground_truth.py
# の V1_MACENKO_STAIN_REFERENCE と同一。
MACENKO_STAIN_REFERENCE = Path("data/baseline/63958_x38976_y7616.png")

# experiments/0027/0028 の atlas 図版単位診断(GT実測)で whiten が net で優位と
# 確認された所見。Hypertrophy と Inclusion body はここには含めない(macenko の
# 方がGT改善幅が大きく、そちらを採用したため — 下記 MACENKO_FINDINGS 参照)。
WHITEN_FINDINGS = frozenset({
    "Necrosis",
    "Proliferation, Kupffer cell",
})

# experiments/0031 で Macenko が有利と確認された所見。Hypertrophy / Inclusion
# body は experiments/0024 の7カテゴリGT比較(outputs/gt_validations/
# gt_validation_v1_macenko_per_tile.csv)で macenko が有利(49→14 / 480→97)、
# かつ whiten でも有利という三択ケースだったが、GT改善幅がより大きい macenko を
# 採用。Increased mitosis も同GTで有利(7→1)、whitenでは元々baseline側だった
# ので競合なし。Fatty Change(コーパス側ラベル "Degeneration, fatty")はGT
# スライドが0枚のため定量評価はできないが、experiments/0031 の目視診断で
# macenko側が視覚的に妥当な一致を示した。
MACENKO_FINDINGS = frozenset({
    "Hypertrophy",
    "Increased mitosis",
    "Inclusion body, intracytoplasmic",
    "Degeneration, fatty",
})


def index_dir_for_finding(finding_type: str) -> Path:
    """finding_type に対応する索引ディレクトリを返す。

    macenko 索引を使う所見は features_dir も MACENKO_FEATURES_DIR に切り替える
    必要があるため、索引ディレクトリだけを使う呼び出し元(slide-levelランキング
    のみ、パッチ単位の厳密re-rankや所見画像自体の埋め込みをしない場合)以外は
    load_index_for_finding / query_tile_transform_for_finding を使うこと。
    """
    if finding_type in MACENKO_FINDINGS:
        return MACENKO_INDEX_DIR
    if finding_type in WHITEN_FINDINGS:
        return WHITEN_INDEX_DIR
    return BASELINE_INDEX_DIR


def query_tile_transform_for_finding(finding_type: str):
    """finding_type が macenko ルーティング対象なら、embed_image_tiles の
    tile_transform に渡す per-tile Macenko 正規化関数を返す(それ以外は None)。

    macenko 索引はコーパス側パッチも per-tile Macenko 正規化済みなので、atlas
    図版などをクエリにする場合はこの変換を通してから埋め込まないと、索引と
    クエリで染色空間が食い違う(scripts/validate_against_ground_truth.py の
    _embed_v1_macenko_normalized と同じ考え方)。

    注意: experiments/0019 のようにコーパス自身のスライドの生h5ベクトルを
    直接クエリにする(embed_image_tiles を経由しない)フローでは、この関数の
    出番はない — その場合クエリ側 h5 も MACENKO_FEATURES_DIR から読む必要が
    あるが、現状 macenko ルーティング対象の所見はどれも成果物トラック
    (experiments/0019 の SUPPORTED_FINDINGS)に入っていないため未対応。
    """
    if finding_type not in MACENKO_FINDINGS:
        return None

    from lib.torchstain_normalize import normalize_to_reference

    def _tile_transform(tile):
        try:
            return normalize_to_reference(tile, MACENKO_STAIN_REFERENCE)
        except Exception:
            return tile

    return _tile_transform


def load_index_for_finding(finding_type: str):
    """finding_type に対応する PatchIndex を、索引・features_dir・変換をすべて
    正しく対にして読み込んで返す。

    - whiten 索引: PatchIndex.transform に変換パラメータを渡すことで、
      `search_similar_patches`(_exact_similarity によるパッチ単位の厳密
      re-rank)が索引の近似候補プールと同じ空間で計算されるようにする
      (README「experiments/0025」の「昇格時のTODO」参照)。
    - macenko 索引: features_dir を MACENKO_FEATURES_DIR に切り替える(別コーパス
      なので PLAIN_FEATURES_DIR は使えない)。クエリ画像の埋め込みには
      query_tile_transform_for_finding も併用すること。
    - それ以外: baseline(変換なし、PLAIN_FEATURES_DIR)。
    """
    from lib.search import PatchIndex

    if finding_type in MACENKO_FINDINGS:
        return PatchIndex.load(
            index_path=MACENKO_INDEX_DIR / "index.faiss",
            manifest_path=MACENKO_INDEX_DIR / "manifest.parquet",
            slide_meta_path=MACENKO_INDEX_DIR / "slide_meta.parquet",
            features_dir=MACENKO_FEATURES_DIR,
        )
    if finding_type in WHITEN_FINDINGS:
        return PatchIndex.load(
            index_path=WHITEN_INDEX_DIR / "index.faiss",
            manifest_path=WHITEN_INDEX_DIR / "manifest.parquet",
            slide_meta_path=WHITEN_INDEX_DIR / "slide_meta.parquet",
            features_dir=PLAIN_FEATURES_DIR,
            transform_path=WHITEN_TRANSFORM_PATH,
        )
    return PatchIndex.load(
        index_path=BASELINE_INDEX_DIR / "index.faiss",
        manifest_path=BASELINE_INDEX_DIR / "manifest.parquet",
        slide_meta_path=BASELINE_INDEX_DIR / "slide_meta.parquet",
        features_dir=PLAIN_FEATURES_DIR,
    )
