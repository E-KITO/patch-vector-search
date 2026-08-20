"""Does a *correctly implemented* Macenko query-side stain normalization help
v1 retrieval, where the previously-tested implementation didn't?

READMEの「何を使うべきか」表は`stain_reference=`(クエリ側Macenko正規化)を非推奨
としており、その根拠は`scripts/compare_query_normalization.py`である。しかし
そのスクリプトのembed_fn(`lib.query_embedding.embed_image(..., stain_reference=...)`)
は内部で`lib.stain_normalize.MacenkoNormalizer`(自作のfrom-scratch numpy実装)を
使っている — `lib/torchstain_normalize.py`のdocstring自身が測定した通り、この自作
実装の自己一致性は0.5〜0.84(本来1.0に近いはず)しかない。一方`torchstain`ベースの
実装(`lib.torchstain_normalize`)は0.96〜0.996で、既にuni_v2コーパス
(`scripts/validate_against_ground_truth.py::_embed_v2_normalized`)向けには
使われているが、v1インデックスに対しては一度も試されていない。

つまり「v1でクエリ側染色正規化は効果が無い」という既存の結論は、精度に問題のある
正規化実装によるものであり、正確な実装での再検証がまだ行われていない。この診断は
それを埋める — 正規化はクエリ画像1枚を埋め込む直前にのみ適用され、コーパス側
(既存のh5特徴量・FAISSインデックス)は一切変更しないため、コーパス全体の
再埋め込みは不要。既存のv1インデックス(experiments/0001+0002)をそのまま使う。

基準パッチ: `scripts/compare_query_normalization.py`が使っていたのと同じ
`63958_x38976_y7616.png`(`scripts/select_average_patch.py`が選んだ候補#01)。
`outputs/average_patch_candidates/`・`data/baseline/`ともにこの環境には
存在しないため、`data/moo_collected_tggate_wsi/raw_wsi/63958.svs`の記録済み
座標から`lib.raw_patch.crop_patch`で再現し、`data/baseline/`に保存し直す
(以後の実行でも同じ基準パッチを再利用できるようにするため)。

Usage:
    .venv/bin/python3 scripts/torchstain_query_normalization_diagnostic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from lib.query_embedding import embed_image_tiles
from lib.raw_patch import crop_patch
from lib.torchstain_normalize import normalize_to_reference
from validate_against_ground_truth import default_pipelines, run_comparison

REFERENCE_SLIDE_ID = "63958"
REFERENCE_COORD_X = 38976
REFERENCE_COORD_Y = 7616
RAW_SLIDE_DIR = Path("data/moo_collected_tggate_wsi/raw_wsi")
REFERENCE_PATH = Path("data/baseline/63958_x38976_y7616.png")


def _ensure_reference_patch(patch_index) -> None:
    if REFERENCE_PATH.exists():
        return
    patch_size_level0 = int(patch_index.slide_meta.loc[REFERENCE_SLIDE_ID, "patch_size_level0"])
    reference_patch = crop_patch(
        REFERENCE_SLIDE_ID, REFERENCE_COORD_X, REFERENCE_COORD_Y, RAW_SLIDE_DIR, patch_size_level0
    )
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    reference_patch.save(REFERENCE_PATH)


def embed_torchstain_normalized(images) -> np.ndarray:
    tiles = []
    for f in images:
        normed = normalize_to_reference(str(f), str(REFERENCE_PATH))
        tiles.append(embed_image_tiles(normed, tile_size=224))
    return np.concatenate(tiles, axis=0)


def main() -> None:
    pipelines = {"baseline_v1": default_pipelines()["baseline_v1"]}
    patch_index = pipelines["baseline_v1"][0]

    _ensure_reference_patch(patch_index)
    pipelines["torchstain_normalized_v1"] = (patch_index, embed_torchstain_normalized)

    df = run_comparison(pipelines, nprobe=64)
    out_path = Path("outputs/gt_validation_torchstain_query_normalization.csv")
    out_path.parent.mkdir(exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
