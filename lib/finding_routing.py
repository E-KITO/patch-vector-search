"""所見(finding)ラベルに基づいて、検索に使う索引(baseline / whiten)を選ぶ。

experiments/0025-0027 で、白色化(異方性除去)を焼き込んだ索引はコーパス内検索
(self_retrieval)を大きく改善する一方、atlas クエリでは所見によって効果が逆転
する(所見ごとに白色化との相性が系統的に違う)ことが分かった。詳細は README の
「experiments/0027: atlas GT 悪化の図版単位切り分け診断」参照。

クエリ画像は常に所見ラベルを伴って渡される前提(ユーザー確認済み — 所見不明の
自由アップロードは想定しない)なので、所見ラベルをキーに索引を出し分ける。
"""
from __future__ import annotations

from pathlib import Path

BASELINE_INDEX_DIR = Path("outputs/0018_20260909_build_faiss_index_deblank/default")
WHITEN_INDEX_DIR = Path("outputs/0025_20260910_whiten_index_rebuild/default/whiten_v1")

# experiments/0027 の atlas 図版単位診断(GT実測)で whiten が net で優位と確認
# された所見。GT実測の無い所見はすべて保守的に baseline に倒す — 「悪化しないと
# 確認できるまでは whiten を使わない」という方針(README 参照)。
WHITEN_FINDINGS = frozenset({
    "Hypertrophy",
    "Necrosis",
    "Proliferation, Kupffer cell",
    "Inclusion body, intracytoplasmic",
})


def index_dir_for_finding(finding_type: str) -> Path:
    """finding_type に対応する索引ディレクトリを返す。

    WHITEN_FINDINGS に無い所見(GT未検証の所見も含む)はすべて baseline。
    """
    return WHITEN_INDEX_DIR if finding_type in WHITEN_FINDINGS else BASELINE_INDEX_DIR
