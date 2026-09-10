"""NNL アトラス図版メタデータ (`data/query/nnl_liver_atlas_figures.csv`) の参照ヘルパー。

CSV は 2026-09-08 に WebFetch で NTP Nonneoplastic Lesion Atlas の全 25 lesion ページ
から収集したもの (経緯は `data/query/nnl_liver_atlas_README.md`)。1 行 = 1 図版で、
`is_normal_control` / `magnification_or_xref` / `caption` / `local_folder` /
`local_filename` 等の列を持つ。

アトラス図版をクエリに使う側 (`scripts/validate_against_ground_truth.py`,
`experiments/0020`) が lesion フォルダを素朴に glob すると、**対照図版**
("Normal liver ... age and sex matched for comparison with Figure N") まで検索
クエリに含めてしまう。肝 NNL で対照図版を持つのは Atrophy (2 枚) と
Hepatocyte - Hypertrophy (2 枚) の計 4 枚のみだが、Hypertrophy は GT 7 カテゴリに
入るため atlas GT 比較 (`validate_against_ground_truth.py`) を汚染する。ここで除外する。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

DEFAULT_ATLAS_CSV = Path("data/query/nnl_liver_atlas_figures.csv")
_IMG_EXTS = (".jpg", ".jpeg", ".png")


@lru_cache(maxsize=8)
def _load(atlas_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(atlas_csv, dtype=str).fillna("")
    df["is_normal_control"] = df["is_normal_control"].str.strip().str.lower()
    return df


def normal_control_filenames(atlas_csv: Path | str = DEFAULT_ATLAS_CSV) -> set[str]:
    """`is_normal_control=yes` の図版の `local_filename` 集合 (ローカルに無い図版は空文字 → 除く)。"""
    df = _load(Path(atlas_csv))
    return set(df.loc[df["is_normal_control"] == "yes", "local_filename"]) - {""}


def query_images(
    cat_dir: Path | str,
    *,
    exclude_normal_control: bool = True,
    atlas_csv: Path | str = DEFAULT_ATLAS_CSV,
) -> list[Path]:
    """lesion フォルダ内のクエリ画像を sorted で返す。既定で対照図版を除外する。

    fail-safe: CSV が見つからない / 図版が CSV に載っていない場合はそのファイルを
    残す (黙って落とさない)。
    """
    cat_dir = Path(cat_dir)
    if not cat_dir.is_dir():
        return []
    imgs = sorted(p for p in cat_dir.iterdir() if p.suffix.lower() in _IMG_EXTS)
    if not exclude_normal_control:
        return imgs
    try:
        drop = normal_control_filenames(atlas_csv)
    except FileNotFoundError:
        return imgs
    return [p for p in imgs if p.name not in drop]
