"""Manual ROI crop ceiling check (README「今後やること」: クエリ画像を人手でROIクロップ
した場合の性能上限を測る)。

README の「IVFクラスタリング仮説の検証状況」(2026-08-19)で、nprobe/PQ量子化誤差の
どちらも否定され、「原因はモデル・索引側ではなく、クエリ画像(NNLアトラス図版)の内容側
(所見と無関係なタイルの混入など)にある可能性が高い」という結論に至った。この診断は
その仮説を実際に検証する — GT対応7カテゴリ・計24枚のアトラス画像を目視で確認し、所見が
明確に画像の一部分に限局している場合のみ手動でROIをクロップし、全画像タイル分割
(embed_image_tiles、既定)と比較する。

クロップ根拠: 24枚を目視確認した結果、当初は7カテゴリ中2カテゴリ(Necrosis、
Extramedullary hematopoiesis)・最大6枚を「巣状の所見だからクロップ可能」と判断したが、
「非専門家の目視判断をどこまで信用してよいか」を1枚ずつ検証し直したところ、大半は
根拠不十分と判明し撤回した。最終的にクロップを採用したのは以下の2枚のみ:

  - imgi_6_figure-001-a71206_large.jpg(Extramedullary hematopoiesis): 唯一、
    実際の矢印注釈が造血細胞塊2か所を直接指しており、クロップ範囲がその矢印の
    先端を含むことを確認済み。
  - imgi_11_figure-006-a30016_large.jpg(Necrosis): 矢印は無いが、(1) 赤血球のみが
    充満した出血・うっ血ではなく、崩壊した索状組織構造と核性デブリ(核崩壊産物)が
    赤色領域内に残存している、(2) 境界が血管・出血に典型的な円形ではなく、
    肝小葉の帯状壊死に典型的な地図状・帯状の形をしている、という2点から
    「壊死ではなく出血/うっ血ではないか」という対立仮説を積極的に排除できると判断した。

以下は撤回した画像とその理由(検証の過程を残すため記録):

  - imgi_12_figure-007-a71459_large.jpg(Necrosis): 壊死結節候補内の白い亀裂状
    パターンが、崩壊した組織構造か出血性のフィブリン網かを判別できなかった。
  - imgi_14_figure-009-a53140_large.jpg(Necrosis): 均一な赤色領域ではなく密な
    炎症性浸潤で、壊死巣周囲の反応なのか壊死そのものかを画像単独では確認できない。
  - imgi_9_figure-004-a50916_large.jpg(Extramedullary hematopoiesis): 候補が
    1つしか無かったが、これは「他に競合候補が無い」だけであり「正常な門脈域
    (portal tract)のリンパ球カフではないと確認した」ことにはならない。実際、
    大きく透明な円形ルーメン(門脈枝の疑い)に接していた。
  - imgi_8_figure-003-a50917_large.jpg(Extramedullary hematopoiesis): 複数の
    類似候補が散在し、当初選んだ候補(血管周囲)はむしろ正常な門脈域である
    可能性が高いと判明した。

根拠が薄い画像は誤った領域をクロップしてかえってシグナルを損なうより、無クロップ
(baseline_v1と同じ)の方が安全という判断で一貫させている。

なお、Necrosis・Extramedullary hematopoiesis以外の5カテゴリ(Hypertrophy、
Increased mitosis、Glycogen accumulation/depletion、Kupffer cell hyperplasia、
Cytoplasmic inclusions)はそもそもクロップ候補にしていない — 今回の画像はいずれも
既に生検写真1枚全体が均一に所見を示すクローズアップで、除外すべき「所見と無関係な
領域」が視覚的に存在しないため(Kupffer cell hyperplasiaは特に、洞様毛細血管に
沿ったKupffer細胞の密度変化という性質上、健常組織と視覚的に区別できる巣が
見当たらなかった)。

以上により、残る22枚(diffuse 5カテゴリ18枚 + 検証の結果撤回した4枚)は全画像の
まま扱い、同一実行内でのno-cropコントロールとして機能する。サンプルサイズは小さい
(Necrosis 4枚中1枚、Hematopoiesis 4枚中1枚のみクロップ)ため、この診断は
「クロップが効くか」を統計的に検出する力は弱く、あくまで「確実な根拠がある場合、
クロップは検索順位を動かすか」という定性的なシグナルを見るためのものと位置づける。

Usage:
    .venv/bin/python3 scripts/manual_roi_crop_diagnostic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from lib.query_embedding import embed_image_tiles
from validate_against_ground_truth import default_pipelines, run_comparison

# {filename: (left, top, right, bottom)}, as fractions of image width/height.
# Only 2 of the 24 GT-category atlas images survived a deliberately strict
# bar: a crop is included here only if there is concrete, checkable evidence
# against the most plausible alternative explanation for what's in frame
# (not just "no competing candidate visible" or a generic textbook
# association) -- see this module's docstring for the full list of images
# considered and why the other 22 were left uncropped.
MANUAL_CROPS: dict[str, tuple[float, float, float, float]] = {
    # Extramedullary hematopoiesis: two small hematopoietic cell clusters in
    # a narrow vertical strip, confirmed against this image's own arrow
    # annotations (the only image in the 24 with a real arrow pointing at
    # the finding) -- verified the crop box contains both arrow tips.
    "imgi_6_figure-001-a71206_large.jpg": (0.30, 0.05, 0.70, 1.0),
    # Necrosis: no arrow, but the red zone shows (1) residual reticular
    # tissue architecture and scattered pyknotic/karyorrhectic nuclear
    # debris -- not the anucleate, texture-less field pure hemorrhage/
    # congestion would produce -- and (2) a geographic/zonal boundary shape
    # (spanning the full frame height) typical of zonal hepatocellular
    # necrosis, not the rounded boundary of a vessel or hemorrhagic focus.
    # Both points argue against the main alternative explanation (a
    # blood-filled space, not necrosis) rather than merely asserting no
    # rival candidate was visible.
    "imgi_11_figure-006-a30016_large.jpg": (0.0, 0.0, 0.42, 1.0),
}


def embed_manual_roi(images) -> np.ndarray:
    """embed_image_tiles, but crop each image to its MANUAL_CROPS box first
    (full image, i.e. a no-op crop, for any filename not in MANUAL_CROPS)."""
    tiles = []
    for f in images:
        image = Image.open(f).convert("RGB")
        left, top, right, bottom = MANUAL_CROPS.get(Path(f).name, (0.0, 0.0, 1.0, 1.0))
        w, h = image.size
        cropped = image.crop((round(left * w), round(top * h), round(right * w), round(bottom * h)))
        tiles.append(embed_image_tiles(cropped, tile_size=224))
    return np.concatenate(tiles, axis=0)


def main() -> None:
    # Reuse default_pipelines()'s already-loaded v1 index rather than
    # loading it a second time; uni_v2 is irrelevant to this check so its
    # pipeline entry (also built by default_pipelines()) is simply dropped.
    pipelines = {"baseline_v1": default_pipelines()["baseline_v1"]}
    pipelines["manual_roi"] = (pipelines["baseline_v1"][0], embed_manual_roi)

    df = run_comparison(pipelines, nprobe=64)
    out_path = Path("outputs/gt_validation_manual_roi_crop.csv")
    out_path.parent.mkdir(exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
