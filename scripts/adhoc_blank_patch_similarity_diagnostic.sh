#!/bin/bash
#SBATCH --job-name=adhoc_blank_patch_similarity_diagnostic
#SBATCH --partition=medium-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/adhoc_blank_patch_similarity_diagnostic/%j.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/adhoc_blank_patch_similarity_diagnostic/%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=2:00:00
# scripts/blank_patch_similarity_diagnostic.py:
# スライド背景パッチが各所見のクエリに対してどれくらいの類似度を出すかを測る。
# job 9962 の granular eosinophilic が候補の93%を背景で埋めた原因が
#   (a) UNI v1 がこの所見の形態を背景クラスタの近くに置いている(埋め込みの問題、
#       uni_v2 検討の論拠になる)か
#   (b) この所見に本当に似た組織がコーパスに乏しく、どの所見に対しても中程度の
#       類似度を持つ背景が相対的に浮上しただけ(データ側の事実、再埋め込みでは
#       解決しない)か
# を切り分ける。両者は逆の予測をするので測れば決着がつく。
#
# 索引もエンコーダも使わない(特徴量は h5 の格納値をそのまま読む)ので GPU 不要。
# 支配的なコストは openslide での 2000パッチのクロップ。実測を見て N_SAMPLE を調整する。
# パラメータは .py 側の定数(N_SAMPLE / FINDINGS / SAT_THRESHOLD 等)。

set -euo pipefail

PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"

echo "Running scripts/blank_patch_similarity_diagnostic.py on $(hostname) ..."

if command -v apptainer &>/dev/null && [ -n "${SIF_PATH:-}" ] && [ -f "${SIF_PATH}" ]; then
    apptainer exec \
        "${SIF_PATH}" \
        bash -c "
            set -euo pipefail
            source ${PROJECT_ROOT}/.venv/bin/activate
            cd ${PROJECT_ROOT}
            python scripts/blank_patch_similarity_diagnostic.py
        "
else
    echo "⚠️ Apptainer not found or SIF_PATH not set. Running on host system."
    source "${PROJECT_ROOT}/.venv/bin/activate"
    cd "${PROJECT_ROOT}"
    python scripts/blank_patch_similarity_diagnostic.py
fi

echo "Done. Results written to outputs/gt_validations/blank_patch_similarity_{patches,summary}.csv"
