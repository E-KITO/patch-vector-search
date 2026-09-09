#!/bin/bash
#SBATCH --job-name=adhoc_self_retrieval_diagnostic_deblank
#SBATCH --partition=small-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/adhoc_self_retrieval_diagnostic_deblank/%j.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/adhoc_self_retrieval_diagnostic_deblank/%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=1:00:00
# scripts/self_retrieval_diagnostic.py を背景除去済みインデックス(experiments/0018)で回す。
# adhoc_self_retrieval_diagnostic.sh の 0018 版 — env-var(SRD_INDEX_DIR)が sbatch に
# 伝播しなかった(job 10494 は既定の 0002 で走ってしまった)ため、--index-dir を
# ベタ書きした専用スクリプトにしている。bare `sbatch scripts/adhoc_self_retrieval_diagnostic_deblank.sh` で可。
#
# 出力は outputs/gt_validations/self_retrieval_diagnostic_0018.csv(スクリプト側が
# 非既定 --index-dir を検出して実験IDを付ける)。0002 版の
# self_retrieval_diagnostic.csv は上書きしない。
#
# UNI エンコーダは呼ばない(クエリベクトルは h5 直読み)ので GPU 不要・CPU のみ。

set -euo pipefail

PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
INDEX_DIR="outputs/0018_20260909_build_faiss_index_deblank/default"

echo "Running scripts/self_retrieval_diagnostic.py --index-dir ${INDEX_DIR} on $(hostname) ..."

if command -v apptainer &>/dev/null && [ -n "${SIF_PATH:-}" ] && [ -f "${SIF_PATH}" ]; then
    apptainer exec \
        "${SIF_PATH}" \
        bash -c "
            set -euo pipefail
            source ${PROJECT_ROOT}/.venv/bin/activate
            cd ${PROJECT_ROOT}
            python scripts/self_retrieval_diagnostic.py --index-dir ${INDEX_DIR}
        "
else
    echo "⚠️ Apptainer not found or SIF_PATH not set. Running on host system."
    source "${PROJECT_ROOT}/.venv/bin/activate"
    cd "${PROJECT_ROOT}"
    python scripts/self_retrieval_diagnostic.py --index-dir ${INDEX_DIR}
fi

echo "Done. Results written to outputs/gt_validations/self_retrieval_diagnostic_0018.csv"
