#!/bin/bash
#SBATCH --job-name=adhoc_self_retrieval_diagnostic
#SBATCH --partition=small-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/adhoc_self_retrieval_diagnostic/%j.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/adhoc_self_retrieval_diagnostic/%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=1:00:00
# scripts/self_retrieval_diagnostic.py: コーパス内 leave-one-out 自己検索で
# uni_v1 索引の「モデル天井」を測る。UNI エンコーダは一切呼ばない
# (クエリベクトルは h5 の格納特徴量をそのまま読む)ので GPU 不要・CPU のみ。
# ~16 findings x 最大12 LOO クエリ、1クエリあたり search_top_slides_multi を
# 1500 パッチ分。実測を見てから MAX_QUERY_SLIDES_PER_FINDING / N_QUERY_PATCHES を調整。
#
# 既定は experiments/0018(背景除去済み、現行の既定索引)。出力は
# self_retrieval_diagnostic.csv。
#
# 挙動切替(env-var を sbatch の前に前置):
#   SRD_INDEX_DIR=outputs/0002_20260808_build_faiss_index/default
#       背景除去前の索引(experiments/0002)で回す。出力名に実験IDが付く
#       (self_retrieval_diagnostic_0002.csv)ので既定の記録を上書きしない。
#
#   sbatch scripts/adhoc_self_retrieval_diagnostic.sh
#   SRD_INDEX_DIR=outputs/0002_20260808_build_faiss_index/default sbatch scripts/adhoc_self_retrieval_diagnostic.sh

set -euo pipefail

PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"

RUN_ARGS=""
if [ -n "${SRD_INDEX_DIR:-}" ]; then RUN_ARGS="${RUN_ARGS} --index-dir ${SRD_INDEX_DIR}"; fi

echo "Running scripts/self_retrieval_diagnostic.py on $(hostname) ..."
echo "RUN_ARGS: ${RUN_ARGS}"

if command -v apptainer &>/dev/null && [ -n "${SIF_PATH:-}" ] && [ -f "${SIF_PATH}" ]; then
    apptainer exec \
        "${SIF_PATH}" \
        bash -c "
            set -euo pipefail
            source ${PROJECT_ROOT}/.venv/bin/activate
            cd ${PROJECT_ROOT}
            python scripts/self_retrieval_diagnostic.py ${RUN_ARGS}
        "
else
    echo "⚠️ Apptainer not found or SIF_PATH not set. Running on host system."
    source "${PROJECT_ROOT}/.venv/bin/activate"
    cd "${PROJECT_ROOT}"
    python scripts/self_retrieval_diagnostic.py ${RUN_ARGS}
fi

echo "Done. Results written under outputs/gt_validations/ (self_retrieval_diagnostic.csv, or _<expid>.csv for a non-default --index-dir)."
