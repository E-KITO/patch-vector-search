#!/bin/bash
#SBATCH --job-name=adhoc_validate_against_ground_truth
#SBATCH --partition=small-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/adhoc_validate_against_ground_truth/%j.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/adhoc_validate_against_ground_truth/%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=1:00:00
# 7カテゴリ・計94枚程度の画像をembed+検索するだけの軽量ジョブ
# (experiments/0003_..._query_demo と同等の資源設定)。
# GPUはUNI推論を速くするために確保しているが、lib/query_embedding.pyは
# GPU無し(cuda利用不可)でも自動的にfloat32のCPU実行にフォールバックする。

set -euo pipefail

PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"

echo "Running scripts/validate_against_ground_truth.py on $(hostname) ..."

if command -v apptainer &>/dev/null && [ -n "${SIF_PATH:-}" ] && [ -f "${SIF_PATH}" ]; then
    apptainer exec \
        --nv \
        "${SIF_PATH}" \
        bash -c "
            set -euo pipefail
            source ${PROJECT_ROOT}/.venv/bin/activate
            export CUDA_HOME=/usr/local/cuda
            cd ${PROJECT_ROOT}
            python scripts/validate_against_ground_truth.py
        "
else
    echo "⚠️ Apptainer not found or SIF_PATH not set. Running on host system."
    source "${PROJECT_ROOT}/.venv/bin/activate"
    cd "${PROJECT_ROOT}"
    python scripts/validate_against_ground_truth.py
fi

echo "Done. Results written to outputs/gt_validation_results.csv"
