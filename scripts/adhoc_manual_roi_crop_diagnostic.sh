#!/bin/bash
#SBATCH --job-name=adhoc_manual_roi_crop_diagnostic
#SBATCH --partition=small-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/adhoc_manual_roi_crop_diagnostic/%j.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/adhoc_manual_roi_crop_diagnostic/%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=1:00:00
# 7カテゴリ・計24枚の画像をembed+検索するだけの軽量ジョブ(手動クロップにより
# タイル数はbaseline_v1と同等かそれ以下)。adhoc_validate_against_ground_truth.sh
# と同等の資源設定。

set -euo pipefail

PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"

echo "Running scripts/manual_roi_crop_diagnostic.py on $(hostname) ..."

if command -v apptainer &>/dev/null && [ -n "${SIF_PATH:-}" ] && [ -f "${SIF_PATH}" ]; then
    apptainer exec \
        --nv \
        "${SIF_PATH}" \
        bash -c "
            set -euo pipefail
            source ${PROJECT_ROOT}/.venv/bin/activate
            export CUDA_HOME=/usr/local/cuda
            cd ${PROJECT_ROOT}
            python scripts/manual_roi_crop_diagnostic.py
        "
else
    echo "⚠️ Apptainer not found or SIF_PATH not set. Running on host system."
    source "${PROJECT_ROOT}/.venv/bin/activate"
    cd "${PROJECT_ROOT}"
    python scripts/manual_roi_crop_diagnostic.py
fi

echo "Done. Results written to outputs/gt_validation_manual_roi_crop.csv"
