#!/bin/bash
#SBATCH --job-name=adhoc_exact_vs_approx_diagnostic
#SBATCH --partition=medium-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/adhoc_exact_vs_approx_diagnostic/%j.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/adhoc_exact_vs_approx_diagnostic/%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=2:00:00
# Step 3: scripts/exact_vs_approx_diagnostic.py を実行し、7カテゴリそれぞれで
# 近似(PQ)スコアによるランキングと、厳密(exact)スコアで再計算したランキングを
# 比較する。カテゴリごとに最大MAX_TILES_RERANKED=8タイル×候補8000件をh5から
# 読み直して厳密計算するため、Step 1/2よりI/Oコストが増える見込み。
# time-limitはStep 1/2の1時間から2時間へ余裕を持たせている
# (experiments/0003_..._query_demo / adhoc_validate_against_ground_truth.sh
# と同等のGPU/CPU/メモリ設定)。

set -euo pipefail

PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"

echo "Running scripts/exact_vs_approx_diagnostic.py on $(hostname) ..."

if command -v apptainer &>/dev/null && [ -n "${SIF_PATH:-}" ] && [ -f "${SIF_PATH}" ]; then
    apptainer exec \
        --nv \
        "${SIF_PATH}" \
        bash -c "
            set -euo pipefail
            source ${PROJECT_ROOT}/.venv/bin/activate
            export CUDA_HOME=/usr/local/cuda
            cd ${PROJECT_ROOT}
            python scripts/exact_vs_approx_diagnostic.py
        "
else
    echo "⚠️ Apptainer not found or SIF_PATH not set. Running on host system."
    source "${PROJECT_ROOT}/.venv/bin/activate"
    cd "${PROJECT_ROOT}"
    python scripts/exact_vs_approx_diagnostic.py
fi

echo "Done. Results written to outputs/gt_validation_exact_vs_approx.csv"
