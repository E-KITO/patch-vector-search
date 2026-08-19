#!/bin/bash
#SBATCH --job-name=adhoc_nprobe_sweep
#SBATCH --partition=small-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/adhoc_nprobe_sweep/%j.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/adhoc_nprobe_sweep/%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=1:00:00
# Step 2: scripts/nprobe_sweep.py を実行し、nprobe=64/256/1024/4096(=nlist)で
# uni_v1のGT比較(7カテゴリ)を繰り返し、outputs/gt_validation_nprobe_sweep.csv
# へまとめて書き出す。1回あたりの実行時間はStep 1(nprobe=64のみ)の実測で約27秒
# だったので、4パターン回しても数分程度の見込み
# (experiments/0003_..._query_demo / adhoc_validate_against_ground_truth.sh
# と同等の資源設定)。

set -euo pipefail

PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"

echo "Running scripts/nprobe_sweep.py on $(hostname) ..."

if command -v apptainer &>/dev/null && [ -n "${SIF_PATH:-}" ] && [ -f "${SIF_PATH}" ]; then
    apptainer exec \
        --nv \
        "${SIF_PATH}" \
        bash -c "
            set -euo pipefail
            source ${PROJECT_ROOT}/.venv/bin/activate
            export CUDA_HOME=/usr/local/cuda
            cd ${PROJECT_ROOT}
            python scripts/nprobe_sweep.py
        "
else
    echo "⚠️ Apptainer not found or SIF_PATH not set. Running on host system."
    source "${PROJECT_ROOT}/.venv/bin/activate"
    cd "${PROJECT_ROOT}"
    python scripts/nprobe_sweep.py
fi

echo "Done. Results written to outputs/gt_validation_nprobe_sweep.csv"
