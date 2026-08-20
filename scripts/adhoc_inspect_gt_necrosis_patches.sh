#!/bin/bash
#SBATCH --job-name=adhoc_inspect_gt_necrosis_patches
#SBATCH --partition=small-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/adhoc_inspect_gt_necrosis_patches/%j.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/adhoc_inspect_gt_necrosis_patches/%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=1:00:00
# GT壊死4スライド(41484, 58720, 58787, 59195)から各16枚、計64枚の実解像度
# パッチをopenslideで切り出して並べるだけの軽量ジョブ(GPU/UNI/FAISS不使用)。
# 他のadhoc_*.shと資源設定を揃えるため --gres=gpu:1 は付けているが実際には
# 使わない。他のadhoc_*.shと同様、投入前に依存先の出力
# (outputs/0002_20260808_build_faiss_index/default/{manifest,slide_meta}.parquet)
# が存在することを前提とする。

set -euo pipefail

PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"

echo "Running scripts/inspect_gt_necrosis_patches.py on $(hostname) ..."

if command -v apptainer &>/dev/null && [ -n "${SIF_PATH:-}" ] && [ -f "${SIF_PATH}" ]; then
    apptainer exec \
        --nv \
        "${SIF_PATH}" \
        bash -c "
            set -euo pipefail
            source ${PROJECT_ROOT}/.venv/bin/activate
            export CUDA_HOME=/usr/local/cuda
            cd ${PROJECT_ROOT}
            python scripts/inspect_gt_necrosis_patches.py
        "
else
    echo "⚠️ Apptainer not found or SIF_PATH not set. Running on host system."
    source "${PROJECT_ROOT}/.venv/bin/activate"
    cd "${PROJECT_ROOT}"
    python scripts/inspect_gt_necrosis_patches.py
fi

echo "Done. Results written to outputs/gt_necrosis_patch_gallery/{slide_id}.png"
