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
# 7カテゴリ・計91枚程度の画像をembed+検索するだけの軽量ジョブ
# (experiments/0003_..._query_demo と同等の資源設定)。
# GPUはUNI推論を速くするために確保しているが、lib/query_embedding.pyは
# GPU無し(cuda利用不可)でも自動的にfloat32のCPU実行にフォールバックする。
#
# baseline_v1 は 2026-09-09 に背景除去済みインデックス(experiments/0018)へ昇格済み。
# 既定では baseline_v1 + (存在すれば)baseline_v2 / v1_macenko を回す。
#
# 挙動切替(env-var を sbatch の前に前置。--export のカンマ分割を避けるため):
#   VGT_PIPELINES=baseline_v1,v1_predeblank
#       default_pipelines() の部分集合だけ回す。背景除去 A/B を再実行するなら
#       baseline_v1(0018)と v1_predeblank(0002)だけあれば十分で、torchstain を
#       伴う v2/macenko の遅い埋め込みを省ける。
#   VGT_OUT=outputs/gt_validations/gt_validation_results_deblank_ab.csv
#       出力先。既定は outputs/gt_validation_results.csv(上書きされる)。
#       退避が必要なら別名を渡すこと。
#
#   VGT_PIPELINES=baseline_v1,v1_predeblank VGT_OUT=outputs/gt_validations/gt_validation_results_deblank_ab.csv sbatch scripts/adhoc_validate_against_ground_truth.sh

set -euo pipefail

PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"

RUN_ARGS=""
if [ -n "${VGT_PIPELINES:-}" ]; then RUN_ARGS="${RUN_ARGS} --pipelines ${VGT_PIPELINES}"; fi
if [ -n "${VGT_OUT:-}" ]; then RUN_ARGS="${RUN_ARGS} --out ${VGT_OUT}"; fi

echo "Running scripts/validate_against_ground_truth.py on $(hostname) ..."
echo "RUN_ARGS: ${RUN_ARGS}"

if command -v apptainer &>/dev/null && [ -n "${SIF_PATH:-}" ] && [ -f "${SIF_PATH}" ]; then
    apptainer exec \
        --nv \
        "${SIF_PATH}" \
        bash -c "
            set -euo pipefail
            source ${PROJECT_ROOT}/.venv/bin/activate
            export CUDA_HOME=/usr/local/cuda
            cd ${PROJECT_ROOT}
            python scripts/validate_against_ground_truth.py ${RUN_ARGS}
        "
else
    echo "⚠️ Apptainer not found or SIF_PATH not set. Running on host system."
    source "${PROJECT_ROOT}/.venv/bin/activate"
    cd "${PROJECT_ROOT}"
    python scripts/validate_against_ground_truth.py ${RUN_ARGS}
fi

echo "Done. Results written to ${VGT_OUT:-outputs/gt_validation_results.csv}"
