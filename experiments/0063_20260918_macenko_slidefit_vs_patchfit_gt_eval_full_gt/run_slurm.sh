#!/bin/bash
#SBATCH --job-name=0063_20260918_macenko_slidefit_vs_patchfit_gt_eval_full_gt
#SBATCH --partition=small-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0063_20260918_macenko_slidefit_vs_patchfit_gt_eval_full_gt/%j_0063_20260918_macenko_slidefit_vs_patchfit_gt_eval_full_gt.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0063_20260918_macenko_slidefit_vs_patchfit_gt_eval_full_gt/%j_0063_20260918_macenko_slidefit_vs_patchfit_gt_eval_full_gt.out
#SBATCH --signal=B:USR1@36
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=1:00:00
# GT対応7カテゴリをmacenko索引2本(パッチ単位fit vs スライド単位fit)に
# 対して検索するだけの軽量ジョブ(scripts/adhoc_validate_against_ground_truth.sh
# と同等の資源設定)。alphaスイープや目視ギャラリーは無い分、experiments/0045
# より大幅に軽い想定。GPUはUNI推論を速くするために確保しているが、
# lib/query_embedding.pyはGPU無しでも自動的にCPU実行にフォールバックする。
#
# 前提: experiments/0049(manifest)・0050(索引)が完了していること。
# 依存ジョブとして投入する場合(手動、0050完了後にjob_idを埋める):
# #SBATCH --dependency=afterok:<0050のjob_id>

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0063_20260918_macenko_slidefit_vs_patchfit_gt_eval_full_gt"

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
