#!/bin/bash
#SBATCH --job-name=0052_20260916_macenko_slidefit_vs_patchfit_visual_diagnostic
#SBATCH --partition=small-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0052_20260916_macenko_slidefit_vs_patchfit_visual_diagnostic/%j_0052_20260916_macenko_slidefit_vs_patchfit_visual_diagnostic.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0052_20260916_macenko_slidefit_vs_patchfit_visual_diagnostic/%j_0052_20260916_macenko_slidefit_vs_patchfit_visual_diagnostic.out
#SBATCH --signal=B:USR1@36
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=1:00:00
# experiments/0051のGT best_rank比較(所見によって改善/悪化/found減少が混在)を
# 受けて、README「experiments/0037」「0041」「0045」の教訓通りGT数値だけで
# 判断せず、GT対応7所見全部についてmacenko_patchfit(既存)vs macenko_slidefit
# (experiments/0050)のギャラリーを目視診断する。alphaスイープは無く各所見
# 2アームのみなので、0045の目視診断部分より大幅に軽い想定。GPUはUNI推論を
# 速くするために確保しているが無くてもCPUにフォールバックする。
#
# 前提: experiments/0050(索引)・0051(GT比較)が完了していること。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0052_20260916_macenko_slidefit_vs_patchfit_visual_diagnostic"

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
