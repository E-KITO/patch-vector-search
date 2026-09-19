#!/bin/bash
#SBATCH --job-name=0054_20260917_jpeg_shift_correction_gt_sweep
#SBATCH --partition=large-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0054_20260917_jpeg_shift_correction_gt_sweep/%j_0054_20260917_jpeg_shift_correction_gt_sweep.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0054_20260917_jpeg_shift_correction_gt_sweep/%j_0054_20260917_jpeg_shift_correction_gt_sweep.out
#SBATCH --signal=B:USR1@144
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=4:00:00
# experiments/0053で切り分けたJPEG圧縮由来のドメインギャップ成分から、
# experiments/0039/0041のdomain_shiftより「純粋」な補正ベクトル(jpeg_shift)を
# 作り、GT対応7所見でfinding_routing実ルーティング空間(plain/whiten/macenko)
# のalphaスイープ(experiments/0045と同じ枠組み)で効果を検証する。加えて
# domain_shift-jpeg_shiftの残差ベクトルも粗いグリッドで検証する。
#
# GPU: 200プローブパッチ×[生+JPEG4品質程度]をplain・macenko両空間で埋め込み
# (uni_v1 UNIエンコーダ)。時間の大半はexperiments/0045と同様、alphaスイープ
# (jpegshift 11点 + residual 5点)×3空間のFAISS exact rerank(NFS h5ランダム
# アクセス)。
# ⚠️ リソースを変えたら --partition と --signal のマージンも手動で見直すこと。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0054_20260917_jpeg_shift_correction_gt_sweep"

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
