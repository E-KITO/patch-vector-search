#!/bin/bash
#SBATCH --job-name=0047_20260914_context_contrast_probe
#SBATCH --partition=medium-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0047_20260914_context_contrast_probe/%j_0047_20260914_context_contrast_probe.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0047_20260914_context_contrast_probe/%j_0047_20260914_context_contrast_probe.out
#SBATCH --signal=B:USR1@72
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=2:00:00
# 候補パッチ自身の埋め込みから、同じスライド内で空間的に隣接する(3x3グリッドの
# 残り8マス)パッチ埋め込みの平均を引いた差分(contrast)ベクトルを作り、
# GT対応7所見それぞれについてplain(単一パッチ)とのLOSO AUROC比較を行う。
# 新規のUNI埋め込みは一切行わない(既存baseline索引0018のh5特徴量を読むだけ)
# ため、GPU不要(experiments/0043・0044と同種のCPU-onlyジョブ)。
#
# CPU: 18Mパッチのmanifestを一度slide_idでgroupby(近傍探索の索引作り)した後、
# 所見ごとに正例(GTスライド由来、最大300枚/スライド)・負例(6000枚)の
# h5読み込みを7所見分繰り返す。
# ⚠️ リソースを変えたら --partition と --signal のマージンも手動で見直すこと。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0047_20260914_context_contrast_probe"

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
