#!/bin/bash
#SBATCH --job-name=0031_20260911_fatty_change_macenko_visual
#SBATCH --partition=medium-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0031_20260911_fatty_change_macenko_visual/%j_0031_20260911_fatty_change_macenko_visual.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0031_20260911_fatty_change_macenko_visual/%j_0031_20260911_fatty_change_macenko_visual.out
#SBATCH --signal=B:USR1@60
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=2:00:00
# Fatty Change(atlas、コーパス内GT0枚のため定量評価不可)を baseline(0018・
# プレーン)と macenko(0012・per-tile Macenko正規化)の両方で目視診断
# (experiments/0013 の枠組み、タイルスコアヒートマップ + パッチギャラリー)。
# 対象は1所見(8図版)×2索引=2クエリセットのみとごく小さいので、0013のような
# 局所SSDステージングは行わず生h5/生WSIをNFS直読み。
#
# GPU: クエリ図版のタイル埋め込み(uni_v1 UNI エンコーダ、macenko側はper-tile
# torchstain正規化込み)に必要。
# ⚠️ リソースを変えたら --partition と --signal のマージンも手動で見直すこと。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0031_20260911_fatty_change_macenko_visual"

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
