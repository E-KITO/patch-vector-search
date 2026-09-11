#!/bin/bash
#SBATCH --job-name=0030_20260911_atlas_visual_diagnostic_routed
#SBATCH --partition=medium-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0030_20260911_atlas_visual_diagnostic_routed/%j_0030_20260911_atlas_visual_diagnostic_routed.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0030_20260911_atlas_visual_diagnostic_routed/%j_0030_20260911_atlas_visual_diagnostic_routed.out
#SBATCH --signal=B:USR1@60
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=2:00:00
# experiments/0013(タイルスコアヒートマップ + 実解像度パッチギャラリー)の枠組みを
# whiten ルーティング先3所見(Hypertrophy / Kupffer Cell Hyperplasia / Cytoplasmic
# Inclusions、計10枚の atlas 図版)に baseline(0018)/ routed(whiten、0025)の両方で
# 適用する。0013/0025-0029 の他ジョブと比べて対象がごく小さい(3所見×2索引=6クエリ
# セット、ギャラリーで開く生WSIも最大18スライド)ので、生WSIは0016と同様NFS直読み
# (stage_wsi_hybrid 相当の複雑なステージングは不要)。
#
# GPU: クエリ図版のタイル埋め込み(uni_v1 UNI エンコーダ)に必要。
# ⚠️ リソースを変えたら --partition と --signal のマージンも手動で見直すこと。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0030_20260911_atlas_visual_diagnostic_routed"

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
