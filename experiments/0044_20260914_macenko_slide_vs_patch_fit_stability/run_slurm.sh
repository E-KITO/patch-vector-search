#!/bin/bash
#SBATCH --job-name=0044_20260914_macenko_slide_vs_patch_fit_stability
#SBATCH --partition=medium-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0044_20260914_macenko_slide_vs_patch_fit_stability/%j_0044_20260914_macenko_slide_vs_patch_fit_stability.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0044_20260914_macenko_slide_vs_patch_fit_stability/%j_0044_20260914_macenko_slide_vs_patch_fit_stability.out
#SBATCH --signal=B:USR1@54
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=1:30:00
# CPU-only: ユーザー提案(スライド/図版全体を1回でMacenko正規化してから
# タイル分割する案)の事前検証。JPGサムネイルは色処理経路が現行パイプライン
# (.svsから直接パッチ切り出し)と異なる交絡を持ち込むため使わず、同じ.svsから
# lib.raw_patch.crop_patchで切り出した複数パッチを連結して1回でfitする
# 「スライド単位」方式と、パッチごとに個別fitする「現行方式」を比較する。
# 30スライド×20パッチ=約600パッチ。GPU不要・UNIエンコーダは呼ばない・
# wsi_preprocessも不要。
# ⚠️ リソースを変えたら --partition と --signal のマージンも手動で見直すこと。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0044_20260914_macenko_slide_vs_patch_fit_stability"

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
