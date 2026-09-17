#!/bin/bash
#SBATCH --job-name=0049_20260916_build_patch_manifest_macenko_slidefit
#SBATCH --partition=medium-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0049_20260916_build_patch_manifest_macenko_slidefit/%j_0049_20260916_build_patch_manifest_macenko_slidefit.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0049_20260916_build_patch_manifest_macenko_slidefit/%j_0049_20260916_build_patch_manifest_macenko_slidefit.out
#SBATCH --signal=B:USR1@72
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=2:00:00
# CPU-only: experiments/0032 と全く同じロジック(coords走査+学習サンプル収集+
# corpus_blankness.parquet による背景除外)を、wsi_preprocess側で新規再構築した
# スライド単位fitのMacenkoコーパスに適用するだけ。GPU不要。0032の実測
# (medium-creator-i/8cpu/48g/2h)をそのまま踏襲。
#
# 前提: wsi_preprocess側のアレイジョブ(run_uni_v1_macenko_slidefit_array.sh)+
# finalizeジョブが完了し、features_uni_v1_macenko_slidefit配下に1000/1000枚の
# h5が揃っていること(2026-09-16時点ではまだ997/1000、欠落3枚の再処理待ち —
# README「wsi_preprocess連携」参照)。
#
# 依存ジョブとして投入する場合(手動):
# #SBATCH --dependency=afterok:<wsi_preprocess finalize再投入ジョブのjob_id>

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0049_20260916_build_patch_manifest_macenko_slidefit"

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
