#!/bin/bash
#SBATCH --job-name=0032_20260911_build_patch_manifest_macenko_deblank
#SBATCH --partition=medium-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0032_20260911_build_patch_manifest_macenko_deblank/%j_0032_20260911_build_patch_manifest_macenko_deblank.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0032_20260911_build_patch_manifest_macenko_deblank/%j_0032_20260911_build_patch_manifest_macenko_deblank.out
#SBATCH --signal=B:USR1@72
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=2:00:00
# CPU-only: experiments/0017 と同じロジック(coords走査+学習サンプル収集+
# corpus_blankness.parquet による背景除外)を Macenko コーパス(experiments/0010)
# に適用するだけ。GPU不要。0017 の実測(medium-creator-i/8cpu/48g/2h)をそのまま踏襲。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0032_20260911_build_patch_manifest_macenko_deblank"

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
