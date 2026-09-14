#!/bin/bash
#SBATCH --job-name=0043_20260914_macenko_reference_patch_audit
#SBATCH --partition=medium-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0043_20260914_macenko_reference_patch_audit/%j_0043_20260914_macenko_reference_patch_audit.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0043_20260914_macenko_reference_patch_audit/%j_0043_20260914_macenko_reference_patch_audit.out
#SBATCH --signal=B:USR1@54
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=1:30:00
# CPU-only: 現行のMacenko基準パッチ(data/baseline/63958_x38976_y7616.png)が
# UNI埋め込みの重心近さという弱いプロキシで選ばれていた問題を見直す診断ジョブ。
# 背景除去済みコーパス(experiments/0018)から150スライド×5パッチ=約750枚を
# 生WSIから実際に切り出し(openslide経由、NFS I/O支配)、torchstainの
# NumpyMacenkoNormalizerで各パッチの染色ベクトル(HERef)・最大濃度(maxCRef)を
# 推定、その中央値への距離でランキングする。現行基準パッチもこの分布の中での
# 位置(典型的か外れ値か)を評価する。GPU不要・UNIエンコーダは呼ばない。
# ⚠️ リソースを変えたら --partition と --signal のマージンも手動で見直すこと。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0043_20260914_macenko_reference_patch_audit"

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
