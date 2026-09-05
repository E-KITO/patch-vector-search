#!/bin/bash
#SBATCH --job-name=0014_20260904_build_finding_patch_set
#SBATCH --partition=medium-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0014_20260904_build_finding_patch_set/%j_0014_20260904_build_finding_patch_set.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0014_20260904_build_finding_patch_set/%j_0014_20260904_build_finding_patch_set.out
#SBATCH --signal=B:USR1@72
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=2:00:00
# 所見ごとに: NNL アトラス図版 embed + 索引検索 + 後処理 + 実解像度パッチ切り出し。
# experiments/0009 と同等の軽量クエリ。GPU は UNI 推論用(無くても CPU float32 に
# フォールバック)。openslide の実解像度クロップが一番重い(数百パッチ)。
# ⚠️ リソースを変えたら --partition と --signal のマージンも手動で見直すこと。

# 他の実験のジョブに依存させたい場合:
# #SBATCH --dependency=afterok:<job_id>

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0014_20260904_build_finding_patch_set"

# =====================================================
# Storage
# =====================================================
# index/manifest(数GB)+ exact re-rank 用の h5 一部 + 生 WSI 数枚しか読まない
# 軽量クエリなので data/ 全体のステージングは不要。NFS 直読み。
USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=1

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"

# =====================================================
# Seq run: 自己検索診断でコーパスがよく表現でき、かつ NNL アトラスの所見名との
# 対応がクリーンな2所見でトラック1を通す。experiment.py の FINDING_TO_ATLAS_DIR
# を参照。
# =====================================================

RUN_MODE="seq"
BASE_COMMAND="python ${PYTHON_PATH} --config config.yml"
GRID_ARGS=(
    "--finding"
)
GRID_VALUES=(
    "Deposit,_glycogen Increased_mitosis"
)

# GRID_VALUES はスペース区切りで seq に展開される。所見名にスペースがあると
# 壊れるので、experiment.py 側で "_" を " " に戻して受け取る(下記 argparse 参照)。

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
