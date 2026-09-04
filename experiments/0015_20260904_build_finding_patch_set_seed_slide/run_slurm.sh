#!/bin/bash
#SBATCH --job-name=0015_20260904_build_finding_patch_set_seed_slide
#SBATCH --partition=medium-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0015_20260904_build_finding_patch_set_seed_slide/%j_0015_20260904_build_finding_patch_set_seed_slide.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0015_20260904_build_finding_patch_set_seed_slide/%j_0015_20260904_build_finding_patch_set_seed_slide.out
#SBATCH --signal=B:USR1@72
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=2:00:00
# 所見ごとに: seed スライドのパッチ特徴量を h5 から読む + 索引検索 + 後処理 +
# 実解像度パッチ切り出し。UNI エンコーダは呼ばないので GPU 不要。openslide の
# 実解像度クロップが一番重い(所見あたり ~150 パッチ)。
# ⚠️ リソースを変えたら --partition と --signal のマージンも手動で見直すこと。

# 他の実験のジョブに依存させたい場合:
# #SBATCH --dependency=afterok:<job_id>

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0015_20260904_build_finding_patch_set_seed_slide"

# =====================================================
# Storage
# =====================================================
# index/manifest(数GB)+ exact re-rank 用の h5 一部 + 生 WSI 数枚しか読まない
# 軽量クエリなので data/ 全体のステージングは不要。NFS 直読み。
USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=1

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"

# =====================================================
# Seq run: 0014 と同じ2所見(自己検索診断でコーパスがよく表現できる)。
# 0015 は seed を NNL アトラス図版ではなくその所見の GT スライドにする。
# experiment.py の SUPPORTED_FINDINGS を参照。
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
