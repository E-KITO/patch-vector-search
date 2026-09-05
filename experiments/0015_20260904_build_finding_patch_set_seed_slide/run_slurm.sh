#!/bin/bash
#SBATCH --job-name=0015_20260904_build_finding_patch_set_seed_slide
#SBATCH --partition=large-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0015_20260904_build_finding_patch_set_seed_slide/%j_0015_20260904_build_finding_patch_set_seed_slide.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0015_20260904_build_finding_patch_set_seed_slide/%j_0015_20260904_build_finding_patch_set_seed_slide.out
#SBATCH --signal=B:USR1@144
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=4:00:00
# 所見ごとに: seed スライドのパッチ特徴量を h5 から読む + 索引検索 + 後処理 +
# 実解像度パッチ切り出し。UNI エンコーダは呼ばないので GPU 不要。
# rerank_pool=6000 で全クエリタイルを厳密再ランキングするので h5 読み込みが
# 支配的(タイル × 候補が触るスライド数だけ h5 open)。初回(rerank_pool=1000)は
# 数分だったが 6x 深いので余裕を見て 4h / large-creator-i にしている。
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
# Seq run: spatial_nms オーバーフロー修正後の再実行。
#   glycogen@deliver   — 9675 で validate 成立済み。全 GT スライドを seed に
#                        未ラベルスライドから代表パッチ集を作る(病理レビュー用)。
#   Hypertrophy@validate — 未検証。seed/hold-out で retrieval+curation が効くか測る。
# experiment.py の SUPPORTED_FINDINGS / MODES を参照。
# =====================================================

RUN_MODE="seq"
BASE_COMMAND="python ${PYTHON_PATH} --config config.yml"
GRID_ARGS=(
    "--task"
)
GRID_VALUES=(
    "Deposit,_glycogen@deliver Hypertrophy@validate"
)

# GRID_VALUES はスペース区切りで seq に展開される。所見名のスペースは "_" で書き、
# experiment.py 側で "_"→" " に戻す。"@" 以降がモード(下記 argparse 参照)。

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
