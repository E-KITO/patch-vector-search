#!/bin/bash
#SBATCH --job-name=0027_20260911_atlas_outlier_diagnostic
#SBATCH --partition=medium-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0027_20260911_atlas_outlier_diagnostic/%j_0027_20260911_atlas_outlier_diagnostic.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0027_20260911_atlas_outlier_diagnostic/%j_0027_20260911_atlas_outlier_diagnostic.out
#SBATCH --signal=B:USR1@120
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=2:00:00
# experiments/0025 の whiten 索引が atlas GT を悪化させた件を、atlas 図版1枚単位に
# 分解して baseline(0018) と比較する診断(experiments/0027、scripts/atlas_per_image_diagnostic.py)。
# 索引の再構築はなし(0018/0025 の既存索引をそのまま読む)。
#
# GPU: atlas 図版(計 ~29枚)のタイル埋め込み(uni_v1 UNI エンコーダ)に必要。
# 実計算は FAISS 検索(nprobe=64, k_candidates=8000)を 29図版 x 2索引 = 58回。
# h5 exact-rerank は行わない(search_top_slides_multi は近似 FAISS 距離のみ)ので、
# 72G の per-slide h5 特徴量ステージは不要 — index.faiss/manifest/slide_meta
# (どちらも数GB)は NFS 直読みで十分。
# ⚠️ 注意: リソースを変更したら --partition と --signal のマージンも手動で見直すこと。
#          partition ごとの時間上限: small-creator-i=1h / medium-creator-i=2h /
#          large-creator-i=4h / x-large-creator-i=無制限(--time 指定は必須)。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0027_20260911_atlas_outlier_diagnostic"

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
