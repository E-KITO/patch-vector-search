#!/bin/bash
#SBATCH --job-name=0009_20260820_query_demo_patch_gallery_hit_threshold_tile_score_heatmap
#SBATCH --partition=small-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0009_20260820_query_demo_patch_gallery_hit_threshold_tile_score_heatmap/%j_0009_20260820_query_demo_patch_gallery_hit_threshold_tile_score_heatmap.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0009_20260820_query_demo_patch_gallery_hit_threshold_tile_score_heatmap/%j_0009_20260820_query_demo_patch_gallery_hit_threshold_tile_score_heatmap.out
#SBATCH --signal=B:USR1@36
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=1:00:00
# 1枚の画像embed+検索だけの軽量ジョブ。GPUはUNI推論を速くするために残しているが、
# lib/query_embedding.pyはGPU無し(cuda利用不可)でも自動的にfloat32のCPU実行にフォールバックする。

# 他の実験のジョブに依存させたい場合、有効化してjob_idを埋める
# （job_idは outputs/{依存先exp}/latest_job_id.txt を参照。投入のたびに
#  変わりうる値なので、都度手動で書き換えること）:
# #SBATCH --dependency=afterok:<job_id>

# Array run にする場合、上の3行の --output/--error/この直後の --array を
# 以下の2行に置き換える（%j→%A_%a、--array=0-N を追加。Nの決め方は下記参照）:
# #SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0009_20260820_query_demo_patch_gallery_hit_threshold_tile_score_heatmap/%A_%a_0009_20260820_query_demo_patch_gallery_hit_threshold_tile_score_heatmap.out
# #SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0009_20260820_query_demo_patch_gallery_hit_threshold_tile_score_heatmap/%A_%a_0009_20260820_query_demo_patch_gallery_hit_threshold_tile_score_heatmap.out
# #SBATCH --array=0-N
#
# ⚠️ 注意: リソース(--gres/--cpus-per-task/--mem/--time)を変更したら、
#          --partition と --signal のマージンも合わせて手動で見直すこと
#          （make create_exp 実行時に一度だけ計算されたもので、自動追従しない）。
# ⚠️ 注意: シェル上での for/while ループによる複数組み合わせ実行は推奨しない。
#          下記の Array run / Seq run の使用を推奨。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0009_20260820_query_demo_patch_gallery_hit_threshold_tile_score_heatmap"

# =====================================================
# Storage
# /workspace はNFS（遅い）、/scratch はノード付属のm.2 SSD（速い・ジョブ終了時に
# 自動削除）。デフォルトで有効。NFS越しに直接読み書きしたい場合のみ0にする
# （例: 出力を実行中にリアルタイムで/workspace側から監視したい等）。
# =====================================================

# このジョブは全998ファイル(~78GB)ではなく、index/manifest(数GB)+exact re-rank用に
# 開くごく一部のh5+サムネイル数枚しか読まない軽量クエリなので、毎回data/全体を
# /scratchへステージングするのは無駄。NFS直読みにする。
USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=1

# =====================================================
# python path
# =====================================================

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"

# =====================================================
# Single run（デフォルト）
# =====================================================

# RUN_MODE="single"
# ⚠️ 投入前に --image を実際のクエリ画像パスへ書き換えること
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml --image ${PROJECT_ROOT}/data/query/Nonneoplastic-Lesion-Atlas-National-Toxicology-Program_Liver/Liver - Fatty Change - Nonneoplastic Lesion Atlas/imgi_9_figure-004-a54647_large.jpg"

# =====================================================
# Array run にしたい場合
#
# 1. 上の RUN_MODE="single" と RUN_COMMAND=... をコメントアウトする
# 2. 下のブロックを有効化する
# 3. ファイル先頭の --output/--error/--array の3行を%A_%a版に切り替える
#    （Nは GRID_VALUES の組み合わせ数-1。make preflight が一致を検証する）
#
# GRID_ARGS[i] と GRID_VALUES[i] が対応し、直積が CONFIGS として展開される。
# 例:
#   GRID_ARGS=("--model" "--dataset")
#   GRID_VALUES=("bert roberta" "pubmed pmc")
#   → --model bert --dataset pubmed / --model bert --dataset pmc / ...
# =====================================================

# RUN_MODE="array"
# BASE_COMMAND="python ${PYTHON_PATH}"
# GRID_ARGS=(
#     "--model"
#     "--dataset"
# )
# GRID_VALUES=(
#     "google/gemma-4-31b-it meta-llama/Llama-3-8b-it"
#     "BC5CDR BIORED"
# )

# =====================================================
# Seq run にしたい場合（1ジョブ内でGRIDを順次実行）
#
# 上と同様に RUN_MODE="seq" にし、BASE_COMMAND/GRID_ARGS/GRID_VALUES を設定する。
# こちらは #SBATCH --array は不要（1ジョブでループするため）。
# =====================================================

# RUN_MODE="seq"
# BASE_COMMAND="python ${PYTHON_PATH} --config config.yml"
# GRID_ARGS=(
#     "--image"
# )
# GRID_VALUES=(
#     "${PROJECT_ROOT}/data/query/necrosis_imgi11_a30016.jpg ${PROJECT_ROOT}/data/query/hematopoiesis_imgi6_a71206.jpg ${PROJECT_ROOT}/data/query/hypertrophy_imgi10_a53058.jpg"
# )

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
