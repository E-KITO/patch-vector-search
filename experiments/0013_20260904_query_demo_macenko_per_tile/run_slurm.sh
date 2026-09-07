#!/bin/bash
#SBATCH --job-name=0013_20260904_query_demo_macenko_per_tile
#SBATCH --partition=medium-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0013_20260904_query_demo_macenko_per_tile/%j_0013_20260904_query_demo_macenko_per_tile.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0013_20260904_query_demo_macenko_per_tile/%j_0013_20260904_query_demo_macenko_per_tile.out
#SBATCH --signal=B:USR1@72
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=2:00:00
# experiments/0009 と同じ可視化を uni_v1 Macenko コーパス(0012)+ per-tile 正規化
# クエリで回す軽量ジョブ。7枚を seq で順次処理。GPU は UNI 推論の高速化用で、
# lib/query_embedding.py は GPU 無しでも float32 の CPU 実行にフォールバックする。

# 他の実験のジョブに依存させたい場合、有効化して job_id を埋める
# （job_id は outputs/{依存先exp}/latest_job_id.txt を参照）:
# #SBATCH --dependency=afterok:<job_id>

# Array run にする場合、上の3行の --output/--error/この直後の --array を
# 以下の2行に置き換える（%j→%A_%a、--array=0-N を追加）:
# #SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0013_20260904_query_demo_macenko_per_tile/%A_%a_0013_20260904_query_demo_macenko_per_tile.out
# #SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0013_20260904_query_demo_macenko_per_tile/%A_%a_0013_20260904_query_demo_macenko_per_tile.out
# #SBATCH --array=0-N
#
# ⚠️ 注意: リソース(--gres/--cpus-per-task/--mem/--time)を変更したら、
#          --partition と --signal のマージンも合わせて手動で見直すこと。
# ⚠️ 注意: シェル上での for/while ループによる複数組み合わせ実行は推奨しない。
#          下記の Array run / Seq run の使用を推奨。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0013_20260904_query_demo_macenko_per_tile"

# =====================================================
# Storage
# =====================================================

# index/manifest(数GB)+ exact re-rank 用に開くごく一部の h5 + サムネイル数枚しか
# 読まない軽量クエリなので、data/ 全体を /scratch にステージングするのは無駄。
# NFS 直読みにする。
USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=1

# =====================================================
# python path
# =====================================================

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"

# =====================================================
# Seq run（1ジョブ内で GRID を順次実行）
#
# experiments/0009 と同じクエリ5枚に加え、GT 比較(scripts/validate_against_
# ground_truth.py の v1_macenko)で大幅悪化した Necrosis・Kupffer の図版と、
# タイル選択バイアス調査で使った Necrosis の手動クロップも入れて、悪化が可視化
# でどう見えるかを確認する。探索パラメータ(config.yml)は 0009 と完全一致に
# してあるので、0009 の出力(baseline_v1・plain query)と直接横並び比較できる。
# =====================================================

RUN_MODE="seq"
BASE_COMMAND="python ${PYTHON_PATH} --config config.yml"
GRID_ARGS=(
    "--image"
)
GRID_VALUES=(
    "${PROJECT_ROOT}/data/query/query_001.png ${PROJECT_ROOT}/data/query/query_002.png ${PROJECT_ROOT}/data/query/query_003.png"
)

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
