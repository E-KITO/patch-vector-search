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
# Seq run: whole-patch テクスチャ系の未検証所見への展開(2026-09-05)。
#   Ground_glass_appearance@deliver — コーパス内4枚。自己検索 best_rank 3.5
#     (chance 250)、同群ペアなし・3化合物にまたがるのでバッチ効果の懸念が小さい。
#   Degeneration,_granular,_eosinophilic@deliver — コーパス内5枚。自己検索
#     best_rank 1.0(chance 200)で全所見中最良。ただし5枚中4枚が EXP 184
#     (gemfibrozil)、残り1枚も fenofibrate = 同じフィブラート系なので、
#     引けたものが「顆粒状好酸性変性一般」か「フィブラート系の肝細胞変化」かは
#     結果の解釈時に注意する。
#
# どちらも deliver: GT スライドが4〜5枚と少なく validate では seed が薄くなるため、
# 全 GT を seed にして未ラベルスライドから集合を作る。recall 指標は得られないので、
# **実行後に必ず scripts/random_patch_baseline.py でランダム対照を取ること**
# (hypertrophy は GT recall が良く見えて実際は選別できていなかった。job 9937)。
#
# k_candidate_patches / rerank_pool は現状維持(5000)で様子を見る。deliver は
# 候補が枯れやすい(glycogen deliver は 174 候補で 137枚止まり)ので、target 150 に
# 届かなければ次で深くする。
#
# 実行済み: 9701(glycogen@deliver + Hypertrophy@validate)、9932(Hypertrophy@validate 再実行)。
# experiment.py の SUPPORTED_FINDINGS / MODES を参照。
# =====================================================

RUN_MODE="seq"
BASE_COMMAND="python ${PYTHON_PATH} --config config.yml"
GRID_ARGS=(
    "--task"
)
GRID_VALUES=(
    "Ground_glass_appearance@deliver Degeneration,_granular,_eosinophilic@deliver"
)

# GRID_VALUES はスペース区切りで seq に展開される。所見名のスペースは "_" で書き、
# experiment.py 側で "_"→" " に戻す。"@" 以降がモード(下記 argparse 参照)。

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
