#!/bin/bash
#SBATCH --job-name=0013_20260904_query_demo_macenko_per_tile
#SBATCH --partition=x-large-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0013_20260904_query_demo_macenko_per_tile/%j_0013_20260904_query_demo_macenko_per_tile.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0013_20260904_query_demo_macenko_per_tile/%j_0013_20260904_query_demo_macenko_per_tile.out
#SBATCH --signal=B:USR1@180
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=8:00:00
# experiments/0009 と同じ可視化を uni_v1 Macenko コーパス(0012)+ per-tile 正規化
# クエリで回す。現在は NNL アトラスの全クエリ画像(94 枚)を 1 枚ずつ独立クエリと
# して sweep する設定(--atlas-root --per-image)。experiment.py が内部ループし、
# 画像ごとに completed ガードが効くので TIMEOUT しても再投入で続きから走る。
#
# ⚠️ 注意: partition ごとの時間上限 — small-creator-i=1h / medium-creator-i=2h /
#          large-creator-i=4h / x-large-creator-i=無制限(--time 指定は必須)。
#          リソース(--gres/--cpus-per-task/--mem/--time)を変更したら
#          --partition と --signal のマージンも手動で見直すこと。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0013_20260904_query_demo_macenko_per_tile"

# =====================================================
# Storage — filesrv01 の HDD 障害調査中につき、計算中の高頻度 I/O は
# NFS ではなく計算ノードのローカル NVMe(/scratch)で行う。
#
# 高頻度 I/O は exact re-rank の h5 アクセス(max_tiles_reranked=null で全タイルを
# rerank_pool=200 で厳密採点 → クエリ 1 セットあたり数万回の小さな h5 読み)。
# USE_LOCAL_SSD_INPUT=1 は data/(844G)全体を rsync してしまい過大なので使わず、
# PRE_NATIVE_COMMAND で Macenko index (1.3G) / h5 (73G) / サムネイル (63M) だけを
# ${SCRATCH_DIR}/staged/ にステージし、PVS_INDEX_DIR / PVS_FEATURES_DIR /
# PVS_THUMBNAILS_DIR で experiment.py に読ませる。
#
# 生 WSI(~634G)は図版生成でしか読まれない。図版が開くのは各クエリの上位
# top_n_slides_to_plot スライドだけなので、その .svs をノードローカルへステージ
# するのは experiment.py の stage_wsi_hybrid() に任せる(全スライドを symlink +
# 上位スライドを実コピー)。PRE_NATIVE の bash(set -euo pipefail + ERR trap)で
# glob 不一致がジョブごと落とす事故が続いたため(jobs 10437/10438)、そちらへ移した。
# =====================================================

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=1

PRE_NATIVE_COMMAND='
  set -euo pipefail
  _S="${SCRATCH_DIR}/staged"
  mkdir -p "${_S}"
  echo "[stage] FAISS index (Macenko v1) -> ${_S}/index"
  rsync -a "${PROJECT_ROOT}/outputs/0012_20260901_build_faiss_index_macenko_v1/default/" "${_S}/index/"
  echo "[stage] Macenko per-slide h5 features (~73G) -> ${_S}/features"
  rsync -a "${PROJECT_ROOT}/data/trident_processed_uni_v1_macenko/20x_224px_0px_overlap/features_uni_v1_macenko/" "${_S}/features/"
  echo "[stage] Macenko thumbnails -> ${_S}/thumbnails"
  rsync -a "${PROJECT_ROOT}/data/trident_processed_uni_v1_macenko/thumbnails/" "${_S}/thumbnails/"
  export PVS_INDEX_DIR="${_S}/index"
  export PVS_FEATURES_DIR="${_S}/features"
  export PVS_THUMBNAILS_DIR="${_S}/thumbnails"
  echo "[stage] done, local NVMe used: $(du -sh "${_S}" 2>/dev/null | cut -f1 || true)"
'

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"
_ATLAS_ROOT="${PROJECT_ROOT}/data/query/Nonneoplastic-Lesion-Atlas-National-Toxicology-Program_Liver"

# =====================================================
# Single run
#
# アトラス全 94 画像を 1 枚ずつ図版付きで回す。2 パス運用:
#   パス1 (初回): 下の --no-galleries 版で回して top_slides.csv を全画像ぶん出す
#                 (図版が要る上位スライドの特定に必要。~30分)
#   パス2 (本番): 下の --overwrite 版。PRE_NATIVE がパス1の top_slides.csv から
#                 上位スライドの WSI を NVMe へステージ → 図版も NVMe で生成
# パス1 を省くと初回は PVS_RAW_SLIDE_DIR 未設定で図版が NFS 直読みになる。
# =====================================================

RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml --atlas-root ${_ATLAS_ROOT} --per-image --overwrite"

# --- パス1: 図版なし(top_slides.csv だけ先に出す) ---
# RUN_COMMAND="python ${PYTHON_PATH} --config config.yml --atlas-root ${_ATLAS_ROOT} --per-image --no-galleries"

# --- 所見フォルダ集約版(25 クエリ)。--per-image を外すだけ ---
# RUN_COMMAND="python ${PYTHON_PATH} --config config.yml --atlas-root ${_ATLAS_ROOT} --overwrite"

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
