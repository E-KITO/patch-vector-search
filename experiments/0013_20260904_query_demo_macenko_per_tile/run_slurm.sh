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
#SBATCH --time=5:00:00
# experiments/0009 と同じ可視化を uni_v1 Macenko コーパス(0012)+ per-tile 正規化
# クエリで回す。現在は NNL アトラスの全所見フォルダ(25)を 1 ジョブで sweep する設定
# (--atlas-root)。experiment.py が各サブフォルダを 1 クエリとして内部ループし、
# 所見ごとに completed ガードが効くので、TIMEOUT しても再投入で続きから走る。
#
# ⚠️ 注意: partition ごとの時間上限 — small-creator-i=1h / medium-creator-i=2h /
#          large-creator-i=4h / x-large-creator-i=無制限(--time 指定は必須)。
#          リソース(--gres/--cpus-per-task/--mem/--time)を変更したら
#          --partition と --signal のマージンも手動で見直すこと。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0013_20260904_query_demo_macenko_per_tile"

# =====================================================
# Storage — filesrv01 の HDD 障害調査中につき、計算中の高頻度 I/O は
# NFS(/tank)ではなく計算ノードのローカル NVMe(/scratch)で行う。
#
# このジョブで NFS 直読みすると問題になるのは exact re-rank の h5 アクセス:
# max_tiles_reranked=null で全タイルを rerank_pool=200 で厳密採点するため、
# クエリ 1 セットあたり数万回の小さな h5 スライス読み込みが発生する
# (Dataset 高頻度アクセスの典型)。
#
# USE_LOCAL_SSD_INPUT=1 は data/(844G)全体を rsync してしまい過大なので使わず、
# PRE_NATIVE_COMMAND で必要なものだけ(Macenko index 1.3G + Macenko h5 73G +
# サムネイル 63M)を ${SCRATCH_DIR}/staged/ にステージし、experiment.py に
# PVS_INDEX_DIR / PVS_FEATURES_DIR / PVS_THUMBNAILS_DIR で読ませる。
# 生 WSI(data/moo_collected_tggate_wsi、~634G)はステージ対象外 — その唯一の
# 読み手であるパッチギャラリー生成を --no-galleries で切る。
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
  # Galleries are the only raw-WSI reader, and only for the top top_n_slides_to_plot
  # slides per finding. A prior --no-galleries sweep already wrote top_slides.csv
  # for every finding, so stage exactly those .svs (union ~34 slides / ~20G) and
  # the galleries run entirely off local NVMe. If no prior CSVs exist the var is
  # left unset and experiment.py falls back to reading WSI from NFS.
  _ids=$(for f in "${PROJECT_ROOT}"/outputs/0013_20260904_query_demo_macenko_per_tile/query__*Nonneoplastic_Lesion_Atlas__pertilenorm/top_slides.csv; do
           [ -f "${f}" ] && tail -n +2 "${f}" | head -3 | cut -d, -f1
         done | sort -u)
  if [ -n "${_ids}" ]; then
    mkdir -p "${_S}/raw_wsi"
    _n=0
    for _sid in ${_ids}; do
      _src="${PROJECT_ROOT}/data/moo_collected_tggate_wsi/raw_wsi/${_sid}.svs"
      if [ -f "${_src}" ]; then rsync -a "${_src}" "${_S}/raw_wsi/" && _n=$((_n+1)); fi
    done
    export PVS_RAW_SLIDE_DIR="${_S}/raw_wsi"
    echo "[stage] ${_n} top-slide WSI -> ${_S}/raw_wsi ($(du -sh "${_S}/raw_wsi" 2>/dev/null | cut -f1 || true))"
  else
    echo "[stage] no prior top_slides.csv -> galleries would read WSI from NFS"
  fi
  echo "[stage] done: $(du -sh "${_S}" 2>/dev/null | cut -f1 || true)"
'

# =====================================================
# python path
# =====================================================

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"

# =====================================================
# Single run — NNL アトラス全所見フォルダを sweep
# (experiment.py が --atlas-root のサブフォルダを内部ループ)
# =====================================================

RUN_MODE="single"
# 図版(patch_gallery)込みで全所見を回す。上位スライドの WSI は PRE_NATIVE_COMMAND で
# NVMe にステージ済みなので NFS 負荷は最初の rsync(~20G, 一括シーケンシャル)のみ。
# --overwrite: 直前の --no-galleries sweep で completed になった 25 run_dir を
# 上書き再実行して patch_gallery を追加する(付けないと全スキップされる)。
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml --atlas-root ${PROJECT_ROOT}/data/query/Nonneoplastic-Lesion-Atlas-National-Toxicology-Program_Liver --overwrite"

# 図版なし・より軽い版(初回 sweep 用):
# RUN_COMMAND="python ${PYTHON_PATH} --config config.yml --atlas-root ${PROJECT_ROOT}/data/query/Nonneoplastic-Lesion-Atlas-National-Toxicology-Program_Liver --no-galleries"

# =====================================================
# 過去の設定: 0009 と同じ 5 枚 + Necrosis/Kupffer 図版 + Necrosis 手動クロップ、
# および query_001..003 の単発検証。戻すときは RUN_MODE を "seq" にして下を有効化。
# =====================================================
# RUN_MODE="seq"
# BASE_COMMAND="python ${PYTHON_PATH} --config config.yml"
# GRID_ARGS=(
#     "--image"
# )
# GRID_VALUES=(
#     "${PROJECT_ROOT}/data/query/query_001.png ${PROJECT_ROOT}/data/query/query_002.png ${PROJECT_ROOT}/data/query/query_003.png"
# )

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
