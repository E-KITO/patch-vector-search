#!/bin/bash
#SBATCH --job-name=0020_20260910_query_demo_deblank_atlas_sweep
#SBATCH --partition=large-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0020_20260910_query_demo_deblank_atlas_sweep/%j_0020_20260910_query_demo_deblank_atlas_sweep.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0020_20260910_query_demo_deblank_atlas_sweep/%j_0020_20260910_query_demo_deblank_atlas_sweep.out
#SBATCH --signal=B:USR1@144
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=4:00:00
# NNL アトラス 25 所見を、背景除去済み索引(0018)と背景除去前(0002)の両方で検索し、
# 所見ごとに「クエリタイル / 0018 上位パッチ / 0002 上位パッチ」を個別 JPEG で
# 書き出す。実行末尾で build_report.py が self-contained な report.html を生成する。
#
# GPU: クエリ図版のタイル埋め込み(uni_v1 UNI エンコーダ)に必要。
# ⚠️ リソース(--gres/--cpus-per-task/--mem/--time)を変えたら --partition と
#    --signal のマージンも手動で見直すこと。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0020_20260910_query_demo_deblank_atlas_sweep"

# =====================================================
# Storage
# =====================================================
# exact re-rank(max_tiles_reranked=null)で全クエリタイル × rerank_pool=200 の
# h5 読みが支配的。experiments/0013 と同様、索引 2 本(各 1.3G)と uni_v1 h5(72G)を
# ノードローカル NVMe にステージしてから読む。data/ 全体(844G)はステージしない。
# 生 WSI はギャラリーのクロップでしか開かず(所見 × 索引 × k ≈ 500 リージョン)、
# openslide はヘッダ + 1 リージョンしか読まないので NFS 直読み。
USE_LOCAL_SSD_INPUT=0
# スイープ出力(所見ごとの JPEG ~1500枚)とレポートは NFS の outputs/{exp}/ に直書きする。
# スクラッチ経由(=1)にすると、スイープをスキップした「report のみ再実行」で
# build_atlas_report.py が空のスクラッチを見て finding を取りこぼす(job 10503 で発生)。
USE_LOCAL_SSD_OUTPUT=0

PRE_NATIVE_COMMAND='
  set -euo pipefail
  # 全 25 所見が完了済みで、かつ --overwrite でない（= report.html の作り直しだけ）なら
  # 72G のステージングは不要。--overwrite 時は全所見が再検索されるのでステージする。
  _NDONE=$(find "${PROJECT_ROOT}/outputs/${EXP_NAME}" -name completion.json -path "*/finding__*" 2>/dev/null | wc -l)
  if [ "${_NDONE}" -ge 25 ] && [[ "${RUN_COMMAND}" != *--overwrite* ]]; then
    echo "[stage] skip — ${_NDONE} findings already complete (report-only rebuild)"
  else
    _S="${SCRATCH_DIR}/staged"
    mkdir -p "${_S}"
    echo "[stage] 0018 index -> ${_S}/index_deblank"
    rsync -a "${PROJECT_ROOT}/outputs/0018_20260909_build_faiss_index_deblank/default/" "${_S}/index_deblank/"
    echo "[stage] 0002 index -> ${_S}/index_predeblank"
    rsync -a "${PROJECT_ROOT}/outputs/0002_20260808_build_faiss_index/default/" "${_S}/index_predeblank/"
    echo "[stage] uni_v1 per-slide h5 features (~72G) -> ${_S}/features"
    rsync -a "${PROJECT_ROOT}/data/trident_processed/20x_224px_0px_overlap/features_uni_v1/" "${_S}/features/"
    export PVS_INDEX_DEBLANK_DIR="${_S}/index_deblank"
    export PVS_INDEX_PREDEBLANK_DIR="${_S}/index_predeblank"
    export PVS_FEATURES_DIR="${_S}/features"
    echo "[stage] done: $(du -sh "${_S}" 2>/dev/null | cut -f1 || true)"
  fi
'

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"
_ATLAS_ROOT="${PROJECT_ROOT}/data/query/Nonneoplastic-Lesion-Atlas-National-Toxicology-Program_Liver"

# =====================================================
# Single run
#
# experiment.py が 25 所見を内部ループし、所見ごとに completion.json ガードが
# 効くので TIMEOUT しても再投入で続きから走る。スイープが正常終了したら
# 続けて scripts/build_atlas_report.py が self-contained な report.html を生成する。
# =====================================================

RUN_MODE="single"
# 所見ごとの completion.json ガードで再投入は続きから走る。対照図版除外
# (lib.atlas_figures) の反映は job 10513 (--overwrite) で完了済み。図版フォルダや
# クエリパラメータを変えたときだけ一時的に --overwrite を足すこと。
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml --atlas-root ${_ATLAS_ROOT} && python ${PROJECT_ROOT}/scripts/build_atlas_report.py --exp-name ${EXP_NAME}"

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
