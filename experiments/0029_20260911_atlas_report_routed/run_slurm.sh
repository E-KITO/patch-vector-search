#!/bin/bash
#SBATCH --job-name=0029_20260911_atlas_report_routed
#SBATCH --partition=large-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0029_20260911_atlas_report_routed/%j_0029_20260911_atlas_report_routed.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0029_20260911_atlas_report_routed/%j_0029_20260911_atlas_report_routed.out
#SBATCH --signal=B:USR1@144
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=4:00:00
# NNL アトラス 25 所見を baseline(0018)と所見ルーティング後(lib.finding_routing、
# 3所見のみ whiten に倒れる)の両方で検索し、所見ごとに「クエリタイル / baseline
# 上位パッチ / routed 上位パッチ」を個別 JPEG で書き出す。experiments/0020 と同じ
# 構造の新規実験(0020 自体は変更しない)。実行末尾で
# scripts/build_atlas_report_routed.py が report.html を生成する。
#
# GPU: クエリ図版のタイル埋め込み(uni_v1 UNI エンコーダ)に必要。
# ⚠️ リソース(--gres/--cpus-per-task/--mem/--time)を変えたら --partition と
#    --signal のマージンも手動で見直すこと。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0029_20260911_atlas_report_routed"

# =====================================================
# Storage(0020 と同じ方針: 索引2本 + uni_v1 h5 をノードローカル NVMe にステージ)
# =====================================================
USE_LOCAL_SSD_INPUT=0
# スイープ出力(所見ごとの JPEG)とレポートは NFS の outputs/{exp}/ に直書きする。
# スクラッチ経由(=1)にすると「report のみ再実行」で空のスクラッチを見て
# finding を取りこぼす(0020 の job 10503 と同じ落とし穴)。
USE_LOCAL_SSD_OUTPUT=0

PRE_NATIVE_COMMAND='
  set -euo pipefail
  _NDONE=$(find "${PROJECT_ROOT}/outputs/${EXP_NAME}" -name completion.json -path "*/finding__*" 2>/dev/null | wc -l)
  if [ "${_NDONE}" -ge 25 ] && [[ "${RUN_COMMAND}" != *--overwrite* ]]; then
    echo "[stage] skip — ${_NDONE} findings already complete (report-only rebuild)"
  else
    _S="${SCRATCH_DIR}/staged"
    mkdir -p "${_S}"
    echo "[stage] baseline(0018) index -> ${_S}/index_baseline"
    rsync -a "${PROJECT_ROOT}/outputs/0018_20260909_build_faiss_index_deblank/default/" "${_S}/index_baseline/"
    echo "[stage] whiten(0025) index -> ${_S}/index_whiten"
    rsync -a "${PROJECT_ROOT}/outputs/0025_20260910_whiten_index_rebuild/default/whiten_v1/" "${_S}/index_whiten/"
    echo "[stage] uni_v1 per-slide h5 features (~72G) -> ${_S}/features"
    rsync -a "${PROJECT_ROOT}/data/trident_processed/20x_224px_0px_overlap/features_uni_v1/" "${_S}/features/"
    export PVS_INDEX_BASELINE_DIR="${_S}/index_baseline"
    export PVS_INDEX_WHITEN_DIR="${_S}/index_whiten"
    export PVS_FEATURES_DIR="${_S}/features"
    echo "[stage] done: $(du -sh "${_S}" 2>/dev/null | cut -f1 || true)"
  fi
'

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"
_ATLAS_ROOT="${PROJECT_ROOT}/data/query/Nonneoplastic-Lesion-Atlas-National-Toxicology-Program_Liver"

# =====================================================
# Single run — 所見ごとの completion.json ガードで再投入は続きから走る。
# =====================================================
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml --atlas-root ${_ATLAS_ROOT} && python ${PROJECT_ROOT}/scripts/build_atlas_report_routed.py --exp-name ${EXP_NAME}"

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
