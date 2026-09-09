#!/bin/bash
#SBATCH --job-name=adhoc_measure_corpus_blankness
#SBATCH --partition=x-large-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/adhoc_measure_corpus_blankness/%j.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/adhoc_measure_corpus_blankness/%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=8:00:00
# scripts/measure_corpus_blankness.py: 既定コーパス(experiments/0002)の全 ~18.4M
# パッチを raw WSI から切り出し、明るさ・彩度(sat_frac)を測定して
# outputs/measure_corpus_blankness/corpus_blankness.parquet に落とす。
# 背景パッチをコーパスから除外する準備(段階1)。閾値の決定と manifest / FAISS 索引の
# 再構築は experiments/0017 / 0018。
#
# 索引もエンコーダも使わない(GPU 不要)。支配的コストは openslide の level-0 領域読み
# ~18.4M 回(NFS 上の 1000 SVS)。スライド単位の multiprocessing で per-slide parquet を
# 書くので再開可能: タイムアウトしても同じジョブを再投入すれば残りのスライドだけ処理する。
#
# .svs を /scratch にステージしてから開くオプション(MCB_STAGE=1)もあるが、job 10485
# vs 10486 のスモークで **NFS 直読みの方が ~25% 速かった**(コピーのオーバーヘッドが
# 回収できない)ため既定は無効。filesrv02 の状態が悪化してフル run が遅い場合の保険。
# 出力(per-slide parquet / merged / _progress.json)は再開と `cat` のため常に NFS。
#
# 進捗: ジョブログに tqdm バー + 25スライドごとのチェックポイント行。
#       さらに outputs/measure_corpus_blankness/_progress.json を毎スライド更新するので
#       `cat outputs/measure_corpus_blankness/_progress.json` でいつでも状況が見える。
#
# 挙動は --export で渡す環境変数で切り替える(スクリプトを編集しなくてよい):
#   MCB_LIMIT=N     最初の N スライドだけ(スモークテスト)
#   MCB_WORKERS=N   ワーカー数(既定: SLURM_CPUS_PER_TASK、無ければ 16)
#   MCB_STAGE=1     .svs を /scratch にステージしてから開く(既定は無効、上記参照)
#   MCB_OVERWRITE=1 既に per-slide parquet があるスライドも測り直す
#
#   フル投入:      sbatch scripts/adhoc_measure_corpus_blankness.sh
#   スモーク:      sbatch --export=ALL,MCB_LIMIT=5,MCB_WORKERS=8 scripts/adhoc_measure_corpus_blankness.sh

set -euo pipefail

PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"

echo "Running scripts/measure_corpus_blankness.py on $(hostname) ..."

RUN_ARGS="--workers ${MCB_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"
if [ "${MCB_STAGE:-}" = "1" ]; then
    STAGE_DIR="/scratch/${USER:-$(id -un)}/mcb_${SLURM_JOB_ID:-manual}"
    mkdir -p "${STAGE_DIR}"
    trap 'rm -rf "${STAGE_DIR}" || true' EXIT
    RUN_ARGS="${RUN_ARGS} --stage-dir ${STAGE_DIR}"
    echo "stage dir: ${STAGE_DIR}"
fi
if [ -n "${MCB_LIMIT:-}" ]; then RUN_ARGS="${RUN_ARGS} --limit ${MCB_LIMIT}"; fi
if [ "${MCB_OVERWRITE:-}" = "1" ]; then RUN_ARGS="${RUN_ARGS} --overwrite"; fi
echo "RUN_ARGS: ${RUN_ARGS}"

if command -v apptainer &>/dev/null && [ -n "${SIF_PATH:-}" ] && [ -f "${SIF_PATH}" ]; then
    apptainer exec \
        --bind /scratch \
        "${SIF_PATH}" \
        bash -c "
            set -euo pipefail
            source ${PROJECT_ROOT}/.venv/bin/activate
            cd ${PROJECT_ROOT}
            python scripts/measure_corpus_blankness.py ${RUN_ARGS}
        "
else
    echo "⚠️ Apptainer not found or SIF_PATH not set. Running on host system."
    source "${PROJECT_ROOT}/.venv/bin/activate"
    cd "${PROJECT_ROOT}"
    python scripts/measure_corpus_blankness.py ${RUN_ARGS}
fi

echo "Done. -> outputs/measure_corpus_blankness/corpus_blankness.parquet"
