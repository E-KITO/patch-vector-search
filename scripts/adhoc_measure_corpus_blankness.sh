#!/bin/bash
#SBATCH --job-name=adhoc_measure_corpus_blankness
#SBATCH --partition=x-large-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/adhoc_measure_corpus_blankness/%j.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/adhoc_measure_corpus_blankness/%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64g
#SBATCH --time=8:00:00
# scripts/measure_corpus_blankness.py: 既定コーパス(experiments/0002)の全 ~18.4M
# パッチを raw WSI から切り出し、明るさ・彩度(sat_frac)を測定して
# outputs/measure_corpus_blankness/corpus_blankness.parquet に落とす。
# 背景パッチをコーパスから除外する準備(段階1)。閾値の決定と manifest / FAISS 索引の
# 再構築は experiments/0017 / 0018。
#
# 索引もエンコーダも使わない(GPU 不要)。支配的コストは openslide の level-0 領域読み
# ~18.4M 回(NFS 上の 1000 SVS、filesrv02 は HDD 障害調査中なので遅め想定)。
# スライド単位の multiprocessing で per-slide parquet を書くので再開可能:
# タイムアウトしても同じジョブを再投入すれば残りのスライドだけ処理する。
#
# 進捗: ジョブログに tqdm バー + 25スライドごとのチェックポイント行。
#       さらに outputs/measure_corpus_blankness/_progress.json を毎スライド更新するので
#       `cat outputs/measure_corpus_blankness/_progress.json` でいつでも状況が見える。
#
# 初回は下の smoke test 行(--limit 5)で 1スライドあたりの所要時間と filesrv02 の
# I/O 速度を確認してから、フル投入するのが安全。

set -euo pipefail

PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"

echo "Running scripts/measure_corpus_blankness.py on $(hostname) ..."

RUN_ARGS="--workers ${SLURM_CPUS_PER_TASK:-32}"
# smoke test: RUN_ARGS="--workers 8 --limit 5"

if command -v apptainer &>/dev/null && [ -n "${SIF_PATH:-}" ] && [ -f "${SIF_PATH}" ]; then
    apptainer exec \
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
