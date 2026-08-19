#!/bin/bash
#SBATCH --job-name=adhoc_exact_vs_approx_diagnostic
#SBATCH --partition=large-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/adhoc_exact_vs_approx_diagnostic/%j.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/adhoc_exact_vs_approx_diagnostic/%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=4:00:00
# Step 3: scripts/exact_vs_approx_diagnostic.py を実行し、7カテゴリそれぞれで
# 近似(PQ)スコアによるランキングと、厳密(exact)スコアで再計算したランキングを
# 比較する。多くのカテゴリは最大MAX_TILES_RERANKED=8タイルに絞って厳密計算するが、
# Kupffer細胞増殖・封入体(FULL_TILE_CATEGORIES)の2カテゴリのみ全タイル対象に
# するため、そこだけI/Oコストが大きく増える見込み(実測ログから1タイルあたり
# 約10秒、対象2カテゴリ計6画像の全タイル数は未計測だが数百枚程度と推定 —
# 会話内の見積もりでは10〜50分程度増える想定、ただし不確実性が大きいため
# time-limitは前回の2時間から4時間へ余裕を持たせている)。
# (experiments/0003_..._query_demo / adhoc_validate_against_ground_truth.sh
# と同等のGPU/CPU/メモリ設定)。
# 途中でタイムアウトしても、scripts/exact_vs_approx_diagnostic.py は
# カテゴリ処理ごとにCSVを書き出すため、それまでの結果は失われない。

set -euo pipefail

PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"

echo "Running scripts/exact_vs_approx_diagnostic.py on $(hostname) ..."

if command -v apptainer &>/dev/null && [ -n "${SIF_PATH:-}" ] && [ -f "${SIF_PATH}" ]; then
    apptainer exec \
        --nv \
        "${SIF_PATH}" \
        bash -c "
            set -euo pipefail
            source ${PROJECT_ROOT}/.venv/bin/activate
            export CUDA_HOME=/usr/local/cuda
            cd ${PROJECT_ROOT}
            python scripts/exact_vs_approx_diagnostic.py
        "
else
    echo "⚠️ Apptainer not found or SIF_PATH not set. Running on host system."
    source "${PROJECT_ROOT}/.venv/bin/activate"
    cd "${PROJECT_ROOT}"
    python scripts/exact_vs_approx_diagnostic.py
fi

echo "Done. Results written to outputs/gt_validation_exact_vs_approx.csv"
