#!/bin/bash
#SBATCH --job-name=0038_20260914_atlas_corpus_domain_classifier
#SBATCH --partition=medium-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0038_20260914_atlas_corpus_domain_classifier/%j_0038_20260914_atlas_corpus_domain_classifier.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0038_20260914_atlas_corpus_domain_classifier/%j_0038_20260914_atlas_corpus_domain_classifier.out
#SBATCH --signal=B:USR1@72
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=2:00:00
# ユーザー指摘(README「experiments/0037」参照): 0037のLOIO AUROCが25所見全て
# (Hypertrophyも含む)でほぼ1.0に飽和した。これは所見を学習できているのではなく、
# 正例(atlas図版由来)と負例(コーパス由来)がそもそも画像ドメインとして分離しやすい
# (スキャナ・染色・解像度・図版特有のノイズ等)せいではないか、という仮説を直接
# 検証する。所見ラベルを無視し、NNLアトラス25所見全ての図版タイルを「atlas」、
# コーパスのランダムパッチを「corpus」として単一の二値分類器を学習し、
# leave-one-finding-out(LOFO、25fold)とleave-one-image-out(LOIO、91fold)の
# AUROCでドメイン分離性そのものを定量化する。加えてコーパスGTが確定している
# 7所見については、コーパス内GTパッチ(atlas図版ではなく本物のTG-GATEs由来)
# もこの分類器でスコアし、atlas図版側のスコア分布と比較する。
#
# GPU: atlas図版91枚・計約4900タイルの埋め込み(uni_v1 UNIエンコーダ)に必要。
# FAISS索引・PatchIndexは不要(検索は行わない、分類器学習とスコアリングのみ)。
# 負例サンプリング・GTパッチスコアリングはコーパスのほぼ全スライドのh5に触れる
# (experiments/0035-0037と同様、NFS h5ランダムアクセス)。
# ⚠️ リソースを変えたら --partition と --signal のマージンも手動で見直すこと。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0038_20260914_atlas_corpus_domain_classifier"

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
