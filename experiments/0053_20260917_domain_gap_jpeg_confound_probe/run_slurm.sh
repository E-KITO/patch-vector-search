#!/bin/bash
#SBATCH --job-name=0053_20260917_domain_gap_jpeg_confound_probe
#SBATCH --partition=medium-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0053_20260917_domain_gap_jpeg_confound_probe/%j_0053_20260917_domain_gap_jpeg_confound_probe.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0053_20260917_domain_gap_jpeg_confound_probe/%j_0053_20260917_domain_gap_jpeg_confound_probe.out
#SBATCH --signal=B:USR1@72
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=2:00:00
# ユーザー指摘: experiments/0038のatlas vs corpusドメインギャップ(所見によらず
# スコア差10〜12)は、atlas図版が全て.jpg(PIL.Image.openでロード)である一方
# corpusパッチは.svs直接切り出しの生ピクセルをJPEGを一度も経由せずUNI埋め込み
# している、という「拡張子(画像パイプライン)の違い」が一因になっていないか
# 切り分けられていなかった。experiments/0038と同じ手順でドメイン分類器を学習し
# (プローブに使うスライドは負例プールから除外)、コーパスから新たに切り出した
# 生パッチについて「生ピクセルのまま埋め込み」vs「JPEG往復(品質95/85/70/50)
# してから埋め込み」のスコア差を、atlas-corpus間の本物のギャップと比較する。
#
# GPU: 生パッチ埋め込み(200パッチ×[生+JPEG4品質]=1000embeds)とatlas図版
# (91枚・約4900タイル)の埋め込みにUNIエンコーダが必要。
# ⚠️ リソースを変えたら --partition と --signal のマージンも手動で見直すこと。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0053_20260917_domain_gap_jpeg_confound_probe"

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
