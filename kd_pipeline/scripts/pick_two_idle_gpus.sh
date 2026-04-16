#!/usr/bin/env bash
# 选取 2 张「利用率足够低」且「空闲显存尽量大」的卡，输出 CUDA_VISIBLE_DEVICES 用的 "i,j"。
# 默认：GPU 利用率 <=5% 视为空闲；可用环境变量 MAX_UTIL=10 放宽。
set -euo pipefail
MAX_UTIL="${MAX_UTIL:-5}"
if ! command -v nvidia-smi &>/dev/null; then
  echo "nvidia-smi not found" >&2
  exit 1
fi
# index, util%, free_MiB
mapfile -t rows < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader,nounits |
  awk -F', ' -v mx="$MAX_UTIL" '$2+0<=mx {print $1, $3+0}' | sort -k2 -nr)
if [[ ${#rows[@]} -lt 2 ]]; then
  echo "空闲 GPU 不足 2 张（当前阈值 util<=${MAX_UTIL}%）。请调大 MAX_UTIL 或稍后重试。" >&2
  exit 1
fi
g0=$(echo "${rows[0]}" | awk '{print $1}')
g1=$(echo "${rows[1]}" | awk '{print $1}')
f0=$(echo "${rows[0]}" | awk '{print $2}')
f1=$(echo "${rows[1]}" | awk '{print $2}')
echo "pick_two_idle_gpus: using GPU ${g0} (${f0} MiB free) + GPU ${g1} (${f1} MiB free)" >&2
# 第一行：逗号分隔，给 CUDA_VISIBLE_DEVICES= 用（勿与 deepspeed --num_gpus 同时用，会被覆盖）
echo "${g0},${g1}"
# 第二行：DeepSpeed --include（绑定物理卡，推荐）
echo "localhost:${g0},${g1}"
