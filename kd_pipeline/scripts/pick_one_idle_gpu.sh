#!/usr/bin/env bash
# 选 1 张「利用率足够低」且「空闲显存尽量大」的卡。
# 默认：GPU 利用率 <=5% 视为空闲；可用环境变量 MAX_UTIL=10 放宽。
set -euo pipefail
MAX_UTIL="${MAX_UTIL:-5}"
if ! command -v nvidia-smi &>/dev/null; then
  echo "nvidia-smi not found" >&2
  exit 1
fi

# index, util%, free_MiB
row=$(
  nvidia-smi --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader,nounits |
    awk -F', ' -v mx="$MAX_UTIL" '$2+0<=mx {print $1, $3+0}' |
    sort -k2 -nr |
    head -1
)

if [[ -z "${row}" ]]; then
  echo "空闲 GPU 不足 1 张（当前阈值 util<=${MAX_UTIL}%）。请调大 MAX_UTIL 或稍后重试。" >&2
  exit 1
fi

g0=$(echo "${row}" | awk '{print $1}')
f0=$(echo "${row}" | awk '{print $2}')
echo "pick_one_idle_gpu: using GPU ${g0} (${f0} MiB free)" >&2

# 第一行：给 CUDA_VISIBLE_DEVICES=
echo "${g0}"
# 第二行：DeepSpeed --include
echo "localhost:${g0}"
