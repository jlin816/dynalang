#!/bin/bash

name=$1
device=$2
seed=$3

shift
shift
shift

export CUDA_VISIBLE_DEVICES=$device; python dynalang/train.py \
  --logdir ~/logdir/langroom/$name \
  --seed $seed \
  --run.script parallel \
  --envs.amount 64 \
  --run.actor_batch 32 \
  --run.actor_threads 2 \
  --run.batch_size 16 \
  --batch_length 64 \
  --configs langroom \
  --rssm.deter 6144 \
  --rssm.bottleneck 2048 \
  "$@"
