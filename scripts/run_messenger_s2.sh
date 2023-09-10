#! /bin/bash
#

task=s2
name=$1
device=$2
seed=$3

shift
shift
shift

export CUDA_VISIBLE_DEVICES=$device; python dynalang/train.py \
  --run.script train \
  --run.log_keys_video log_image \
  --logdir ~/logdir/messenger/${task}_${name} \
  --use_wandb True \
  --task messenger_${task} \
  --envs.amount 16 \
  --env.messenger.length 64 \
  --env.messenger.vis True \
  --dataset_excluded_keys info \
  --seed $seed \
  --decoder.image_dist binary \
  --encoder.mlp_keys token_embed \
  --decoder.mlp_keys token_embed \
  --batch_size 24 \
  --batch_length 256 \
  --run.train_ratio 64 \
  "$@"
