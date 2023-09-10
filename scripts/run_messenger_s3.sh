#! /bin/bash
#

task=s3
name=$1
device=$2
seed=$3

shift
shift
shift

export CUDA_VISIBLE_DEVICES=$device; python dynalang/train.py \
  --run.script parallel \
  --run.log_keys_video log_image \
  --logdir ~/logdir/messenger/${task}_${name} \
  --use_wandb True \
  --task messenger_${task} \
  --env.messenger.length 128 \
  --env.messenger.vis True \
  --dataset_excluded_keys info \
  --seed $seed \
  --decoder.image_dist binary \
  --encoder.mlp_keys token_embed \
  --decoder.mlp_keys token_embed \
  --batch_size 24 \
  --batch_length 512 \
  --envs.amount 66 \
  --run.actor_batch 32 \
  --run.actor_threads 4 \
  --run.train_ratio 32 \
  "$@"
