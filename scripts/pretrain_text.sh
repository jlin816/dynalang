#! /bin/bash
#

task=s2
name=$1
device=$2
seed=$3
dataset=$4
eval_dir=$5

shift
shift
shift
shift
shift 

export CUDA_VISIBLE_DEVICES=$device; python dynalang/train.py \
  --run.script offline-text \
  --run.pretrain 1e8 \
  --run.save_every 1e6 \
  --replay_size 1e6 \
  --text_dataset $dataset \
  --logdir ~/logdir/textpt/$name \
  --eval_dir $eval_dir \
  --use_wandb True \
  --task messenger_${task} \
  --dataset_excluded_keys info \
  --seed $seed \
  --decoder.image_dist binary \
  --encoder.mlp_keys token$ \
  --decoder.mlp_keys token$ \
  --decoder.vector_dist onehot \
  --batch_size 512 \
  --batch_length 128 \
  --run.pretrain_wm_only True \
  --loss_scales.cont 0 \
  --loss_scales.reward 0 \
  --loss_scales.image 0 \
  --loss_scales.vector 1 \
  --zero_data_keys action,reward,cont,image \
  --zero_cnn True \
  --grad_heads decoder \
  --skip_cnn_training True \
  --env.messenger.length 64 \
  --envs.amount 1 \
  "$@"
