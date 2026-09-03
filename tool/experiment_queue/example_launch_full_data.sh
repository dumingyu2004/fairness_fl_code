#!/usr/bin/env bash
set -o pipefail
cd "$HOME/.config/superpowers/worktrees/fairness_fl_code/fix-repeats-dirichlet" || exit 1
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PY="$HOME/anaconda3/envs/FL/bin/python"
exec "$PY" -u main_SENT_CLF.py \
  -algorithm FedAvg \
  -dataset moji \
  -task SENT_CLF \
  -batch_size 32 \
  -test_batch_size 256 \
  -cuda 0 \
  -max_len 32 \
  -system_data_count 2000000 \
  -split_strategy Dirichlet05 \
  -communication_round_I 50 \
  -algorithm_epoch_T 1 \
  -num_clients_K 20 \
  -exp_repeat_times 3 \
  -parallel_repeats 1 \
  -base_seed 42 \
  -use_amp true \
  -resume \
  -dataloader_num_workers 8 \
  -tb_monitor "{\"gradient\":false,\"embedding\":false,\"fisher\":false,\"sharpness\":false,\"activation\":false,\"update_stats\":false,\"client_divergence\":false}"
