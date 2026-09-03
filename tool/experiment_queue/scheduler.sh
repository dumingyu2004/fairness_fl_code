#!/usr/bin/env bash
QDIR=/home/ronnie/fl_queue
JOBS=$QDIR/jobs.tsv
LOGDIR=$QDIR/logs
TOTAL_TARGET=4
STAGGER=90
PY=/home/ronnie/anaconda3/envs/FL/bin/python
mkdir -p "$LOGDIR"
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

total_running() { nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -vi mps | grep -c . ; }

launch_job() {
  local wt=$1 algo=$2 ds=$3 sp=$4 c=$5 tag=$6
  local done_metrics
  done_metrics=$(find "$wt/save_path/$ds/$sp/$algo/BERTCLASSIFIER/${c}Clients/experiment_state" \
    -name metrics.json -path "*repeat_0*" 2>/dev/null | wc -l)
  if [ "$done_metrics" -ge 3 ]; then
    echo "$(date "+%F %T") SKIP(done): $tag" >> "$QDIR/scheduler.log"; return
  fi
  echo "$(date "+%F %T") LAUNCH: $tag" >> "$QDIR/scheduler.log"
  (
    cd "$wt" || exit 1
    export PYTHONPATH="$wt"
    exec "$PY" -u main_SENT_CLF.py \
      -algorithm "$algo" -dataset "$ds" -task SENT_CLF \
      -batch_size 32 -test_batch_size 512 -cuda 0 -max_len 32 \
      -system_data_count 2000000 \
      -split_strategy "$sp" -communication_round_I 50 -algorithm_epoch_T 1 \
      -num_clients_K "$c" -exp_repeat_times 3 -parallel_repeats 1 -base_seed 42 \
      -use_amp true -resume -dataloader_num_workers 8 \
      -partition_cache_root /home/ronnie/fl_partition_cache \
      -tb_monitor "{\"gradient\":false,\"embedding\":false,\"fisher\":false,\"sharpness\":false,\"activation\":false,\"update_stats\":false,\"client_divergence\":false}"
  ) > "$LOGDIR/$tag.log" 2>&1 < /dev/null &
  echo $! > "$LOGDIR/$tag.pid"
}

while IFS="|" read -r wt algo ds sp c tag; do
  [ -z "$wt" ] && continue
  case "$wt" in \#*) continue ;; esac
  while [ "$(total_running)" -ge "$TOTAL_TARGET" ]; do sleep 60; done
  launch_job "$wt" "$algo" "$ds" "$sp" "$c" "$tag"
  sleep "$STAGGER"
done < "$JOBS"
echo "$(date "+%F %T") queue drained, waiting for last jobs" >> "$QDIR/scheduler.log"
wait
echo "$(date "+%F %T") ALL DONE" >> "$QDIR/scheduler.log"
