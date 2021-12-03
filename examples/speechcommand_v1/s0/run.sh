#!/bin/bash
# Copyright 2021  Binbin Zhang
#                 Jingyong Hou

. ./path.sh

export CUDA_VISIBLE_DEVICES="0"

stage=2
stop_stage=4
num_keywords=11

config=conf/mdtc.yaml
norm_mean=false
norm_var=false
gpu_id=4

checkpoint=
dir=exp/mdtc_debug

num_average=1
score_checkpoint=$dir/avg_${num_average}.pt

# your data dir
download_dir=/mnt/mnt-data-3/jingyong.hou/data
speech_command_dir=$download_dir/speech_commands_v1
. tools/parse_options.sh || exit 1;

set -euo pipefail

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  echo "Download and extract all datasets"
  local/data_download.sh --dl_dir $download_dir
  python local/split_dataset.py $download_dir/speech_commands_v1
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Start preparing Kaldi format files"
  for x in train test valid;
  do
    data=data/$x
    mkdir -p $data
    # make wav.scp utt2spk text file
    find $speech_command_dir/$x -name *.wav | grep -v "_background_noise_" > $data/wav.list
    python local/prepare_speech_command.py --wav_list=$data/wav.list --data_dir=$data
  done
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Compute CMVN and Format datasets"
  tools/compute_cmvn_stats.py --num_workers 16 --train_config $config \
    --in_scp data/train/wav.scp \
    --out_cmvn data/train/global_cmvn

  for x in train valid test; do
    tools/wav_to_duration.sh --nj 8 data/$x/wav.scp data/$x/wav.dur
    tools/make_list.py data/$x/wav.scp data/$x/text \
      data/$x/wav.dur data/$x/data.list
  done
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Start training ..."
  mkdir -p $dir
  cmvn_opts=
  $norm_mean && cmvn_opts="--cmvn_file data/train/global_cmvn"
  $norm_var && cmvn_opts="$cmvn_opts --norm_var"
  python kws/bin/train.py --gpu $gpu_id \
    --config $config \
    --train_data data/train/data.list \
    --cv_data data/valid/data.list \
    --model_dir $dir \
    --num_workers 8 \
    --num_keywords $num_keywords \
    --min_duration 50 \
    $cmvn_opts \
    ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # Do model average
  python kws/bin/average_model.py \
    --dst_model $score_checkpoint \
    --src_path $dir  \
    --num ${num_average} \
    --val_best

  # Compute posterior score
  result_dir=$dir/test_$(basename $score_checkpoint)
  mkdir -p $result_dir
  python kws/bin/test.py --gpu 3 \
    --config $dir/config.yaml \
    --test_data data/test/data.list \
    --batch_size 256 \
    --num_workers 8 \
    --checkpoint $score_checkpoint
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  python kws/bin/export_jit.py --config $dir/config.yaml \
    --checkpoint $score_checkpoint \
    --output_file $dir/final.zip \
    --output_quant_file $dir/final.quant.zip
fi
