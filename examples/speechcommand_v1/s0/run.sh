#!/bin/bash
# Copyright 2021  Binbin Zhang
#                 Jingyong Hou

. ./path.sh

export CUDA_VISIBLE_DEVICES="0"

stage=-1
stop_stage=0

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

