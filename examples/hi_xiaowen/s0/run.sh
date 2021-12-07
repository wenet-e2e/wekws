#!/bin/bash
# Copyright 2021  Binbin Zhang

. ./path.sh

export CUDA_VISIBLE_DEVICES="0"

stage=0
stop_stage=4
num_keywords=2

config=conf/ds_tcn.yaml
norm_mean=true
norm_var=true
gpu_id=0

checkpoint=
dir=exp/ds_tcn

num_average=30
score_checkpoint=$dir/avg_${num_average}.pt

download_dir=./data/local # your data dir

. tools/parse_options.sh || exit 1;

set -euo pipefail

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  echo "Download and extracte all datasets"
  local/mobvoi_data_download.sh --dl_dir $download_dir
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Preparing datasets..."
  mkdir -p dict
  echo "<filler> -1" > dict/words.txt
  echo "Hi_Xiaowen 0" >> dict/words.txt
  echo "Nihao_Wenwen 1" >> dict/words.txt

  for folder in train dev test; do
    mkdir -p data/$folder
    for prefix in p n; do
      mkdir -p data/${prefix}_$folder
      json_path=$download_dir/mobvoi_hotword_dataset_resources/${prefix}_$folder.json
      local/prepare_data.py $download_dir/mobvoi_hotword_dataset $json_path \
        data/${prefix}_$folder
    done
    cat data/p_$folder/wav.scp data/n_$folder/wav.scp > data/$folder/wav.scp
    cat data/p_$folder/text data/n_$folder/text > data/$folder/text
    rm -rf data/p_$folder data/n_$folder
  done
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Compute CMVN and Format datasets"
  tools/compute_cmvn_stats.py --num_workers 16 --train_config $config \
    --in_scp data/train/wav.scp \
    --out_cmvn data/train/global_cmvn

  for x in train dev test; do
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
    --cv_data data/dev/data.list \
    --model_dir $dir \
    --num_workers 8 \
    --num_keywords $num_keywords \
    --min_duration 50 \
    --seed 666 \
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
  python kws/bin/score.py --gpu 0 \
    --config $dir/config.yaml \
    --test_data data/test/data.list \
    --batch_size 256 \
    --checkpoint $score_checkpoint \
    --score_file $result_dir/score.txt \
    --num_workers 8
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  # Compute detection error tradeoff
  result_dir=$dir/test_$(basename $score_checkpoint)
  for keyword in 0 1; do
    python kws/bin/compute_det.py \
      --keyword $keyword \
      --test_data data/test/data.list \
      --score_file $result_dir/score.txt \
      --stats_file $result_dir/stats.${keyword}.txt
  done
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  python kws/bin/export_jit.py --config $dir/config.yaml \
    --checkpoint $score_checkpoint \
    --output_file $dir/final.zip \
    --output_quant_file $dir/final.quant.zip
fi

