#!/bin/bash
# Copyright 2021  Binbin Zhang
#                 Menglong Xu

. ./path.sh

stage=0
stop_stage=4
num_keywords=1

config=conf/ds_tcn.yaml
norm_mean=true
norm_var=true
gpus="0"

checkpoint=
dir=exp/ds_tcn

num_average=30
score_checkpoint=$dir/avg_${num_average}.pt

download_dir=./data/local # your data dir

. tools/parse_options.sh || exit 1;

set -euo pipefail

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  echo "Extracte all datasets"
  local/snips_data_extract.sh --dl_dir $download_dir
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Preparing datasets..."
  mkdir -p dict
  echo "<filler> -1" > dict/words.txt
  echo "Hey_Snips 0" >> dict/words.txt

  for folder in train dev test; do
    mkdir -p data/$folder
    json_path=$download_dir/hey_snips_research_6k_en_train_eval_clean_ter/$folder.json
    local/prepare_data.py $download_dir/hey_snips_research_6k_en_train_eval_clean_ter/audio_files $json_path \
      data/$folder
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
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
   kws/bin/train.py --gpus $gpus \
    --config $config \
    --train_data data/train/data.list \
    --cv_data data/dev/data.list \
    --model_dir $dir \
    --num_workers 8 \
    --num_keywords $num_keywords \
    --min_duration 50 \
    --seed 777 \
    $cmvn_opts \
    ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Do model average, Compute FRR/FAR ..."
  python kws/bin/average_model.py \
    --dst_model $score_checkpoint \
    --src_path $dir  \
    --num ${num_average} \
    --val_best
  result_dir=$dir/test_$(basename $score_checkpoint)
  mkdir -p $result_dir
  python kws/bin/score.py \
    --config $dir/config.yaml \
    --test_data data/test/data.list \
    --batch_size 256 \
    --checkpoint $score_checkpoint \
    --score_file $result_dir/score.txt \
    --num_workers 8
  first_keyword=0
  last_keyword=$(($num_keywords+$first_keyword-1))
  for keyword in $(seq $first_keyword $last_keyword); do
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

