#!/bin/bash
# Copyright 2021  Binbin Zhang
#                 Menglong Xu

. ./path.sh

set -euo pipefail

stage=$1
stop_stage=$2
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
noise_lmdb=
reverb_lmdb=

. tools/parse_options.sh || exit 1;
window_shift=50

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  echo "Extracte all datasets"
  local/snips_data_extract.sh --dl_dir $download_dir
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Preparing datasets..."
  mkdir -p dict
  echo "<FILLER> -1" > dict/dict.txt
  echo "<HEY_SNIPS> 0" >> dict/dict.txt
  awk '{print $1}' dict/dict.txt > dict/words.txt

  for folder in train dev test; do
    mkdir -p data/$folder
    json_path=$download_dir/hey_snips_research_6k_en_train_eval_clean_ter/$folder.json
    local/prepare_data.py $download_dir/hey_snips_research_6k_en_train_eval_clean_ter/audio_files $json_path \
      dict/dict.txt data/$folder
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
   wekws/bin/train.py --gpus $gpus \
    --config $config \
    --train_data data/train/data.list \
    --cv_data data/dev/data.list \
    --model_dir $dir \
    --num_workers 8 \
    --num_keywords $num_keywords \
    --min_duration 50 \
    --seed 666 \
    --dict ./dict \
    $cmvn_opts \
    ${reverb_lmdb:+--reverb_lmdb $reverb_lmdb} \
    ${noise_lmdb:+--noise_lmdb $noise_lmdb} \
    ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Do model average, Compute FRR/FAR ..."
  python wekws/bin/average_model.py \
    --dst_model $score_checkpoint \
    --src_path $dir  \
    --num ${num_average} \
    --val_best
  result_dir=$dir/test_$(basename $score_checkpoint)
  mkdir -p $result_dir
  python wekws/bin/score.py \
    --config $dir/config.yaml \
    --test_data data/test/data.list \
    --gpu 0 \
    --batch_size 256 \
    --checkpoint $score_checkpoint \
    --score_file $result_dir/score.txt \
    --dict ./dict \
    --num_workers 8

  for keyword in `tail -n +2 dict/words.txt`; do
    python wekws/bin/compute_det.py \
      --keyword $keyword \
      --test_data data/test/data.list \
      --window_shift $window_shift \
      --score_file $result_dir/score.txt \
      --stats_file $result_dir/stats.${keyword}.txt
  done
  python wekws/bin/plot_det_curve.py \
    --keywords_dict dict/dict.txt \
    --stats_dir $result_dir \
    --figure_file $result_dir/det.png \
    --xlim 10 \
    --x_step 2 \
    --ylim 10 \
    --y_step 2
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  jit_model=$(basename $score_checkpoint | sed -e 's:.pt$:.zip:g')
  onnx_model=$(basename $score_checkpoint | sed -e 's:.pt$:.onnx:g')
  python wekws/bin/export_jit.py \
    --config $dir/config.yaml \
    --checkpoint $score_checkpoint \
    --jit_model $dir/$jit_model
  python wekws/bin/export_onnx.py \
    --config $dir/config.yaml \
    --checkpoint $score_checkpoint \
    --onnx_model $dir/$onnx_model
fi
