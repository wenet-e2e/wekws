#!/bin/bash

# Copyright  2018-2020  Yiming Wang
#            2018-2020  Daniel Povey
#            2021       Binbin Zhang
#                       Menglong Xu

[ -f ./path.sh ] && . ./path.sh

dl_dir=data/download

. tools/parse_options.sh || exit 1;

mkdir -p $dl_dir

# Fill the following form:
# https://forms.gle/JtmFYM7xK1SaMfZYA
# to download the dataset
dataset=hey_snips_kws_4.0.tar.gz
src_path=$dl_dir

if [ -d $dl_dir/$(basename "$dataset" .tar.gz) ]; then
  echo "Not extracting $(basename "$dataset" .tar.gz) as it is already there."
else
  echo "Extracting $dataset..."
  tar -xvzf $src_path/$dataset -C $dl_dir || exit 1;
  echo "Done extracting $dataset."
fi

exit 0
