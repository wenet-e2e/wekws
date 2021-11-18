#!/bin/bash

# Copyright  2021  Jingyong Hou (houjingyong@gmail.com)
[ -f ./path.sh ] && . ./path.sh

dl_dir=./data/local

. tools/parse_options.sh || exit 1;
data_dir=$dl_dir
file_name=speech_commands_v0.01.tar.gz
speech_command_dir=$data_dir/speech_commands_v1
audio_dir=$data_dir/speech_commands_v1/audio
url=http://download.tensorflow.org/data/$file_name
mkdir -p $data_dir
if [ ! -f $data_dir/$file_name ]; then
    echo "downloading $url..."
    wget -O $data_dir/$file_name $url
else
    echo "$file_name exist in $data_dir, skip download it"
fi

if [ ! -f $speech_command_dir/.extracted ]; then
    mkdir -p $audio_dir
    tar -xzvf $data_dir/$file_name -C $audio_dir
    touch $speech_command_dir/.extracted
else
    echo "$speech_command_dir/.exatracted exist in $speech_command_dir, skip exatraction"
fi

exit 0
