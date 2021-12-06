#!/bin/bash

# Copyright (c) 2021 Jingyong Hou (houjingyong@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
