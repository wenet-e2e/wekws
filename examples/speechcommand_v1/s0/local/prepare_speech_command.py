#!/usr/bin/env python3
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

import os
import argparse

CLASSES = 'unknown, yes, no, up, down, left, right, on, off, stop, go'.split(
    ', ')
CLASS_TO_IDX = {CLASSES[i]: str(i) for i in range(len(CLASSES))}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='prepare kaldi format file for google speech command')
    parser.add_argument(
        '--wav_list',
        required=True,
        help='full path of a wav file in google speech command dataset')
    parser.add_argument('--data_dir',
                        required=True,
                        help='folder to write kaldi format files')
    args = parser.parse_args()

    data_dir = args.data_dir
    f_wav_scp = open(os.path.join(data_dir, 'wav.scp'), 'w')
    f_text = open(os.path.join(data_dir, 'text'), 'w')
    with open(args.wav_list) as f:
        for line in f.readlines():
            keyword, file_name = line.strip().split('/')[-2:]
            file_name_new = file_name.split('.')[0]
            wav_id = '_'.join([keyword, file_name_new])
            file_dir = line.strip()
            f_wav_scp.writelines(wav_id + ' ' + file_dir + '\n')
            label = CLASS_TO_IDX[
                keyword] if keyword in CLASS_TO_IDX else CLASS_TO_IDX["unknown"]
            f_text.writelines(wav_id + ' ' + str(label) + '\n')
    f_wav_scp.close()
    f_text.close()
