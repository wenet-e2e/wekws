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
import shutil
import argparse


def move_files(src_folder, to_folder, list_file):
    with open(list_file) as f:
        for line in f.readlines():
            line = line.rstrip()
            dirname = os.path.dirname(line)
            dest = os.path.join(to_folder, dirname)
            if not os.path.exists(dest):
                os.mkdir(dest)
            shutil.move(os.path.join(src_folder, line), dest)


if __name__ == '__main__':
    '''Splits the google speech commands into train, validation and test set'''
    parser = argparse.ArgumentParser(
        description='Split google command dataset.')
    parser.add_argument(
        'root',
        type=str,
        help='the path to the root folder of the google commands dataset')
    args = parser.parse_args()

    audio_folder = os.path.join(args.root, 'audio')
    validation_path = os.path.join(audio_folder, 'validation_list.txt')
    test_path = os.path.join(audio_folder, 'testing_list.txt')

    valid_folder = os.path.join(args.root, 'valid')
    test_folder = os.path.join(args.root, 'test')
    train_folder = os.path.join(args.root, 'train')

    os.mkdir(valid_folder)
    os.mkdir(test_folder)

    move_files(audio_folder, test_folder, test_path)
    move_files(audio_folder, valid_folder, validation_path)
    os.rename(audio_folder, train_folder)
