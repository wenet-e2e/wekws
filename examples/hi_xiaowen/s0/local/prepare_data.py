#!/usr/bin/env python3

# Copyright 2018-2020  Yiming Wang
#           2018-2020  Daniel Povey
#           2021       Binbin Zhang
# Apache 2.0
""" This script prepares the Mobvoi data into kaldi format.
"""

import argparse
import os
import json


def main():
    parser = argparse.ArgumentParser(description="""Prepare data.""")
    parser.add_argument('wav_dir',
                        type=str,
                        help='dir containing all the wav files')
    parser.add_argument('path', type=str, help='path to the json file')
    parser.add_argument('out_dir', type=str, help='out dir')
    args = parser.parse_args()

    with open(args.path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        utt_id, label = [], []
        for entry in data:
            utt_id.append(entry['utt_id'])
            label.append(int(entry['keyword_id']))

    abs_dir = os.path.abspath(args.wav_dir)
    wav_path = os.path.join(args.out_dir, 'wav.scp')
    text_path = os.path.join(args.out_dir, 'text')
    with open(wav_path, 'w', encoding='utf-8') as f_wav, \
         open(text_path, 'w', encoding='utf-8') as f_text:
        for utt, l in zip(utt_id, label):
            f_wav.write('{} {}\n'.format(utt,
                                         os.path.join(abs_dir, utt + ".wav")))
            f_text.write('{} {}\n'.format(utt, l))


if __name__ == "__main__":
    main()
