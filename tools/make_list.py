#!/usr/bin/env python3

# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2023 Jing Du(thuduj12@163.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import json
import re

symbol_str = '[’!"#$%&\'()*+,-./:;<>=?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'

def split_mixed_label(input_str):
    tokens = []
    s = input_str.lower()
    while len(s) > 0:
        match = re.match(r'[A-Za-z!?,<>()\']+', s)
        if match is not None:
            word = match.group(0)
        else:
            word = s[0:1]
        tokens.append(word)
        s = s.replace(word, '', 1).strip(' ')
    return tokens

def query_token_set(txt, symbol_table, lexicon_table):
    tokens_str = tuple()
    tokens_idx = tuple()

    parts = split_mixed_label(txt)
    for part in parts:
        if part == '!sil' or part == '(sil)' or part == '<sil>':
            tokens_str = tokens_str + ('!sil', )
        elif part == '<blk>' or part == '<blank>':
            tokens_str = tokens_str + ('<blk>', )
        elif part == '(noise)' or part == 'noise)' or part == '(noise' or part == '<noise>':
            tokens_str = tokens_str + ('<GBG>', )
        elif part in symbol_table:
            tokens_str = tokens_str + (part, )
        elif part in lexicon_table:
            for ch in lexicon_table[part]:
                tokens_str = tokens_str + (ch, )
        else:
            # case with symbols or meaningless english letter combination
            part = re.sub(symbol_str, '', part)
            for ch in part:
                tokens_str = tokens_str + (ch, )

    for ch in tokens_str:
        if ch in symbol_table:
            tokens_idx = tokens_idx + (symbol_table[ch], )
        elif ch == '!sil':
            if 'sil' in symbol_table:
                tokens_idx = tokens_idx + (symbol_table['sil'], )
            else:
                tokens_idx = tokens_idx + (symbol_table['<blk>'], )
        elif ch == '<GBG>':
            if '<GBG>' in symbol_table:
                tokens_idx = tokens_idx + (symbol_table['<GBG>'], )
            else:
                tokens_idx = tokens_idx + (symbol_table['<blk>'], )
        else:
            if '<GBG>' in symbol_table:
                tokens_idx = tokens_idx + (symbol_table['<GBG>'], )
                logging.info(
                    f'\'{ch}\' is not in token set, replace with <GBG>')
            else:
                tokens_idx = tokens_idx + (symbol_table['<blk>'], )
                logging.info(
                    f'\'{ch}\' is not in token set, replace with <blk>')

    return tokens_str, tokens_idx


def query_token_list(txt, symbol_table, lexicon_table):
    tokens_str = []
    tokens_idx = []

    parts = split_mixed_label(txt)
    for part in parts:
        if part == '!sil' or part == '(sil)' or part == '<sil>':
            tokens_str.append('!sil')
        elif part == '<blk>' or part == '<blank>':
            tokens_str.append('<blk>')
        elif part == '(noise)' or part == 'noise)' or part == '(noise' or part == '<noise>':
            tokens_str.append('<GBG>')
        elif part in symbol_table:
            tokens_str.append(part)
        elif part in lexicon_table:
            for ch in lexicon_table[part]:
                tokens_str.append(ch)
        else:
            # case with symbols or meaningless english letter combination
            part = re.sub(symbol_str, '', part)
            for ch in part:
                tokens_str.append(ch)

    for ch in tokens_str:
        if ch in symbol_table:
            tokens_idx.append(symbol_table[ch])
        elif ch == '!sil':
            if 'sil' in symbol_table:
                tokens_idx.append(symbol_table['sil'])
            else:
                tokens_idx.append(symbol_table['<blk>'])
        elif ch == '<GBG>':
            if '<GBG>' in symbol_table:
                tokens_idx.append(symbol_table['<GBG>'])
            else:
                tokens_idx.append(symbol_table['<blk>'])
        else:
            if '<GBG>' in symbol_table:
                tokens_idx.append(symbol_table['<GBG>'])
                logging.info(
                    f'\'{ch}\' is not in token set, replace with <GBG>')
            else:
                tokens_idx.append(symbol_table['<blk>'])
                logging.info(
                    f'\'{ch}\' is not in token set, replace with <blk>')

    return tokens_str, tokens_idx

def read_token(token_file):
    tokens_table = {}
    with open(token_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            tokens_table[arr[0]] = int(arr[1]) - 1
    fin.close()
    return tokens_table


def read_lexicon(lexicon_file):
    lexicon_table = {}
    with open(lexicon_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().replace('\t', ' ').split()
            assert len(arr) >= 2
            lexicon_table[arr[0]] = arr[1:]
    fin.close()
    return lexicon_table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('wav_file', help='wav file')
    parser.add_argument('text_file', help='text file')
    parser.add_argument('duration_file', help='duration file')
    parser.add_argument('output_file', help='output list file')
    parser.add_argument('--token_file', type=str, default=None, help='the path of tokens.txt')
    parser.add_argument('--lexicon_file', type=str, default=None, help='the path of lexicon.txt')
    args = parser.parse_args()

    wav_table = {}
    with open(args.wav_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            wav_table[arr[0]] = arr[1]

    duration_table = {}
    with open(args.duration_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            duration_table[arr[0]] = float(arr[1])

    token_table = None
    if args.token_file:
        token_table = read_token(args.token_file)
    lexicon_table = None
    if args.lexicon_file:
        lexicon_table = read_lexicon(args.lexicon_file)

    with open(args.text_file, 'r', encoding='utf8') as fin, \
         open(args.output_file, 'w', encoding='utf8') as fout:
        for line in fin:
            arr = line.strip().split(maxsplit=1)
            key = arr[0]
            tokens = None
            if token_table is not None and lexicon_table is not None :
                if len(arr) < 2:  # for some utterence, no text
                    txt = [1]  # the <blank>/sil is indexed by 1
                    tokens = ["sil"]
                else:
                    tokens, txt = query_token_list(arr[1], token_table, lexicon_table)
            else:
                txt = int(arr[1])
            assert key in wav_table
            wav = wav_table[key]
            assert key in duration_table
            duration = duration_table[key]
            if tokens is None:
                line = dict(key=key, txt=txt, duration=duration, wav=wav)
            else:
                line = dict(key=key, tok=tokens, txt=txt, duration=duration, wav=wav)

            json_line = json.dumps(line, ensure_ascii=False)
            fout.write(json_line + '\n')
