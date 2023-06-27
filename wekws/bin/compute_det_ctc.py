# Copyright (c) 2021 Binbin Zhang(binbzha@qq.com)
#               2022 Shaoqing Yu(954793264@qq.com)
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
import glob
import json
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import pypinyin  # for Chinese Character
from tools.make_list import query_token_set, read_lexicon, read_token

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


def space_mixed_label(input_str):
    splits = split_mixed_label(input_str)
    space_str = ''.join(f'{sub} ' for sub in splits)
    return space_str.strip()

def load_label_and_score(keywords_list, label_file, score_file, true_keywords):
    score_table = {}
    with open(score_file, 'r', encoding='utf8') as fin:
        # read score file and store in table
        for line in fin:
            arr = line.strip().split()
            key = arr[0]
            is_detected = arr[1]
            if is_detected == 'detected':
                keyword=true_keywords[arr[2]]
                if key not in score_table:
                    score_table.update({
                        key: {
                            'kw': space_mixed_label(keyword),
                            'confi': float(arr[3])
                        }
                    })
            else:
                if key not in score_table:
                    score_table.update({key: {'kw': 'unknown', 'confi': -1.0}})

    label_lists = []
    with open(label_file, 'r', encoding='utf8') as fin:
        for line in fin:
            obj = json.loads(line.strip())
            label_lists.append(obj)

    # build empty structure for keyword-filler infos
    keyword_filler_table = {}
    for keyword in keywords_list:
        keyword = true_keywords[keyword]
        keyword = space_mixed_label(keyword)
        keyword_filler_table[keyword] = {}
        keyword_filler_table[keyword]['keyword_table'] = {}
        keyword_filler_table[keyword]['keyword_duration'] = 0.0
        keyword_filler_table[keyword]['filler_table'] = {}
        keyword_filler_table[keyword]['filler_duration'] = 0.0

    for obj in label_lists:
        assert 'key' in obj
        assert 'wav' in obj
        assert 'tok' in obj   # here we use the tokens
        assert 'duration' in obj

        key = obj['key']
        txt = "".join(obj['tok'])
        txt = space_mixed_label(txt)
        txt_regstr_lrblk = ' ' + txt + ' '
        duration = obj['duration']
        assert key in score_table

        for keyword in keywords_list:
            keyword = true_keywords[keyword]
            keyword = space_mixed_label(keyword)
            keyword_regstr_lrblk = ' ' + keyword + ' '
            if txt_regstr_lrblk.find(keyword_regstr_lrblk) != -1:
                if keyword == score_table[key]['kw']:
                    keyword_filler_table[keyword]['keyword_table'].update(
                        {key: score_table[key]['confi']})
                else:
                    # uttrance detected but not match this keyword
                    keyword_filler_table[keyword]['keyword_table'].update(
                        {key: -1.0})
                keyword_filler_table[keyword]['keyword_duration'] += duration
            else:
                if keyword == score_table[key]['kw']:
                    keyword_filler_table[keyword]['filler_table'].update(
                        {key: score_table[key]['confi']})
                else:
                    # uttrance if detected, which is not FA for this keyword
                    keyword_filler_table[keyword]['filler_table'].update(
                        {key: -1.0})
                keyword_filler_table[keyword]['filler_duration'] += duration

    return keyword_filler_table

def load_stats_file(stats_file):
    values = []
    with open(stats_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            threshold, fa_per_hour, frr = arr
            values.append([float(fa_per_hour), float(frr) * 100])
    values.reverse()
    return np.array(values)

def plot_det(dets_dir, figure_file, xlim=5, x_step=1, ylim=35, y_step=5):
    det_title = "DetCurve"
    plt.figure(dpi=200)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.size'] = 12

    for file in glob.glob(f'{dets_dir}/*stats*.txt'):
        logging.info(f'reading det data from {file}')
        label = os.path.basename(file).split('.')[1]
        label = "".join(pypinyin.lazy_pinyin(label))
        values = load_stats_file(file)
        plt.plot(values[:, 0], values[:, 1], label=label)

    plt.xlim([0, xlim])
    plt.ylim([0, ylim])
    plt.xticks(range(0, xlim + x_step, x_step))
    plt.yticks(range(0, ylim + y_step, y_step))
    plt.xlabel('False Alarm Per Hour')
    plt.ylabel('False Rejection Rate (%)')
    plt.grid(linestyle='--')
    plt.legend(loc='best', fontsize=6)
    plt.savefig(figure_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compute det curve')
    parser.add_argument('--test_data', required=True, help='label file')
    parser.add_argument('--keywords', type=str, default=None, help='keywords, split with comma(,)')
    parser.add_argument('--token_file', type=str, default=None, help='the path of tokens.txt')
    parser.add_argument('--lexicon_file', type=str, default=None, help='the path of lexicon.txt')
    parser.add_argument('--score_file', required=True, help='score file')
    parser.add_argument('--step', type=float, default=0.01,
                        help='threshold step')
    parser.add_argument('--window_shift', type=int, default=50,
                        help='window_shift is used to skip the frames after triggered')
    parser.add_argument('--stats_dir',
                        required=False,
                        default=None,
                        help='false reject/alarm stats dir, default in score_file')
    parser.add_argument('--det_curve_path',
                        required=False,
                        default=None,
                        help='det curve path, default is stats_dir/det.png')
    parser.add_argument(
        '--xlim',
        type=int,
        default=5,
        help='xlim：range of x-axis, x is false alarm per hour')
    parser.add_argument('--x_step', type=int, default=1, help='step on x-axis')
    parser.add_argument(
        '--ylim',
        type=int,
        default=35,
        help='ylim：range of y-axis, y is false rejection rate')
    parser.add_argument('--y_step', type=int, default=5, help='step on y-axis')

    args = parser.parse_args()
    window_shift = args.window_shift
    keywords_list = args.keywords.strip().split(',')

    token_table = read_token(args.token_file)
    lexicon_table = read_lexicon(args.lexicon_file)
    true_keywords = {}
    for keyword in keywords_list:
        strs, indexes = query_token_set(keyword, token_table, lexicon_table)
        true_keywords[keyword] = ''.join(strs)

    keyword_filler_table = load_label_and_score(keywords_list, args.test_data, args.score_file, true_keywords)

    for keyword in keywords_list:
        keyword = true_keywords[keyword]
        keyword = space_mixed_label(keyword)
        keyword_dur = keyword_filler_table[keyword]['keyword_duration']
        keyword_num = len(keyword_filler_table[keyword]['keyword_table'])
        filler_dur = keyword_filler_table[keyword]['filler_duration']
        filler_num = len(keyword_filler_table[keyword]['filler_table'])
        assert keyword_num > 0, 'Can\'t compute det for {} without positive sample'
        assert filler_num > 0, 'Can\'t compute det for {} without negative sample'

        logging.info('Computing det for {}'.format(keyword))
        logging.info('  Keyword duration: {} Hours, wave number: {}'.format(
            keyword_dur / 3600.0, keyword_num))
        logging.info('  Filler duration: {} Hours'.format(filler_dur / 3600.0))

        if args.stats_dir :
            stats_dir = args.stats_dir
        else:
            stats_dir = os.path.dirname(args.score_file)
        stats_file = os.path.join(stats_dir, 'stats.' + keyword.replace(' ', '_') + '.txt')
        with open(stats_file, 'w', encoding='utf8') as fout:
            threshold = 0.0
            while threshold <= 1.0:
                num_false_reject = 0
                num_true_detect = 0
                # transverse the all keyword_table
                for key, confi in keyword_filler_table[keyword]['keyword_table'].items():
                    if confi < threshold:
                        num_false_reject += 1
                    else:
                        num_true_detect += 1

                num_false_alarm = 0
                # transverse the all filler_table
                for key, confi in keyword_filler_table[keyword][
                    'filler_table'].items():
                    if confi >= threshold:
                        num_false_alarm += 1
                        # print(f'false alarm: {keyword}, {key}, {confi}')

                false_reject_rate = num_false_reject / keyword_num
                true_detect_rate = num_true_detect / keyword_num

                num_false_alarm = max(num_false_alarm, 1e-6)
                false_alarm_per_hour = num_false_alarm / (filler_dur / 3600.0)
                false_alarm_rate = num_false_alarm / filler_num

                fout.write('{:.3f} {:.6f} {:.6f}\n'.format(
                    threshold, false_alarm_per_hour, false_reject_rate))
                threshold += args.step
    if args.det_curve_path :
        det_curve_path = args.det_curve_path
    else:
        det_curve_path = os.path.join(stats_dir, 'det.png')
    plot_det(stats_dir, det_curve_path, args.xlim, args.x_step, args.ylim, args.y_step)
