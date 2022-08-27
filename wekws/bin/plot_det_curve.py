# Copyright (c) 2021 Binbin Zhang(binbzha@qq.com)
#                    Menglong Xu
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
import os
import numpy as np
import matplotlib.pyplot as plt


def load_stats_file(stats_file):
    values = []
    with open(stats_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            threshold, fa_per_hour, frr = arr
            values.append([float(fa_per_hour), float(frr) * 100])
    values.reverse()
    return np.array(values)


def plot_det_curve(
        keywords,
        stats_dir,
        figure_file,
        xlim,
        x_step,
        ylim,
        y_step):
    plt.figure(dpi=200)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.size'] = 12

    for index, keyword in enumerate(keywords):
        stats_file = os.path.join(stats_dir, 'stats.' + str(index) + '.txt')
        values = load_stats_file(stats_file)
        plt.plot(values[:, 0], values[:, 1], label=keyword)

    plt.xlim([0, xlim])
    plt.ylim([0, ylim])
    plt.xticks(range(0, xlim + x_step, x_step))
    plt.yticks(range(0, ylim + y_step, y_step))
    plt.xlabel('False Alarm Per Hour')
    plt.ylabel('False Rejection Rate (\\%)')
    plt.grid(linestyle='--')
    plt.legend(loc='best', fontsize=16)
    plt.savefig(figure_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plot det curve')
    parser.add_argument(
        '--keywords_dict',
        required=True,
        help='path to the dictionary of keywords')
    parser.add_argument('--stats_dir', required=True, help='dir of stats files')
    parser.add_argument(
        '--figure_file',
        required=True,
        help='path to save det curve')
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

    keywords = []
    with open(args.keywords_dict, 'r', encoding='utf8') as fin:
        for line in fin:
            keyword, index = line.strip().split()
            if int(index) > -1:
                keywords.append(keyword)

    plot_det_curve(
        keywords,
        args.stats_dir,
        args.figure_file,
        args.xlim,
        args.x_step,
        args.ylim,
        args.y_step)
