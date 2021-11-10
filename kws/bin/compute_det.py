# Copyright (c) 2021 Binbin Zhang(binbzha@qq.com)
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
import json


def load_label_and_score(keyword, label_file, score_file):
    score_table = {}
    with open(score_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            key = arr[0]
            score = float(arr[keyword + 1])
            score_table[key] = score
    keyword_table = {}
    filler_table = {}
    filler_duration = 0.0
    with open(label_file, 'r', encoding='utf8') as fin:
        for line in fin:
            obj = json.loads(line.strip())
            assert 'key' in obj
            assert 'txt' in obj
            assert 'duration' in obj
            key = obj['key']
            index = obj['txt']
            duration = obj['duration']
            assert key in score_table
            if index == keyword:
                keyword_table[key] = score_table[key]
            else:
                filler_table[key] = score_table[key]
                filler_duration += duration
    return keyword_table, filler_table, filler_duration


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compute det curve')
    parser.add_argument('--test_data', required=True, help='label file')
    parser.add_argument('--keyword', type=int, default=0, help='score file')
    parser.add_argument('--score_file', required=True, help='score file')
    parser.add_argument('--step', type=float, default=0.01, help='score file')
    parser.add_argument('--stats_file',
                        required=True,
                        help='false reject/alarm stats file')
    args = parser.parse_args()

    keyword_table, filler_table, filler_duration = load_label_and_score(
        args.keyword, args.test_data, args.score_file)
    print('Filler total duration Hours: {}'.format(filler_duration / 3600.0))

    with open(args.stats_file, 'w', encoding='utf8') as fout:
        threshold = 0.0
        while threshold <= 1.0:
            num_false_reject = 0
            for key, score in keyword_table.items():
                if score < threshold:
                    num_false_reject += 1
            num_false_alarm = 0
            for key, score in filler_table.items():
                if score >= threshold:
                    num_false_alarm += 1
            false_reject_rate = num_false_reject / len(keyword_table)
            num_false_alarm = max(num_false_alarm, 1e-6)
            false_alarm_per_hour = num_false_alarm / (filler_duration / 3600.0)
            fout.write('{:.6f} {:.6f} {:.6f}\n'.format(threshold,
                                                       false_alarm_per_hour,
                                                       false_reject_rate))
            threshold += args.step
