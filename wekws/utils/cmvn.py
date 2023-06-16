#!/usr/bin/env python3
# Copyright (c) 2020 Binbin Zhang
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

import json
import math
import re

import numpy as np


def load_cmvn(json_cmvn_file):
    """ Load the json format cmvn stats file and calculate cmvn

    Args:
        json_cmvn_file: cmvn stats file in json format

    Returns:
        a numpy array of [means, vars]
    """
    with open(json_cmvn_file) as f:
        cmvn_stats = json.load(f)

    means = cmvn_stats['mean_stat']
    variance = cmvn_stats['var_stat']
    count = cmvn_stats['frame_num']
    for i in range(len(means)):
        means[i] /= count
        variance[i] = variance[i] / count - means[i] * means[i]
        if variance[i] < 1.0e-20:
            variance[i] = 1.0e-20
        variance[i] = 1.0 / math.sqrt(variance[i])
    cmvn = np.array([means, variance])
    return cmvn

def load_kaldi_cmvn(cmvn_file):
    """ Load the kaldi format cmvn stats file and no need to calculate

    Args:
        cmvn_file: cmvn stats file in kaldi format

    Returns:
        a numpy array of [means, vars]
    """

    means = None
    variance = None
    with open(cmvn_file) as f:
        all_lines = f.readlines()
        for idx, line in enumerate(all_lines):
            if line.find('AddShift') != -1:
                segs = line.strip().split(' ')
                assert len(segs) == 3
                next_line = all_lines[idx + 1]
                means_str = re.findall(r'[\[](.*?)[\]]', next_line)[0]
                means_list = means_str.strip().split(' ')
                means = [0 - float(s) for s in means_list]
                assert len(means) == int(segs[1])
            elif line.find('Rescale') != -1:
                segs = line.strip().split(' ')
                assert len(segs) == 3
                next_line = all_lines[idx + 1]
                vars_str = re.findall(r'[\[](.*?)[\]]', next_line)[0]
                vars_list = vars_str.strip().split(' ')
                variance = [float(s) for s in vars_list]
                assert len(variance) == int(segs[1])
            elif line.find('Splice') != -1:
                segs = line.strip().split(' ')
                assert len(segs) == 3
                next_line = all_lines[idx + 1]
                splice_str = re.findall(r'[\[](.*?)[\]]', next_line)[0]
                splice_list = splice_str.strip().split(' ')
                assert len(splice_list) * int(segs[2]) == int(segs[1])
                copy_times = len(splice_list)
            else:
                continue

    cmvn = np.array([means, variance])
    cmvn = np.tile(cmvn, (1, copy_times))

    return cmvn
