# Copyright (c) 2021 Wenet Community. (authors: Binbin Zhang)
#               2023 Wenet Community. (authors: Dinghao Zhou)
#               2025 Wenet Community. (authors: Menglong Xu)
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

import copy
import torch

from functools import partial
from wenet.dataset.dataset import Dataset


def context_expansion(sample, left=1, right=1):
    """ expand left and right frames
        Args:
            sample: {key, feats, feats_lengths, ...}
            left (int): feature left context frames
            right (int): feature right context frames

        Returns:
            sample: {key, feats, feats_lengths, ...}
    """
    feats = sample['feats']  # (B, T, D)
    batch_size = feats.shape[0]
    ctx_frm = feats.shape[1]
    ctx_dim = feats.shape[2] * (left + right + 1)
    feats_ctx = torch.zeros(batch_size, ctx_frm, ctx_dim, dtype=torch.float32)
    index = 0
    for lag in range(-left, right + 1):
        feats_ctx[:, :, index:index + feats.shape[2]] = torch.roll(
            feats, -lag, 1)
        index = index + feats.shape[2]
    # replication pad left margin
    for idx in range(left):
        for cpx in range(left - idx):
            feats_ctx[:, idx, cpx * feats.shape[2]:(cpx + 1) * feats.shape[2]] = \
                feats_ctx[:, left, :feats.shape[2]]

    feats_ctx = feats_ctx[:, :feats_ctx.shape[1] - right]
    sample['feats'] = feats_ctx  # (B, T, D * n) where n = left + right + 1
    sample['feats_lengths'] = sample['feats_lengths'] - right
    return sample

def frame_skip(sample, skip_rate=1):
    """ skip frame
        Args:
            sample: {key, feats, feats_lengths, ...}
            skip_rate (int): take every N-frames for model input

        Returns:
            sample: {key, feats, feats_lengths, ...}
    """
    feats_skip = sample['feats'][:, ::skip_rate, :]
    sample['feats'] = feats_skip
    feats_lengths = torch.ceil(sample['feats_lengths'] / skip_rate)
    sample['feats_lengths'] = feats_lengths.to(dtype=torch.int16)
    return sample

def init_dataset(dataset_type: str = 'asr',
                 data_type: str = 'raw',
                 data_list_file=None,
                 tokenizer=None,
                 conf=None,
                 partition=True,
                 split='train'):
    assert dataset_type in ['asr']
    assert data_list_file is not None
    assert conf is not None

    if split != 'train':
        cv_conf = copy.deepcopy(conf)
        cv_conf['cycle'] = 1
        cv_conf['speed_perturb'] = False
        cv_conf['spec_aug'] = False
        cv_conf['spec_sub'] = False
        cv_conf['spec_trim'] = False
        cv_conf['shuffle'] = False
        cv_conf['list_shuffle'] = False
        conf = cv_conf

    dataset = Dataset(data_type, data_list_file, tokenizer, conf, partition)

    if conf.get('context_expansion', False):
        dataset = dataset.map(
            partial(context_expansion, **conf.get('context_expansion_conf', {})))

    if conf.get('frame_skip', 1) > 1:
        dataset = dataset.map(partial(frame_skip, skip_rate=conf.get('frame_skip')))

    return dataset
