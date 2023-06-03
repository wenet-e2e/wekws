# Copyright (c) 2021 Binbin Zhang(binbzha@qq.com)
#               2022 Shaoqing Yu(954793264@qq.com)
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

from __future__ import print_function

import argparse
import copy
import logging
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

from wekws.dataset.dataset import Dataset
from wekws.model.kws_model import init_model
from wekws.utils.checkpoint import load_checkpoint


def get_args():
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--batch_size',
                        default=16,
                        type=int,
                        help='batch size for inference')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='prefetch number')
    parser.add_argument('--score_file',
                        required=True,
                        help='output score file')
    parser.add_argument('--jit_model',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    test_conf = copy.deepcopy(configs['dataset_conf'])
    test_conf['filter_conf']['max_length'] = 102400
    test_conf['filter_conf']['min_length'] = 0
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['shuffle'] = False
    test_conf['feature_extraction_conf']['dither'] = 0.0
    test_conf['batch_conf']['batch_size'] = args.batch_size

    test_dataset = Dataset(args.test_data, test_conf)
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=None,
                                  pin_memory=args.pin_memory,
                                  num_workers=args.num_workers,
                                  prefetch_factor=args.prefetch)

    if args.jit_model:
        model = torch.jit.load(args.checkpoint)
        # For script model, only cpu is supported.
        device = torch.device('cpu')
    else:
        # Init asr model from configs
        model = init_model(configs['model'])
        load_checkpoint(model, args.checkpoint)
        use_cuda = args.gpu >= 0 and torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)
    model.eval()
    score_abs_path = os.path.abspath(args.score_file)
    with torch.no_grad(), open(score_abs_path, 'w', encoding='utf8') as fout:
        for batch_idx, batch in enumerate(test_data_loader):
            keys, feats, target, lengths, target_lengths = batch
            feats = feats.to(device)
            lengths = lengths.to(device)
            logits, _ = model(feats)
            num_keywords = logits.shape[2]
            logits = logits.cpu()
            for i in range(len(keys)):
                key = keys[i]
                score = logits[i][:lengths[i]]
                for keyword_i in range(num_keywords):
                    keyword_scores = score[:, keyword_i]
                    score_frames = ' '.join(['{:.6f}'.format(x)
                                            for x in keyword_scores.tolist()])
                    fout.write('{} {} {}\n'.format(
                        key, keyword_i, score_frames))
            if batch_idx % 10 == 0:
                print('Progress batch {}'.format(batch_idx))
                sys.stdout.flush()


if __name__ == '__main__':
    main()
