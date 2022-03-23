# Copyright (c) 2021 Binbin Zhang(binbzha@qq.com)
#               2022 Shaoqing Yu(yu954793264@163.com)
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

from kws.dataset.dataset import Dataset
from kws.model.kws_model import init_model
from kws.utils.checkpoint import load_checkpoint


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
    parser.add_argument('--score_file_dir',
                        required=True,
                        help='output score file')
    parser.add_argument('--num_keywords',
                        required=True,
                        help='the number of keywords')
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
    # add to write different keyword score file
    num_keywords = int(args.num_keywords)
    score_file_list = []
    dir_abs_path = os.path.abspath(args.score_file_dir)
    for i in range(num_keywords):
        temp_list = ['score_longwav', 'txt']
        temp_list.insert(1, str(i))
        suffix = '.'.join(temp_list)
        # print('suffix = ', suffix)
        score_abs_path = os.path.join(dir_abs_path, suffix)
        score_file_list.append(score_abs_path)

    for abs_path in score_file_list:
        with torch.no_grad(), open(abs_path, 'w', encoding='utf8') as fout:
            keyword_label = abs_path.split('/')[-1].split('.')[1]
            # print('keyword_label = ', keyword_label)
            for batch_idx, batch in enumerate(test_data_loader):
                keys, feats, target, lengths = batch
                feats = feats.to(device)
                lengths = lengths.to(device)
                # mask = padding_mask(lengths).unsqueeze(2)
                logits = model(feats)
                # mask对应的true的部分用0填充
                # Getting every frames desn't need to mask
                # logits = logits.masked_fill(mask, 0.0)
                logits = logits.cpu()
                for i in range(len(keys)):
                    key = keys[i]
                    score = logits[i][:lengths[i]]
                    score = score[:, int(keyword_label)]
                    # keep 2 significant digits
                    score = ' '.join([str("%.2g" % x) for x in score.tolist()])
                    fout.write('{} {}\n'.format(key, score))
                if batch_idx % 10 == 0:
                    print('Progress batch {}'.format(batch_idx))
                    sys.stdout.flush()


if __name__ == '__main__':
    main()
