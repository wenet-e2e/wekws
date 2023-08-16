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

from __future__ import print_function

import argparse
import copy
import logging
import os
import sys
import math

import torch
import yaml
from torch.utils.data import DataLoader

from wekws.dataset.dataset import Dataset
from wekws.model.kws_model import init_model
from wekws.utils.checkpoint import load_checkpoint
from wekws.model.loss import ctc_prefix_beam_search
from tools.make_list import query_token_set, read_lexicon, read_token

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
    parser.add_argument('--keywords', type=str, default=None,
                        help='the keywords, split with comma(,)')
    parser.add_argument('--token_file', type=str, default=None,
                        help='the path of tokens.txt')
    parser.add_argument('--lexicon_file', type=str, default=None,
                        help='the path of lexicon.txt')

    args = parser.parse_args()
    return args

def is_sublist(main_list, check_list):
    if len(main_list) < len(check_list):
        return -1

    if len(main_list) == len(check_list):
        return 0 if main_list == check_list else -1

    for i in range(len(main_list) - len(check_list)):
        if main_list[i] == check_list[0]:
            for j in range(len(check_list)):
                if main_list[i + j] != check_list[j]:
                    break
            else:
                return i
    else:
        return -1


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

    token_table = read_token(args.token_file)
    lexicon_table = read_lexicon(args.lexicon_file)
    # 4. parse keywords tokens
    assert args.keywords is not None, 'at least one keyword is needed'
    logging.info(f"keywords is {args.keywords}, "
                 f"Chinese is converted into Unicode.")
    keywords_str = args.keywords.encode('utf-8').decode('unicode_escape')
    keywords_list = keywords_str.strip().replace(' ', '').split(',')
    keywords_token = {}
    keywords_idxset = {0}
    keywords_strset = {'<blk>'}
    keywords_tokenmap = {'<blk>': 0}
    for keyword in keywords_list:
        strs, indexes = query_token_set(keyword, token_table, lexicon_table)
        keywords_token[keyword] = {}
        keywords_token[keyword]['token_id'] = indexes
        keywords_token[keyword]['token_str'] = ''.join('%s ' % str(i)
                                                       for i in indexes)
        [keywords_strset.add(i) for i in strs]
        [keywords_idxset.add(i) for i in indexes]
        for txt, idx in zip(strs, indexes):
            if keywords_tokenmap.get(txt, None) is None:
                keywords_tokenmap[txt] = idx

    token_print = ''
    for txt, idx in keywords_tokenmap.items():
        token_print += f'{txt}({idx}) '
    logging.info(f'Token set is: {token_print}')

    with torch.no_grad(), open(score_abs_path, 'w', encoding='utf8') as fout:
        for batch_idx, batch in enumerate(test_data_loader):
            keys, feats, target, lengths, target_lengths = batch
            feats = feats.to(device)
            lengths = lengths.to(device)
            logits, _ = model(feats)
            logits = logits.softmax(2)  # (batch_size, maxlen, vocab_size)
            logits = logits.cpu()
            for i in range(len(keys)):
                key = keys[i]
                score = logits[i][:lengths[i]]
                hyps = ctc_prefix_beam_search(score,
                                              lengths[i],
                                              keywords_idxset)
                hit_keyword = None
                hit_score = 1.0
                start = 0
                end = 0
                for one_hyp in hyps:
                    prefix_ids = one_hyp[0]
                    # path_score = one_hyp[1]
                    prefix_nodes = one_hyp[2]
                    assert len(prefix_ids) == len(prefix_nodes)
                    for word in keywords_token.keys():
                        lab = keywords_token[word]['token_id']
                        offset = is_sublist(prefix_ids, lab)
                        if offset != -1:
                            hit_keyword = word
                            start = prefix_nodes[offset]['frame']
                            end = prefix_nodes[offset + len(lab) - 1]['frame']
                            for idx in range(offset, offset + len(lab)):
                                hit_score *= prefix_nodes[idx]['prob']
                            break
                    if hit_keyword is not None:
                        hit_score = math.sqrt(hit_score)
                        break

                if hit_keyword is not None:
                    fout.write('{} detected {} {:.3f}\n'.format(
                        key, hit_keyword, hit_score))
                    logging.info(
                        f"batch:{batch_idx}_{i} detect {hit_keyword} "
                        f"in {key} from {start} to {end} frame. "
                        f"duration {end - start}, "
                        f"score {hit_score}, Activated.")
                else:
                    fout.write('{} rejected\n'.format(key))
                    logging.info(f"batch:{batch_idx}_{i} {key} Deactivated.")

            if batch_idx % 10 == 0:
                print('Progress batch {}'.format(batch_idx))
                sys.stdout.flush()


if __name__ == '__main__':
    main()
