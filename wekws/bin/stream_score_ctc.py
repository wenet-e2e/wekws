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
from collections import defaultdict
from torch.utils.data import DataLoader

from wekws.dataset.dataset import Dataset
from wekws.model.kws_model import init_model
from wekws.utils.checkpoint import load_checkpoint
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
                        default=1,
                        type=int,
                        help='batch size for inference')
    parser.add_argument('--num_workers',
                        default=1,
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
    parser.add_argument('--score_beam_size',
                        default=3,
                        type=int,
                        help='The first prune beam, f'
                             'ilter out those frames with low scores.')
    parser.add_argument('--path_beam_size',
                        default=20,
                        type=int,
                        help='The second prune beam, '
                             'keep only path_beam_size candidates.')
    parser.add_argument('--threshold',
                        type=float,
                        default=0.0,
                        help='The threshold of kws. '
                             'If ctc_search probs exceed this value,'
                             'the keyword will be activated.')
    parser.add_argument('--min_frames',
                        default=5,
                        type=int,
                        help='The min frames of keyword duration.')
    parser.add_argument('--max_frames',
                        default=250,
                        type=int,
                        help='The max frames of keyword duration.')

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

    downsampling_factor = test_conf.get('frame_skip', 1)

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
                # hyps = ctc_prefix_beam_search(score, lengths[i],
                #                               keywords_idxset)
                maxlen = score.size(0)
                ctc_probs = score
                cur_hyps = [(tuple(), (1.0, 0.0, []))]

                hit_keyword = None
                activated = False
                hit_score = 1.0
                start = 0
                end = 0

                # 2. CTC beam search step by step
                for t in range(0, maxlen):
                    probs = ctc_probs[t]  # (vocab_size,)
                    t *= downsampling_factor   # the real time
                    # key: prefix, value (pb, pnb), default value(-inf, -inf)
                    next_hyps = defaultdict(lambda: (0.0, 0.0, []))

                    # 2.1 First beam prune: select topk best
                    top_k_probs, top_k_index = probs.topk(args.score_beam_size)

                    # filter prob score that is too small
                    filter_probs = []
                    filter_index = []
                    for prob, idx in zip(
                            top_k_probs.tolist(), top_k_index.tolist()):
                        if keywords_idxset is not None:
                            if prob > 0.05 and idx in keywords_idxset:
                                filter_probs.append(prob)
                                filter_index.append(idx)
                        else:
                            if prob > 0.05:
                                filter_probs.append(prob)
                                filter_index.append(idx)

                    if len(filter_index) == 0:
                        continue

                    for s in filter_index:
                        ps = probs[s].item()

                        for prefix, (pb, pnb, cur_nodes) in cur_hyps:
                            last = prefix[-1] if len(prefix) > 0 else None
                            if s == 0:  # blank
                                n_pb, n_pnb, nodes = next_hyps[prefix]
                                n_pb = n_pb + pb * ps + pnb * ps
                                nodes = cur_nodes.copy()
                                next_hyps[prefix] = (n_pb, n_pnb, nodes)
                            elif s == last:
                                if not math.isclose(pnb, 0.0, abs_tol=0.000001):
                                    # Update *ss -> *s;
                                    n_pb, n_pnb, nodes = next_hyps[prefix]
                                    n_pnb = n_pnb + pnb * ps
                                    nodes = cur_nodes.copy()
                                    # update frame and prob
                                    if ps > nodes[-1]['prob']:
                                        nodes[-1]['prob'] = ps
                                        nodes[-1]['frame'] = t
                                    next_hyps[prefix] = (n_pb, n_pnb, nodes)

                                if not math.isclose(pb, 0.0, abs_tol=0.000001):
                                    # Update *s-s -> *ss, - is for blank
                                    n_prefix = prefix + (s,)
                                    n_pb, n_pnb, nodes = next_hyps[n_prefix]
                                    n_pnb = n_pnb + pb * ps
                                    nodes = cur_nodes.copy()
                                    nodes.append(dict(
                                        token=s, frame=t, prob=ps))
                                    next_hyps[n_prefix] = (n_pb, n_pnb, nodes)
                            else:
                                n_prefix = prefix + (s,)
                                n_pb, n_pnb, nodes = next_hyps[n_prefix]
                                if nodes:
                                    # update frame and prob
                                    if ps > nodes[-1]['prob']:
                                        # nodes[-1]['prob'] = ps
                                        # nodes[-1]['frame'] = t
                                        # avoid change other beam has this node.
                                        nodes.pop()
                                        nodes.append(dict(
                                            token=s, frame=t, prob=ps))
                                else:
                                    nodes = cur_nodes.copy()
                                    nodes.append(dict(
                                        token=s, frame=t, prob=ps))
                                n_pnb = n_pnb + pb * ps + pnb * ps
                                next_hyps[n_prefix] = (n_pb, n_pnb, nodes)

                    # 2.2 Second beam prune
                    next_hyps = sorted(
                        next_hyps.items(),
                        key=lambda x: (x[1][0] + x[1][1]), reverse=True)

                    cur_hyps = next_hyps[:args.path_beam_size]

                    hyps = [(y[0], y[1][0] + y[1][1], y[1][2])
                            for y in cur_hyps]

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
                                end = prefix_nodes[
                                    offset + len(lab) - 1]['frame']
                                for idx in range(offset, offset + len(lab)):
                                    hit_score *= prefix_nodes[idx]['prob']
                                break
                        if hit_keyword is not None:
                            hit_score = math.sqrt(hit_score)
                            break

                    duration = end - start
                    if hit_keyword is not None:
                        if hit_score >= args.threshold and \
                                args.min_frames <= duration <= args.max_frames:
                            activated = True
                            fout.write('{} detected {} {:.3f}\n'.format(
                                key, hit_keyword, hit_score))
                            logging.info(
                                f"batch:{batch_idx}_{i} detect {hit_keyword} "
                                f"in {key} from {start} to {end} frame. "
                                f"duration {duration}, s"
                                f"core {hit_score} Activated.")

                            # clear the ctc_prefix buffer, and hit_keyword
                            cur_hyps = [(tuple(), (1.0, 0.0, []))]
                            hit_keyword = None
                            hit_score = 1.0
                        elif hit_score < args.threshold:
                            logging.info(
                                f"batch:{batch_idx}_{i} detect {hit_keyword} "
                                f"in {key} from {start} to {end} frame. "
                                f"but {hit_score} less than "
                                f"{args.threshold}, Deactivated. ")
                        elif args.min_frames > duration \
                                or duration > args.max_frames:
                            logging.info(
                                f"batch:{batch_idx}_{i} detect {hit_keyword} "
                                f"in {key} from {start} to {end} frame. "
                                f"but {duration} beyond "
                                f"range({args.min_frames}~{args.max_frames}), "
                                f"Deactivated. ")
                if not activated:
                    fout.write('{} rejected\n'.format(key))
                    logging.info(f"batch:{batch_idx}_{i} {key} Deactivated.")

            if batch_idx % 10 == 0:
                print('Progress batch {}'.format(batch_idx))
                sys.stdout.flush()


if __name__ == '__main__':
    main()
