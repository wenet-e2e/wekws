# Copyright (c) 2021 Binbin Zhang
#               2023 Jing Du
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

import torch
import math
import sys
import torch.nn.functional as F
from collections import defaultdict
from typing import List, Tuple

from wekws.utils.mask import padding_mask


def max_pooling_loss(logits: torch.Tensor,
                     target: torch.Tensor,
                     lengths: torch.Tensor,
                     min_duration: int = 0):
    ''' Max-pooling loss
        For keyword, select the frame with the highest posterior.
            The keyword is triggered when any of the frames is triggered.
        For none keyword, select the hardest frame, namely the frame
            with lowest filler posterior(highest keyword posterior).
            the keyword is not triggered when all frames are not triggered.

    Attributes:
        logits: (B, T, D), D is the number of keywords
        target: (B)
        lengths: (B)
        min_duration: min duration of the keyword
    Returns:
        (float): loss of current batch
        (float): accuracy of current batch
    '''
    mask = padding_mask(lengths)
    num_utts = logits.size(0)
    num_keywords = logits.size(2)

    target = target.cpu()
    loss = 0.0
    for i in range(num_utts):
        for j in range(num_keywords):
            # Add entropy loss CE = -(t * log(p) + (1 - t) * log(1 - p))
            if target[i] == j:
                # For the keyword, do max-polling
                prob = logits[i, :, j]
                m = mask[i].clone().detach()
                m[:min_duration] = True
                prob = prob.masked_fill(m, 0.0)
                prob = torch.clamp(prob, 1e-8, 1.0)
                max_prob = prob.max()
                loss += -torch.log(max_prob)
            else:
                # For other keywords or filler, do min-polling
                prob = 1 - logits[i, :, j]
                prob = prob.masked_fill(mask[i], 1.0)
                prob = torch.clamp(prob, 1e-8, 1.0)
                min_prob = prob.min()
                loss += -torch.log(min_prob)
    loss = loss / num_utts

    # Compute accuracy of current batch
    mask = mask.unsqueeze(-1)
    logits = logits.masked_fill(mask, 0.0)
    max_logits, index = logits.max(1)
    num_correct = 0
    for i in range(num_utts):
        max_p, idx = max_logits[i].max(0)
        # Predict correct as the i'th keyword
        if max_p > 0.5 and idx == target[i]:
            num_correct += 1
        # Predict correct as the filler, filler id < 0
        if max_p < 0.5 and target[i] < 0:
            num_correct += 1
    acc = num_correct / num_utts
    # acc = 0.0
    return loss, acc


def acc_frame(
    logits: torch.Tensor,
    target: torch.Tensor,
):
    if logits is None:
        return 0
    pred = logits.max(1, keepdim=True)[1]
    correct = pred.eq(target.long().view_as(pred)).sum().item()
    return correct * 100.0 / logits.size(0)

def acc_utterance(logits: torch.Tensor, target: torch.Tensor,
                  logits_length: torch.Tensor, target_length: torch.Tensor):
    if logits is None:
        return 0

    logits = logits.softmax(2)  # (1, maxlen, vocab_size)
    logits = logits.cpu()
    target = target.cpu()

    total_word = 0
    total_ins = 0
    total_sub = 0
    total_del = 0
    calculator = Calculator()
    for i in range(logits.size(0)):
        score = logits[i][:logits_length[i]]
        hyps = ctc_prefix_beam_search(score, logits_length[i], None, 3, 5)
        lab = [str(item) for item in target[i][:target_length[i]].tolist()]
        rec = []
        if len(hyps) > 0:
            rec = [str(item) for item in hyps[0][0]]
        result = calculator.calculate(lab, rec)
        # print(f'result:{result}')
        if result['all'] != 0:
            total_word += result['all']
            total_ins += result['ins']
            total_sub += result['sub']
            total_del += result['del']

    return float(total_word - total_ins - total_sub
                 - total_del) * 100.0 / total_word

def ctc_loss(logits: torch.Tensor,
             target: torch.Tensor,
             logits_lengths: torch.Tensor,
             target_lengths: torch.Tensor,
             need_acc: bool = False):
    """ CTC Loss
    Args:
        logits: (B, D), D is the number of keywords plus 1 (non-keyword)
        target: (B)
        logits_lengths: (B)
        target_lengths: (B)
    Returns:
        (float): loss of current batch
    """

    acc = 0.0
    if need_acc:
        acc = acc_utterance(logits, target, logits_lengths, target_lengths)

    # logits: (B, L, D) -> (L, B, D)
    logits = logits.transpose(0, 1)
    logits = logits.log_softmax(2)
    loss = F.ctc_loss(
        logits, target, logits_lengths, target_lengths, reduction='sum')
    loss = loss / logits.size(1)  # batch mean

    return loss, acc

def cross_entropy(logits: torch.Tensor, target: torch.Tensor):
    """ Cross Entropy Loss
    Attributes:
        logits: (B, D), D is the number of keywords plus 1 (non-keyword)
        target: (B)
        lengths: (B)
        min_duration: min duration of the keyword
    Returns:
        (float): loss of current batch
        (float): accuracy of current batch
    """
    loss = F.cross_entropy(logits, target.type(torch.int64))
    acc = acc_frame(logits, target)
    return loss, acc


def criterion(type: str,
              logits: torch.Tensor,
              target: torch.Tensor,
              lengths: torch.Tensor,
              target_lengths: torch.Tensor = None,
              min_duration: int = 0,
              validation: bool = False, ):
    if type == 'ce':
        loss, acc = cross_entropy(logits, target)
        return loss, acc
    elif type == 'max_pooling':
        loss, acc = max_pooling_loss(logits, target, lengths, min_duration)
        return loss, acc
    elif type == 'ctc':
        loss, acc = ctc_loss(
            logits, target, lengths, target_lengths, validation)
        return loss, acc
    else:
        exit(1)

def ctc_prefix_beam_search(
    logits: torch.Tensor,
    logits_lengths: torch.Tensor,
    keywords_tokenset: set = None,
    score_beam_size: int = 3,
    path_beam_size: int = 20,
) -> Tuple[List[List[int]], torch.Tensor]:
    """ CTC prefix beam search inner implementation

    Args:
        logits (torch.Tensor): (1, max_len, vocab_size)
        logits_lengths (torch.Tensor): (1, )
        keywords_tokenset (set): token set for filtering score
        score_beam_size (int): beam size for score
        path_beam_size (int): beam size for path

    Returns:
        List[List[int]]: nbest results
    """
    maxlen = logits.size(0)
    # ctc_probs = logits.softmax(1)  # (1, maxlen, vocab_size)
    ctc_probs = logits

    cur_hyps = [(tuple(), (1.0, 0.0, []))]

    # 2. CTC beam search step by step
    for t in range(0, maxlen):
        probs = ctc_probs[t]  # (vocab_size,)
        # key: prefix, value (pb, pnb), default value(-inf, -inf)
        next_hyps = defaultdict(lambda: (0.0, 0.0, []))

        # 2.1 First beam prune: select topk best
        top_k_probs, top_k_index = probs.topk(
            score_beam_size)  # (score_beam_size,)

        # filter prob score that is too small
        filter_probs = []
        filter_index = []
        for prob, idx in zip(top_k_probs.tolist(), top_k_index.tolist()):
            if keywords_tokenset is not None:
                if prob > 0.05 and idx in keywords_tokenset:
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
                        if ps > nodes[-1]['prob']:  # update frame and prob
                            nodes[-1]['prob'] = ps
                            nodes[-1]['frame'] = t
                        next_hyps[prefix] = (n_pb, n_pnb, nodes)

                    if not math.isclose(pb, 0.0, abs_tol=0.000001):
                        # Update *s-s -> *ss, - is for blank
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb, nodes = next_hyps[n_prefix]
                        n_pnb = n_pnb + pb * ps
                        nodes = cur_nodes.copy()
                        nodes.append(dict(token=s, frame=t,
                                          prob=ps))  # to record token prob
                        next_hyps[n_prefix] = (n_pb, n_pnb, nodes)
                else:
                    n_prefix = prefix + (s, )
                    n_pb, n_pnb, nodes = next_hyps[n_prefix]
                    if nodes:
                        if ps > nodes[-1]['prob']:  # update frame and prob
                            # nodes[-1]['prob'] = ps
                            # nodes[-1]['frame'] = t
                            # avoid change other beam which has this node.
                            nodes.pop()
                            nodes.append(dict(token=s, frame=t, prob=ps))
                    else:
                        nodes = cur_nodes.copy()
                        nodes.append(dict(token=s, frame=t,
                                          prob=ps))  # to record token prob
                    n_pnb = n_pnb + pb * ps + pnb * ps
                    next_hyps[n_prefix] = (n_pb, n_pnb, nodes)

        # 2.2 Second beam prune
        next_hyps = sorted(
            next_hyps.items(), key=lambda x: (x[1][0] + x[1][1]), reverse=True)

        cur_hyps = next_hyps[:path_beam_size]

    hyps = [(y[0], y[1][0] + y[1][1], y[1][2]) for y in cur_hyps]
    return hyps


class Calculator:

    def __init__(self):
        self.data = {}
        self.space = []
        self.cost = {}
        self.cost['cor'] = 0
        self.cost['sub'] = 1
        self.cost['del'] = 1
        self.cost['ins'] = 1

    def calculate(self, lab, rec):
        # Initialization
        lab.insert(0, '')
        rec.insert(0, '')
        while len(self.space) < len(lab):
            self.space.append([])
        for row in self.space:
            for element in row:
                element['dist'] = 0
                element['error'] = 'non'
            while len(row) < len(rec):
                row.append({'dist': 0, 'error': 'non'})
        for i in range(len(lab)):
            self.space[i][0]['dist'] = i
            self.space[i][0]['error'] = 'del'
        for j in range(len(rec)):
            self.space[0][j]['dist'] = j
            self.space[0][j]['error'] = 'ins'
        self.space[0][0]['error'] = 'non'
        for token in lab:
            if token not in self.data and len(token) > 0:
                self.data[token] = {
                    'all': 0,
                    'cor': 0,
                    'sub': 0,
                    'ins': 0,
                    'del': 0
                }
        for token in rec:
            if token not in self.data and len(token) > 0:
                self.data[token] = {
                    'all': 0,
                    'cor': 0,
                    'sub': 0,
                    'ins': 0,
                    'del': 0
                }
        # Computing edit distance
        for i, lab_token in enumerate(lab):
            for j, rec_token in enumerate(rec):
                if i == 0 or j == 0:
                    continue
                min_dist = sys.maxsize
                min_error = 'none'
                dist = self.space[i - 1][j]['dist'] + self.cost['del']
                error = 'del'
                if dist < min_dist:
                    min_dist = dist
                    min_error = error
                dist = self.space[i][j - 1]['dist'] + self.cost['ins']
                error = 'ins'
                if dist < min_dist:
                    min_dist = dist
                    min_error = error
                if lab_token == rec_token:
                    dist = self.space[i - 1][j - 1]['dist'] + self.cost['cor']
                    error = 'cor'
                else:
                    dist = self.space[i - 1][j - 1]['dist'] + self.cost['sub']
                    error = 'sub'
                if dist < min_dist:
                    min_dist = dist
                    min_error = error
                self.space[i][j]['dist'] = min_dist
                self.space[i][j]['error'] = min_error
        # Tracing back
        result = {
            'lab': [],
            'rec': [],
            'all': 0,
            'cor': 0,
            'sub': 0,
            'ins': 0,
            'del': 0
        }
        i = len(lab) - 1
        j = len(rec) - 1
        while True:
            if self.space[i][j]['error'] == 'cor':  # correct
                if len(lab[i]) > 0:
                    self.data[lab[i]]['all'] = self.data[lab[i]]['all'] + 1
                    self.data[lab[i]]['cor'] = self.data[lab[i]]['cor'] + 1
                    result['all'] = result['all'] + 1
                    result['cor'] = result['cor'] + 1
                result['lab'].insert(0, lab[i])
                result['rec'].insert(0, rec[j])
                i = i - 1
                j = j - 1
            elif self.space[i][j]['error'] == 'sub':  # substitution
                if len(lab[i]) > 0:
                    self.data[lab[i]]['all'] = self.data[lab[i]]['all'] + 1
                    self.data[lab[i]]['sub'] = self.data[lab[i]]['sub'] + 1
                    result['all'] = result['all'] + 1
                    result['sub'] = result['sub'] + 1
                result['lab'].insert(0, lab[i])
                result['rec'].insert(0, rec[j])
                i = i - 1
                j = j - 1
            elif self.space[i][j]['error'] == 'del':  # deletion
                if len(lab[i]) > 0:
                    self.data[lab[i]]['all'] = self.data[lab[i]]['all'] + 1
                    self.data[lab[i]]['del'] = self.data[lab[i]]['del'] + 1
                    result['all'] = result['all'] + 1
                    result['del'] = result['del'] + 1
                result['lab'].insert(0, lab[i])
                result['rec'].insert(0, '')
                i = i - 1
            elif self.space[i][j]['error'] == 'ins':  # insertion
                if len(rec[j]) > 0:
                    self.data[rec[j]]['ins'] = self.data[rec[j]]['ins'] + 1
                    result['ins'] = result['ins'] + 1
                result['lab'].insert(0, '')
                result['rec'].insert(0, rec[j])
                j = j - 1
            elif self.space[i][j]['error'] == 'non':  # starting point
                break
            else:  # shouldn't reach here
                print(
                    'this should not happen, '
                    'i = {i} , j = {j} , error = {error}'
                    .format(i=i, j=j, error=self.space[i][j]['error']))
        return result

    def overall(self):
        result = {'all': 0, 'cor': 0, 'sub': 0, 'ins': 0, 'del': 0}
        for token in self.data:
            result['all'] = result['all'] + self.data[token]['all']
            result['cor'] = result['cor'] + self.data[token]['cor']
            result['sub'] = result['sub'] + self.data[token]['sub']
            result['ins'] = result['ins'] + self.data[token]['ins']
            result['del'] = result['del'] + self.data[token]['del']
        return result

    def cluster(self, data):
        result = {'all': 0, 'cor': 0, 'sub': 0, 'ins': 0, 'del': 0}
        for token in data:
            if token in self.data:
                result['all'] = result['all'] + self.data[token]['all']
                result['cor'] = result['cor'] + self.data[token]['cor']
                result['sub'] = result['sub'] + self.data[token]['sub']
                result['ins'] = result['ins'] + self.data[token]['ins']
                result['del'] = result['del'] + self.data[token]['del']
        return result

    def keys(self):
        return list(self.data.keys())
