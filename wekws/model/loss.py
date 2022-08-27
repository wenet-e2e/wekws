# Copyright (c) 2021 Binbin Zhang
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
import torch.nn.functional as F

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
    loss = F.cross_entropy(logits, target)
    acc = acc_frame(logits, target)
    return loss, acc


def criterion(type: str,
              logits: torch.Tensor,
              target: torch.Tensor,
              lengths: torch.Tensor,
              min_duration: int = 0):
    if type == 'ce':
        loss, acc = cross_entropy(logits, target)
        return loss, acc
    elif type == 'max_pooling':
        loss, acc = max_pooling_loss(logits, target, lengths, min_duration)
        return loss, acc
    else:
        exit(1)
