# Copyright (c) 2021 Jingyong Hou (houjingyong@gmail.com)
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

import numpy as np
import torch


def RHE(indice: torch.Tensor, k: int):
    """Regional hard example mining from 'Mining effective negative training samples for keyword spotting'

    Attributes:
        index: indice of
        k:
        lengths: (B)
        min_duration: min duration of the keyword
    Returns:
        (torch.Tensor): indice of selected regional hard example
    """
    if k <= 0:
        return indice
    lenght = len(indice)
    available_indice = torch.tensor([1] * (lenght))
    reserve = []
    for i in range(lenght):
        if 1 == available_indice[indice[i]]:
            reserve.append(indice[i])
            rm_s = max(indice[i] - k, 0)
            rm_e = min(indice[i] + k, lenght)
            available_indice[rm_s : rm_e + 1] = 0
        else:
            continue

        if torch.sum(available_indice) <= 0:
            break
    return torch.tensor(reserve).long()


def downsample_training_sample_and_calculate_loss(logits, targets, ratio: float = 10):
    num_training = 0
    loss = 0
    for i in range(len(logits)):
        output = torch.cat(logits[i])
        target = torch.LongTensor(np.concatenate(targets[i]))
        # how many positive targets
        positive_index = target >= 1  # the label of positive label is 1
        negative_index = target < 1  # the label of negative label is 0
        num_p = torch.sum(positive_index)
        selected_p_output = output[positive_index]
        loss += torch.sum(torch.log(selected_p_output))

        all_n_output = output[negative_index]
        num_n = min(int(ratio * num_p), len(all_n_output))
        _, sorted_index = torch.sort(all_n_output, descending=True)
        selected_n_output = all_n_output[sorted_index[:num_n]]
        num_training += len(selected_p_output) + len(selected_n_output)
    return loss / num_training


def max_pooling_RHE_binary_CE(logits, targets, lengths, RHE_thr=10000, max_ratio=1):

    """Max-pooling loss with regional hard example mining
        For each keyword utterance, select the frame with the highest posterior.
            The keyword is triggered when any of the frames is triggered.
        For each non-keyword utterance, select several hard examples using the RHE algorithm.

    Attributes:
        logits: (B, T, D), D is the number of keywords
        target: (B)
        lengths: (B)
        RHE_thr: how many neighbor logits we remove each time we find a hard examle
    Returns:
        (float): loss of current batch
        (float): accuracy of current batch
    """
    num_hit = 0
    # Here we clamp the sigmoid output to prevent NaN problem
    # When we calculate loss
    logits = torch.clamp(torch.sigmoid(logits), 1e-8, 1.0 - 1e-8)
    num_utts = logits.size(0)
    num_keyword = logits.size(2)

    new_logits = []
    new_targets = []
    for j in range(num_keyword):
        new_logits.append([])
        new_targets.append([])

    for i in range(num_utts):
        end_idx = lengths[i]
        for j in range(num_keyword):
            if targets[i] == j:
                max_idx = logits[i, :end_idx].argmax()
                new_logits[j].append(logits[i, max_idx, j])
                new_targets[j].append([1])
                if logits[i, max_idx, j] >= 0.5:
                    num_hit += 1
            else:
                sorted_logits, sorted_index = torch.sort(logits[i, :end_idx], dim=0)
                reversed_index = torch.flip(sorted_index, dims=[0])
                selected_indexes = RHE(reversed_index[:, j], RHE_thr)
                new_logits[j].append(logits[i, selected_indexes, j])
                new_targets[j].append([0] * len(selected_indexes))
                if torch.sum(sorted_logits[-1, :] >= 0.5) <= 0:
                    # all the binary probilities are smaller than 0.5
                    num_hit += 1

    # Here we select training samples acorrding to max_ratio
    loss = downsample_training_sample_and_calculate_loss(
        new_logits,
        new_targets,
        ratio=max_ratio,
    )
    acc = num_hit / num_utts
    return loss, acc


if __name__ == '__main__':
    index = torch.tensor([3, 2, 0, 7, 5, 8, 1, 4, 6])
    print(RHE(index, 0))  # [3, 2, 0, 7, 5, 8, 1, 4, 6]
    print(RHE(index, 1))  # [3, 0, 7, 5 ]
    print(RHE(index, 2))  # [3, 0, 7]
    print(RHE(index, 3))  # [3, 7]
    print(RHE(index, 100))  # [3]
