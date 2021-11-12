# Copyright (c) 2021 Jingyong Hou
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
import torch.nn as nn


def acc_frame(
    logits: torch.Tensor,
    target: torch.Tensor,
):
    if logits is None:
        return 0
    pred = logits.max(1, keepdim=True)[1]
    correct = pred.eq(target.long().view_as(pred)).sum().item()
    return correct * 100.0 / logits.size(0)


def cross_entropy(logits: torch.Tensor,
                  target: torch.Tensor,
                  lengths: torch.Tensor,
                  min_duration: int = 0):
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
    cross_entropy = nn.CrossEntropyLoss()
    loss = cross_entropy(logits, target)
    acc = acc_frame(logits, target)
    return loss, acc
