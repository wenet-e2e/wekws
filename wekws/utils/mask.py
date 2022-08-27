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


def padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    """
    Examples:
        >>> lengths = torch.tensor([2, 2, 3], dtype=torch.int32)
        >>> mask = padding_mask(lengths)
        >>> print(mask)
        tensor([[False, False,  True],
                [False, False,  True],
                [False, False, False]])
    """
    batch_size = lengths.size(0)
    max_len = int(lengths.max().item())
    seq = torch.arange(max_len, dtype=torch.int64, device=lengths.device)
    seq = seq.expand(batch_size, max_len)
    return seq >= lengths.unsqueeze(1)
