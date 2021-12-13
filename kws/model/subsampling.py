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

# There is no right context or lookahead in our Subsampling design, so
# If there is CNN in Subsampling, it's a causal CNN.


class SubsamplingBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.subsampling_rate = 1


class NoSubsampling(SubsamplingBase):
    """No subsampling in accordance to the 'none' preprocessing
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class LinearSubsampling1(SubsamplingBase):
    """Linear transform the input without subsampling
    """
    def __init__(self, idim: int, odim: int):
        super().__init__()
        self.out = torch.nn.Sequential(
            torch.nn.Linear(idim, odim),
            torch.nn.ReLU(),
        )
        self.subsampling_rate = 1
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.out(x)
        x = self.dequant(x)
        return x

    def fuse_modules(self):
        torch.quantization.fuse_modules(self, [['out.0', 'out.1']],
                                        inplace=True)


class Conv1dSubsampling1(SubsamplingBase):
    """Conv1d transform without subsampling
    """
    def __init__(self, idim: int, odim: int):
        super().__init__()
        self.out = torch.nn.Sequential(
            torch.nn.Conv1d(idim, odim, 3),
            torch.nn.BatchNorm1d(odim),
            torch.nn.ReLU(),
        )
        self.subsampling_rate = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.out(x)
        return x
