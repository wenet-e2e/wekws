#!/usr/bin/env python3
# Copyright (c) 2021 Binbin Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self,
                 channel: int,
                 kernel_size: int,
                 dilation: int,
                 dropout: float = 0.1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(
        self,
        x: torch.Tensor,
        cache: torch.Tensor = torch.zeros(0, 0, 0, dtype=torch.float)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x(torch.Tensor): Input tensor (B, D, T)
            cache(torch.Tensor): Input cache(B, D, self.padding)
        Returns:
            torch.Tensor(B, D, T): output
            torch.Tensor(B, D, self.padding): new cache
        """
        # The CNN used here is causal convolution
        if cache.size(0) == 0:
            y = F.pad(x, (self.padding, 0), value=0.0)
        else:
            y = torch.cat((cache, x), dim=2)
        assert y.size(2) > self.padding
        new_cache = y[:, :, -self.padding:]

        y = self.quant(y)
        # self.cnn is defined in the subclass of Block
        y = self.cnn(y)
        y = self.dequant(y)
        y = y + x  # residual connection
        return y, new_cache

    def fuse_modules(self):
        self.cnn.fuse_modules()


class CnnBlock(Block):
    def __init__(self,
                 channel: int,
                 kernel_size: int,
                 dilation: int,
                 dropout: float = 0.1):
        super().__init__(channel, kernel_size, dilation, dropout)
        self.cnn = nn.Sequential(
            nn.Conv1d(channel,
                      channel,
                      kernel_size,
                      stride=1,
                      dilation=dilation),
            nn.BatchNorm1d(channel),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def fuse_modules(self):
        torch.quantization.fuse_modules(self, [['cnn.0', 'cnn.1', 'cnn.2']],
                                        inplace=True)


class DsCnnBlock(Block):
    """ Depthwise Separable Convolution
    """
    def __init__(self,
                 channel: int,
                 kernel_size: int,
                 dilation: int,
                 dropout: float = 0.1):
        super().__init__(channel, kernel_size, dilation, dropout)
        self.cnn = nn.Sequential(
            nn.Conv1d(channel,
                      channel,
                      kernel_size,
                      stride=1,
                      dilation=dilation,
                      groups=channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(),
            nn.Conv1d(channel, channel, kernel_size=1, stride=1),
            nn.BatchNorm1d(channel),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def fuse_modules(self):
        torch.quantization.fuse_modules(
            self, [['cnn.0', 'cnn.1', 'cnn.2'], ['cnn.3', 'cnn.4', 'cnn.5']],
            inplace=True)


class TCN(nn.Module):
    def __init__(self,
                 num_layers: int,
                 channel: int,
                 kernel_size: int,
                 dropout: float = 0.1,
                 block_class=CnnBlock):
        super().__init__()
        self.padding = 0
        self.network = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2**i
            self.padding += (kernel_size - 1) * dilation
            self.network.append(
                block_class(channel, kernel_size, dilation, dropout))

    def forward(
        self,
        x: torch.Tensor,
        in_cache: torch.Tensor = torch.zeros(0, 0, 0, dtype=torch.float)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor (B, T, D)
            in_cache(torhc.Tensor): (B, D, C), C is the accumulated cache size

        Returns:
            torch.Tensor(B, T, D)
            torch.Tensor(B, D, C): C is the accumulated cache size
        """
        x = x.transpose(1, 2)  # (B, D, T)
        out_caches = []
        offset = 0
        for block in self.network:
            if in_cache.size(0) > 0:
                c_in = in_cache[:, :, offset:offset + block.padding]
            else:
                c_in = torch.zeros(0, 0, 0)
            x, c_out = block(x, c_in)
            out_caches.append(c_out)
            offset += block.padding
        x = x.transpose(1, 2)  # (B, T, D)
        new_cache = torch.cat(out_caches, dim=2)
        return x, new_cache

    def fuse_modules(self):
        for m in self.network:
            m.fuse_modules()


if __name__ == '__main__':
    tcn = TCN(4, 64, 8, block_class=CnnBlock)
    print(tcn)
    print(tcn.padding)
    num_params = sum(p.numel() for p in tcn.parameters())
    print('the number of model params: {}'.format(num_params))
    x = torch.zeros(3, 15, 64)
    y = tcn(x)
