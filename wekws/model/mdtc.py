#!/usr/bin/env python3
# Copyright (c) 2021 Jingyong Hou (houjingyong@gmail.com)
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


class DSDilatedConv1d(nn.Module):
    """Dilated Depthwise-Separable Convolution"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        stride: int = 1,
        bias: bool = True,
    ):
        super(DSDilatedConv1d, self).__init__()
        self.padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            padding=0,
            dilation=dilation,
            stride=stride,
            groups=in_channels,
            bias=bias,
        )
        self.bn = nn.BatchNorm1d(in_channels)
        self.pointwise = nn.Conv1d(in_channels,
                                   out_channels,
                                   kernel_size=1,
                                   padding=0,
                                   dilation=1,
                                   bias=bias)

    def forward(self, inputs: torch.Tensor):
        outputs = self.conv(inputs)
        outputs = self.bn(outputs)
        outputs = self.pointwise(outputs)
        return outputs


class TCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        res_channels: int,
        kernel_size: int,
        dilation: int,
        causal: bool,
    ):
        super(TCNBlock, self).__init__()
        self.in_channels = in_channels
        self.res_channels = res_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.causal = causal
        self.padding = dilation * (kernel_size - 1)
        self.half_padding = self.padding // 2
        self.conv1 = DSDilatedConv1d(
            in_channels=in_channels,
            out_channels=res_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(res_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=res_channels,
                               out_channels=res_channels,
                               kernel_size=1)
        self.bn2 = nn.BatchNorm1d(res_channels)
        self.relu2 = nn.ReLU()

    def forward(
        self,
        inputs: torch.Tensor,
        cache: torch.Tensor = torch.zeros(0, 0, 0, dtype=torch.float)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs(torch.Tensor): Input tensor (B, D, T)
            cache(torch.Tensor): Input cache(B, D, self.padding)
        Returns:
            torch.Tensor(B, D, T): outputs
            torch.Tensor(B, D, self.padding): new cache
        """
        if cache.size(0) == 0:
            outputs = F.pad(inputs, (self.padding, 0), value=0.0)
        else:
            outputs = torch.cat((cache, inputs), dim=2)
        assert outputs.size(2) > self.padding
        new_cache = outputs[:, :, -self.padding:]

        outputs = self.relu1(self.bn1(self.conv1(outputs)))
        outputs = self.bn2(self.conv2(outputs))
        if self.in_channels == self.res_channels:
            res_out = self.relu2(outputs + inputs)
        else:
            res_out = self.relu2(outputs)
        return res_out, new_cache


class TCNStack(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stack_num: int,
        stack_size: int,
        res_channels: int,
        kernel_size: int,
        causal: bool,
    ):
        super(TCNStack, self).__init__()
        self.in_channels = in_channels
        self.stack_num = stack_num
        self.stack_size = stack_size
        self.res_channels = res_channels
        self.kernel_size = kernel_size
        self.causal = causal
        self.res_blocks = self.stack_tcn_blocks()
        self.padding = self.calculate_padding()

    def calculate_padding(self):
        padding = 0
        for block in self.res_blocks:
            padding += block.padding
        return padding

    def build_dilations(self):
        dilations = []
        for s in range(0, self.stack_size):
            for l in range(0, self.stack_num):
                dilations.append(2**l)
        return dilations

    def stack_tcn_blocks(self):
        dilations = self.build_dilations()
        res_blocks = nn.ModuleList()

        res_blocks.append(
            TCNBlock(
                self.in_channels,
                self.res_channels,
                self.kernel_size,
                dilations[0],
                self.causal,
            ))
        for dilation in dilations[1:]:
            res_blocks.append(
                TCNBlock(
                    self.res_channels,
                    self.res_channels,
                    self.kernel_size,
                    dilation,
                    self.causal,
                ))
        return res_blocks

    def forward(
        self,
        inputs: torch.Tensor,
        in_cache: torch.Tensor = torch.zeros(0, 0, 0, dtype=torch.float)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = inputs  # (B, D, T)
        out_caches = []
        offset = 0
        for block in self.res_blocks:
            if in_cache.size(0) > 0:
                c_in = in_cache[:, :, offset:offset + block.padding]
            else:
                c_in = torch.zeros(0, 0, 0)
            outputs, c_out = block(outputs, c_in)
            out_caches.append(c_out)
            offset += block.padding
        new_cache = torch.cat(out_caches, dim=2)
        return outputs, new_cache


class MDTC(nn.Module):
    """Multi-scale Depthwise Temporal Convolution (MDTC).
    In MDTC, stacked depthwise one-dimensional (1-D) convolution with
    dilated connections is adopted to efficiently model long-range
    dependency of speech. With a large receptive field while
    keeping a small number of model parameters, the structure
    can model temporal context of speech effectively. It aslo
    extracts multi-scale features from different hidden layers
    of MDTC with different receptive fields.
    """
    def __init__(
        self,
        stack_num: int,
        stack_size: int,
        in_channels: int,
        res_channels: int,
        kernel_size: int,
        causal: bool,
    ):
        super(MDTC, self).__init__()
        assert kernel_size % 2 == 1
        self.kernel_size = kernel_size
        assert causal is True, "we now only support causal mdtc"
        self.causal = causal
        self.preprocessor = TCNBlock(in_channels,
                                     res_channels,
                                     kernel_size,
                                     dilation=1,
                                     causal=causal)
        self.relu = nn.ReLU()
        self.blocks = nn.ModuleList()
        self.padding = self.preprocessor.padding
        for i in range(stack_num):
            self.blocks.append(
                TCNStack(res_channels, stack_size, 1, res_channels,
                         kernel_size, causal))
            self.padding += self.blocks[-1].padding
        self.half_padding = self.padding // 2
        print('Receptive Fields: %d' % self.padding)

    def forward(
        self,
        x: torch.Tensor,
        in_cache: torch.Tensor = torch.zeros(0, 0, 0, dtype=torch.float)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = x.transpose(1, 2)  # (B, D, T)
        outputs_list = []
        out_caches = []
        offset = 0
        if in_cache.size(0) > 0:
            c_in = in_cache[:, :, offset:offset + self.preprocessor.padding]
        else:
            c_in = torch.zeros(0, 0, 0)

        outputs, c_out = self.preprocessor(outputs, c_in)
        outputs = self.relu(outputs)
        out_caches.append(c_out)
        offset += self.preprocessor.padding
        for block in self.blocks:
            if in_cache.size(0) > 0:
                c_in = in_cache[:, :, offset:offset + block.padding]
            else:
                c_in = torch.zeros(0, 0, 0)
            outputs, c_out = block(outputs, c_in)
            outputs_list.append(outputs)
            out_caches.append(c_out)
            offset += block.padding

        outputs = torch.zeros_like(outputs_list[-1], dtype=outputs_list[-1].dtype)
        for x in outputs_list:
            outputs += x
        outputs = outputs.transpose(1, 2)  # (B, T, D)
        new_cache = torch.cat(out_caches, dim=2)
        return outputs, new_cache


if __name__ == '__main__':
    mdtc = MDTC(3, 4, 64, 64, 5, causal=True)
    print(mdtc)

    num_params = sum(p.numel() for p in mdtc.parameters())
    print('the number of model params: {}'.format(num_params))
    x = torch.randn(128, 200, 64)  # batch-size * time * dim
    y, c = mdtc(x)
    print('input shape: {}'.format(x.shape))
    print('output shape: {}'.format(y.shape))
    print('cache shape: {}'.format(c.shape))

    print('########################################')
    for _ in range(10):
        y, c = mdtc(y, c)
    print('output shape: {}'.format(y.shape))
    print('cache shape: {}'.format(c.shape))
