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

import sys
from typing import Optional

import torch

from kws.model.cmvn import GlobalCMVN
from kws.model.subsampling import LinearSubsampling1, Conv1dSubsampling1
from kws.model.tcn import TCN, CnnBlock, DsCnnBlock
from kws.utils.cmvn import load_cmvn


class KwsModel(torch.nn.Module):
    """ Our model consists of four parts:
    1. global_cmvn: Optional, (idim, idim)
    2. subsampling: subsampling the input, (idim, hdim)
    3. body: body of the whole network, (hdim, hdim)
    4. linear: a linear layer, (hdim, odim)
    """
    def __init__(self, idim: int, odim: int, hdim: int,
                 global_cmvn: Optional[torch.nn.Module],
                 subsampling: torch.nn.Module, body: torch.nn.Module):
        super().__init__()
        self.idim = idim
        self.odim = odim
        self.hdim = hdim
        self.global_cmvn = global_cmvn
        self.subsampling = subsampling
        self.body = body
        self.linear = torch.nn.Linear(hdim, odim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.global_cmvn is not None:
            x = self.global_cmvn(x)
        x = self.subsampling(x)
        x, _ = self.body(x)
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x


def init_model(configs):
    cmvn = configs.get('cmvn', {})
    if cmvn['cmvn_file'] is not None:
        mean, istd = load_cmvn(cmvn['cmvn_file'])
        global_cmvn = GlobalCMVN(
            torch.from_numpy(mean).float(),
            torch.from_numpy(istd).float(), cmvn['norm_var'])
    else:
        global_cmvn = None

    input_dim = configs['input_dim']
    output_dim = configs['output_dim']
    hidden_dim = configs['hidden_dim']

    subsampling_type = configs['subsampling']['type']
    if subsampling_type == 'linear':
        subsampling = LinearSubsampling1(input_dim, hidden_dim)
    elif subsampling_type == 'cnn1d_s1':
        subsampling = Conv1dSubsampling1(input_dim, hidden_dim)
    else:
        print('Unknown subsampling type {}'.format(subsampling_type))
        sys.exit(1)

    body_type = configs['body']['type']
    num_layers = configs['body']['num_layers']
    if body_type == 'gru':
        body = torch.nn.GRU(hidden_dim,
                            hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)
    elif body_type == 'tcn':
        # Depthwise Separable
        ds = configs['body'].get('ds', False)
        if ds:
            block_class = DsCnnBlock
        else:
            block_class = CnnBlock
        kernel_size = configs['body'].get('kernel_size', 8)
        dropout = configs['body'].get('drouput', 0.1)
        body = TCN(num_layers, hidden_dim, kernel_size, dropout, block_class)
    else:
        print('Unknown body type {}'.format(body_type))
        sys.exit(1)

    kws_model = KwsModel(input_dim, output_dim, hidden_dim, global_cmvn,
                         subsampling, body)
    return kws_model
