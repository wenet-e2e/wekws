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

import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn

from wekws.model.cmvn import GlobalCMVN
from wekws.model.classifier import (GlobalClassifier, LastClassifier,
                                    LinearClassifier)
from wekws.model.subsampling import (LinearSubsampling1, Conv1dSubsampling1,
                                     NoSubsampling)
from wekws.model.tcn import TCN, CnnBlock, DsCnnBlock
from wekws.model.mdtc import MDTC
from wekws.utils.cmvn import load_cmvn, load_kaldi_cmvn
from wekws.model.fsmn import FSMN


class KWSModel(nn.Module):
    """Our model consists of four parts:
    1. global_cmvn: Optional, (idim, idim)
    2. preprocessing: feature dimention projection, (idim, hdim)
    3. backbone: backbone of the whole network, (hdim, hdim)
    4. classifier: output layer or classifier of KWS model, (hdim, odim)
    5. activation:
        nn.Sigmoid for wakeup word
        nn.Identity for speech command dataset
    """
    def __init__(
        self,
        idim: int,
        odim: int,
        hdim: int,
        global_cmvn: Optional[nn.Module],
        preprocessing: Optional[nn.Module],
        backbone: nn.Module,
        classifier: nn.Module,
        activation: nn.Module,
    ):
        super().__init__()
        self.idim = idim
        self.odim = odim
        self.hdim = hdim
        self.global_cmvn = global_cmvn
        self.preprocessing = preprocessing
        self.backbone = backbone
        self.classifier = classifier
        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,
        in_cache: torch.Tensor = torch.zeros(0, 0, 0, dtype=torch.float)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.global_cmvn is not None:
            x = self.global_cmvn(x)
        x = self.preprocessing(x)
        x, out_cache = self.backbone(x, in_cache)
        x = self.classifier(x)
        x = self.activation(x)
        return x, out_cache

    def forward_softmax(self,
                        x: torch.Tensor,
                        in_cache: torch.Tensor = torch.zeros(
                            0, 0, 0, dtype=torch.float)
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.global_cmvn is not None:
            x = self.global_cmvn(x)
        x = self.preprocessing(x)
        x, out_cache = self.backbone(x, in_cache)
        x = self.classifier(x)
        x = self.activation(x)
        x = x.softmax(2)
        return x, out_cache

    def fuse_modules(self):
        self.preprocessing.fuse_modules()
        self.backbone.fuse_modules()


def init_model(configs):
    cmvn = configs.get('cmvn', {})
    if 'cmvn_file' in cmvn and cmvn['cmvn_file'] is not None:
        if "kaldi" in cmvn['cmvn_file']:
            mean, istd = load_kaldi_cmvn(cmvn['cmvn_file'])
        else:
            mean, istd = load_cmvn(cmvn['cmvn_file'])
        global_cmvn = GlobalCMVN(
            torch.from_numpy(mean).float(),
            torch.from_numpy(istd).float(),
            cmvn['norm_var'],
        )
    else:
        global_cmvn = None

    input_dim = configs['input_dim']
    output_dim = configs['output_dim']
    hidden_dim = configs['hidden_dim']

    prep_type = configs['preprocessing']['type']
    if prep_type == 'linear':
        preprocessing = LinearSubsampling1(input_dim, hidden_dim)
    elif prep_type == 'cnn1d_s1':
        preprocessing = Conv1dSubsampling1(input_dim, hidden_dim)
    elif prep_type == 'none':
        preprocessing = NoSubsampling()
    else:
        print('Unknown preprocessing type {}'.format(prep_type))
        sys.exit(1)

    backbone_type = configs['backbone']['type']
    if backbone_type == 'gru':
        num_layers = configs['backbone']['num_layers']
        backbone = torch.nn.GRU(hidden_dim,
                                hidden_dim,
                                num_layers=num_layers,
                                batch_first=True)
    elif backbone_type == 'tcn':
        # Depthwise Separable
        num_layers = configs['backbone']['num_layers']
        ds = configs['backbone'].get('ds', False)
        if ds:
            block_class = DsCnnBlock
        else:
            block_class = CnnBlock
        kernel_size = configs['backbone'].get('kernel_size', 8)
        dropout = configs['backbone'].get('drouput', 0.1)
        backbone = TCN(num_layers, hidden_dim, kernel_size, dropout,
                       block_class)
    elif backbone_type == 'mdtc':
        stack_size = configs['backbone']['stack_size']
        num_stack = configs['backbone']['num_stack']
        kernel_size = configs['backbone']['kernel_size']
        hidden_dim = configs['backbone']['hidden_dim']
        causal = configs['backbone']['causal']
        backbone = MDTC(num_stack,
                        stack_size,
                        hidden_dim,
                        hidden_dim,
                        kernel_size,
                        causal=causal)
    elif backbone_type == 'fsmn':
        input_affine_dim = configs['backbone']['input_affine_dim']
        num_layers = configs['backbone']['num_layers']
        linear_dim = configs['backbone']['linear_dim']
        proj_dim = configs['backbone']['proj_dim']
        left_order = configs['backbone']['left_order']
        right_order = configs['backbone']['right_order']
        left_stride = configs['backbone']['left_stride']
        right_stride = configs['backbone']['right_stride']
        output_affine_dim = configs['backbone']['output_affine_dim']
        backbone = FSMN(input_dim, input_affine_dim, num_layers, linear_dim,
                        proj_dim, left_order, right_order, left_stride,
                        right_stride, output_affine_dim, output_dim)

    else:
        print('Unknown body type {}'.format(backbone_type))
        sys.exit(1)
    if 'classifier' in configs:
        # For speech command dataset, we use 2 FC layer as classifier,
        # we add dropout after first FC layer to prevent overfitting
        classifier_type = configs['classifier']['type']
        dropout = configs['classifier']['dropout']

        classifier_base = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(64, output_dim))
        if classifier_type == 'global':
            # global means we add a global average pooling before classifier
            classifier = GlobalClassifier(classifier_base)
        elif classifier_type == 'last':
            # last means we use last frame to do backpropagation, so the model
            # can be infered streamingly
            classifier = LastClassifier(classifier_base)
        elif classifier_type == 'identity':
            classifier = nn.Identity()
        else:
            print('Unknown classifier type {}'.format(classifier_type))
            sys.exit(1)
        activation = nn.Identity()
    else:
        classifier = LinearClassifier(hidden_dim, output_dim)
        activation = nn.Sigmoid()

    # Here we add a possible "activation_type",
    # one can choose to use other activation function.
    # We use nn.Identity just for CTC loss
    if "activation" in configs:
        activation_type = configs["activation"]["type"]
        if activation_type == 'identity':
            activation = nn.Identity()
        else:
            print('Unknown activation type {}'.format(activation_type))
            sys.exit(1)

    kws_model = KWSModel(input_dim, output_dim, hidden_dim, global_cmvn,
                         preprocessing, backbone, classifier, activation)
    return kws_model
