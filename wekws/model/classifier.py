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


class GlobalClassifier(nn.Module):
    """Add a global average pooling before the classifier"""
    def __init__(self, classifier: nn.Module):
        super(GlobalClassifier, self).__init__()
        self.classifier = classifier

    def forward(self, x: torch.Tensor):
        x = torch.mean(x, dim=1)
        return self.classifier(x)


class LastClassifier(nn.Module):
    """Select last frame to do the classification"""
    def __init__(self, classifier: nn.Module):
        super(LastClassifier, self).__init__()
        self.classifier = classifier

    def forward(self, x: torch.Tensor):
        x = x[:, -1, :]
        return self.classifier(x)

class ElementClassifier(nn.Module):
    """Classify all the frames in an utterance"""
    def __init__(self, classifier: nn.Module):
        super(ElementClassifier, self).__init__()
        self.classifier = classifier

    def forward(self, x: torch.Tensor):
        return self.classifier(x)

class LinearClassifier(nn.Module):
    """ Wrapper of Linear """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor):
        x = self.quant(x)
        x = self.linear(x)
        x = self.dequant(x)
        return x
