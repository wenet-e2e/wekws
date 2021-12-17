# Copyright (c) 2021 Binbin Zhang(binbzha@qq.com)
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

import argparse

import torch
import yaml
import onnxruntime as ort


def get_args():
    parser = argparse.ArgumentParser(description='export to onnx model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--jit_model',
                        required=True,
                        help='pytorch jit script model')
    parser.add_argument('--onnx_model',
                        required=True,
                        help='output onnx model')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    feature_dim = configs['model']['input_dim']
    model = torch.jit.load(args.jit_model)
    print(model)
    # dummy_input: (batch, time, feature_dim)
    dummy_input = torch.randn(1, 100, feature_dim, dtype=torch.float)
    torch.onnx.export(model,
                      dummy_input,
                      args.onnx_model,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {
                          1: 'T'
                      }})

    torch_output = model(dummy_input)
    ort_sess = ort.InferenceSession(args.onnx_model)
    onnx_input = dummy_input.numpy()
    onnx_output = ort_sess.run(None, {'input': onnx_input})
    if torch.allclose(torch_output, torch.tensor(onnx_output[0])):
        print('Export to onnx succeed!')
    else:
        print('''Export to onnx succeed, but pytorch/onnx have different
                 outputs when given the same input, please check!!!''')


if __name__ == '__main__':
    main()
