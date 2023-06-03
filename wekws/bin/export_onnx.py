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

import onnx
import onnxruntime as ort

from wekws.model.kws_model import init_model
from wekws.utils.checkpoint import load_checkpoint


def get_args():
    parser = argparse.ArgumentParser(description='export to onnx model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--onnx_model',
                        required=True,
                        help='output onnx model')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    feature_dim = configs['model']['input_dim']
    model = init_model(configs['model'])
    if configs['training_config'].get('criterion', 'max_pooling') == 'ctc':
        # if we use ctc_loss, the logits need to be convert into probs before ctc_prefix_beam_search
        model.forward = model.forward_softmax
    print(model)

    load_checkpoint(model, args.checkpoint)
    model.eval()
    # dummy_input: (batch, time, feature_dim)
    dummy_input = torch.randn(1, 100, feature_dim, dtype=torch.float)
    cache = torch.zeros(1,
                        model.hdim,
                        model.backbone.padding,
                        dtype=torch.float)
    torch.onnx.export(model, (dummy_input, cache),
                      args.onnx_model,
                      input_names=['input', 'cache'],
                      output_names=['output', 'r_cache'],
                      dynamic_axes={
                          'input': {
                              1: 'T'
                          },
                          'output': {
                              1: 'T'
                          }},
                      opset_version=13,
                      verbose=False,
                      do_constant_folding=True)

    # Add hidden dim and cache size
    onnx_model = onnx.load(args.onnx_model)
    meta = onnx_model.metadata_props.add()
    meta.key, meta.value = 'cache_dim', str(model.hdim)
    meta = onnx_model.metadata_props.add()
    meta.key, meta.value = 'cache_len', str(model.backbone.padding)
    onnx.save(onnx_model, args.onnx_model)

    # Verify onnx precision
    torch_output = model(dummy_input, cache)
    ort_sess = ort.InferenceSession(args.onnx_model)
    onnx_output = ort_sess.run(None, {
        'input': dummy_input.numpy(),
        'cache': cache.numpy()
    })

    if torch.allclose(torch_output[0],
                      torch.tensor(onnx_output[0]), atol=1e-6) and \
       torch.allclose(torch_output[1],
                      torch.tensor(onnx_output[1]), atol=1e-6):
        print('Export to onnx succeed!')
    else:
        print('''Export to onnx succeed, but pytorch/onnx have different
                 outputs when given the same input, please check!!!''')


if __name__ == '__main__':
    main()
