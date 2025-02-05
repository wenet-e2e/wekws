# Copyright (c) 2021 Wenet Community. (authors: Binbin Zhang)
#               2023 Wenet Community. (authors: Dinghao Zhou)
#               2025 Wenet Community. (authors: Menglong Xu)
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

import copy
from wenet.dataset.dataset import Dataset


def init_dataset(dataset_type: str = 'asr',
                 data_type: str = 'raw',
                 data_list_file=None,
                 tokenizer=None,
                 conf=None,
                 partition=True,
                 split='train'):
    assert dataset_type in ['asr']
    assert data_list_file is not None
    assert conf is not None

    if split != 'train':
        cv_conf = copy.deepcopy(conf)
        cv_conf['cycle'] = 1
        cv_conf['speed_perturb'] = False
        cv_conf['spec_aug'] = False
        cv_conf['spec_sub'] = False
        cv_conf['spec_trim'] = False
        cv_conf['shuffle'] = False
        cv_conf['list_shuffle'] = False
        conf = cv_conf

    return Dataset(data_type, data_list_file, tokenizer, conf, partition)
