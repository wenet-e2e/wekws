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

import random

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

import wekws.dataset.processor as processor
from wekws.utils.file_utils import read_lists
from wekws.dataset.lmdb_data import LmdbData


class Processor(IterableDataset):
    def __init__(self, source, f, *args, **kw):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    def __iter__(self):
        """ Return an iterator over the source dataset processed by the
            given processor.
        """
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)

    def apply(self, f):
        assert callable(f)
        return Processor(self, f, *self.args, **self.kw)


class DistributedSampler:
    def __init__(self, shuffle=True, partition=True):
        self.epoch = -1
        self.update()
        self.shuffle = shuffle
        self.partition = partition

    def update(self):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        return dict(rank=self.rank,
                    world_size=self.world_size,
                    worker_id=self.worker_id,
                    num_workers=self.num_workers)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample(self, data):
        """ Sample data according to rank/world_size/num_workers

            Args:
                data(List): input data list

            Returns:
                List: data list after sample
        """
        data = data.copy()
        if self.partition:
            if self.shuffle:
                random.Random(self.epoch).shuffle(data)
            data = data[self.rank::self.world_size]
        data = data[self.worker_id::self.num_workers]
        return data


class DataList(IterableDataset):
    def __init__(self, lists, shuffle=True, partition=True):
        self.lists = lists
        self.sampler = DistributedSampler(shuffle, partition)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        sampler_info = self.sampler.update()
        lists = self.sampler.sample(self.lists)
        for src in lists:
            # yield dict(src=src)
            data = dict(src=src)
            data.update(sampler_info)
            yield data


def Dataset(data_list_file, conf,
            partition=True,
            reverb_lmdb=None,
            noise_lmdb=None):
    """ Construct dataset from arguments

        We have two shuffle stage in the Dataset. The first is global
        shuffle at shards tar/raw file level. The second is global shuffle
        at training samples level.

        Args:
            data_type(str): raw/shard
            partition(bool): whether to do data partition in terms of rank
            reverb_lmdb: reverb data source lmdb file
            noise_lmdb: noise data source lmdb file
    """
    lists = read_lists(data_list_file)
    shuffle = conf.get('shuffle', True)
    dataset = DataList(lists, shuffle=shuffle, partition=partition)
    dataset = Processor(dataset, processor.parse_raw)
    filter_conf = conf.get('filter_conf', {})
    dataset = Processor(dataset, processor.filter, **filter_conf)

    resample_conf = conf.get('resample_conf', {})
    dataset = Processor(dataset, processor.resample, **resample_conf)

    speed_perturb = conf.get('speed_perturb', False)
    if speed_perturb:
        dataset = Processor(dataset, processor.speed_perturb)
    if reverb_lmdb and conf.get('reverb_prob', 0) > 0:
        reverb_data = LmdbData(reverb_lmdb)
        dataset = Processor(dataset, processor.add_reverb,
                            reverb_data, conf['reverb_prob'])
    if noise_lmdb and conf.get('noise_prob', 0) > 0:
        noise_data = LmdbData(noise_lmdb)
        dataset = Processor(dataset, processor.add_noise,
                            noise_data, conf['noise_prob'])
    feature_extraction_conf = conf.get('feature_extraction_conf', {})
    if feature_extraction_conf['feature_type'] == 'mfcc':
        dataset = Processor(dataset, processor.compute_mfcc,
                            **feature_extraction_conf)
    elif feature_extraction_conf['feature_type'] == 'fbank':
        dataset = Processor(dataset, processor.compute_fbank,
                            **feature_extraction_conf)
    spec_aug = conf.get('spec_aug', True)
    if spec_aug:
        spec_aug_conf = conf.get('spec_aug_conf', {})
        dataset = Processor(dataset, processor.spec_aug, **spec_aug_conf)

    context_expansion = conf.get('context_expansion', False)
    if context_expansion:
        context_expansion_conf = conf.get('context_expansion_conf', {})
        dataset = Processor(dataset, processor.context_expansion,
                            **context_expansion_conf)

    frame_skip = conf.get('frame_skip', 1)
    if frame_skip > 1:
        dataset = Processor(dataset, processor.frame_skip, frame_skip)

    if shuffle:
        shuffle_conf = conf.get('shuffle_conf', {})
        dataset = Processor(dataset, processor.shuffle, **shuffle_conf)

    batch_conf = conf.get('batch_conf', {})
    dataset = Processor(dataset, processor.batch, **batch_conf)
    dataset = Processor(dataset, processor.padding)
    return dataset


if __name__ == '__main__':
    import sys
    dataset = Dataset(sys.argv[1], {})
    for data in dataset:
        print(data)
