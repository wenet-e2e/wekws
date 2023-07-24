# Copyright (c) 2020 Binbin Zhang
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

from __future__ import print_function

import argparse
import copy
import logging
import os

import torch
import torch.distributed as dist
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from wekws.dataset.dataset import Dataset
from wekws.utils.checkpoint import load_checkpoint, save_checkpoint
from wekws.model.kws_model import init_model
from wekws.utils.executor import Executor
from wekws.utils.train_utils import count_parameters, set_mannul_seed


def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--gpus',
                        default='-1',
                        help='gpu lists, seperated with `,`, -1 for cpu')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--seed', type=int, default=777, help='random seed')
    parser.add_argument('--checkpoint', help='checkpoint model')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--cmvn_file', default=None, help='global cmvn file')
    parser.add_argument('--norm_var',
                        action='store_true',
                        default=False,
                        help='norm var option')
    parser.add_argument('--num_keywords',
                        default=1,
                        type=int,
                        help='number of keywords')
    parser.add_argument('--min_duration',
                        default=50,
                        type=int,
                        help='min duration frames of the keyword')
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='prefetch number')
    parser.add_argument('--reverb_lmdb',
                        default=None,
                        help='reverb lmdb file')
    parser.add_argument('--noise_lmdb',
                        default=None,
                        help='noise lmdb file')

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    # Set random seed
    set_mannul_seed(args.seed)
    print(args)
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    gpu = int(args.gpus.split(',')[rank])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    if world_size > 1:
        logging.info('training on multiple gpus, this gpu {}'.format(gpu))
        dist.init_process_group(backend=args.dist_backend)

    train_conf = configs['dataset_conf']
    cv_conf = copy.deepcopy(train_conf)
    cv_conf['speed_perturb'] = False
    cv_conf['spec_aug'] = False
    cv_conf['shuffle'] = False

    train_dataset = Dataset(args.train_data,
                            train_conf,
                            reverb_lmdb=args.reverb_lmdb,
                            noise_lmdb=args.noise_lmdb)
    cv_dataset = Dataset(args.cv_data, cv_conf)

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=None,
                                   pin_memory=args.pin_memory,
                                   num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch)
    cv_data_loader = DataLoader(cv_dataset,
                                batch_size=None,
                                pin_memory=args.pin_memory,
                                num_workers=args.num_workers,
                                prefetch_factor=args.prefetch)

    input_dim = configs['dataset_conf']['feature_extraction_conf'][
        'num_mel_bins']
    output_dim = args.num_keywords

    # Write model_dir/config.yaml for inference and export
    if 'input_dim' not in configs['model']:
        configs['model']['input_dim'] = input_dim
    configs['model']['output_dim'] = output_dim
    if args.cmvn_file is not None:
        configs['model']['cmvn'] = {}
        configs['model']['cmvn']['norm_var'] = args.norm_var
        configs['model']['cmvn']['cmvn_file'] = args.cmvn_file
    if rank == 0:
        saved_config_path = os.path.join(args.model_dir, 'config.yaml')
        with open(saved_config_path, 'w') as fout:
            data = yaml.dump(configs)
            fout.write(data)

    # Init asr model from configs
    model = init_model(configs['model'])
    print(model)
    num_params = count_parameters(model)
    print('the number of model params: {}'.format(num_params))

    # !!!IMPORTANT!!!
    # Try to export the model by script, if fails, we should refine
    # the code to satisfy the script export requirements
    if rank == 0:
        pass
        # TODO: for now streaming FSMN do not support export to JITScript,
        # TODO: because there is nn.Sequential with Tuple input
        #  in current FSMN modules.
        #  the issue is in https://stackoverflow.com/questions/75714299/
        #  pytorch-jit-script-error-when-sequential-container-
        #  takes-a-tuple-input/76553450#76553450

        # script_model = torch.jit.script(model)
        # script_model.save(os.path.join(args.model_dir, 'init.zip'))
    executor = Executor()
    # If specify checkpoint, load some info from checkpoint
    if args.checkpoint is not None:
        infos = load_checkpoint(model, args.checkpoint)
    else:
        infos = {}
    start_epoch = infos.get('epoch', -1) + 1
    cv_loss = infos.get('cv_loss', 0.0)
    # get the last epoch lr
    lr_last_epoch = infos.get('lr', configs['optim_conf']['lr'])
    configs['optim_conf']['lr'] = lr_last_epoch
    model_dir = args.model_dir
    writer = None
    if rank == 0:
        os.makedirs(model_dir, exist_ok=True)
        exp_id = os.path.basename(model_dir)
        writer = SummaryWriter(os.path.join(args.tensorboard_dir, exp_id))

    if world_size > 1:
        assert (torch.cuda.is_available())
        # cuda model is required for nn.parallel.DistributedDataParallel
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
        device = torch.device("cuda")
    else:
        use_cuda = gpu >= 0 and torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        model = model.to(device)

    optimizer = optim.Adam(model.parameters(), **configs['optim_conf'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        threshold=0.01,
    )

    training_config = configs['training_config']
    training_config['min_duration'] = args.min_duration
    num_epochs = training_config.get('max_epoch', 100)
    final_epoch = None
    if start_epoch == 0 and rank == 0:
        save_model_path = os.path.join(model_dir, 'init.pt')
        save_checkpoint(model, save_model_path)

    # Start training loop
    for epoch in range(start_epoch, num_epochs):
        train_dataset.set_epoch(epoch)
        training_config['epoch'] = epoch
        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {}'.format(epoch, lr))
        executor.train(model, optimizer, train_data_loader, device, writer,
                       training_config)
        cv_loss, cv_acc = executor.cv(model, cv_data_loader, device,
                                      training_config)
        logging.info('Epoch {} CV info cv_loss {} cv_acc {}'.format(
            epoch, cv_loss, cv_acc))

        if rank == 0:
            save_model_path = os.path.join(model_dir, '{}.pt'.format(epoch))
            save_checkpoint(model, save_model_path, {
                'epoch': epoch,
                'lr': lr,
                'cv_loss': cv_loss,
            })
            writer.add_scalar('epoch/cv_loss', cv_loss, epoch)
            writer.add_scalar('epoch/cv_acc', cv_acc, epoch)
            writer.add_scalar('epoch/lr', lr, epoch)
        final_epoch = epoch
        scheduler.step(cv_loss)

    if final_epoch is not None and rank == 0:
        final_model_path = os.path.join(model_dir, 'final.pt')
        os.symlink('{}.pt'.format(final_epoch), final_model_path)
        writer.close()


if __name__ == '__main__':
    main()
