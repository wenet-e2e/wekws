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

import logging

import torch
from torch.nn.utils import clip_grad_norm_

from wekws.model.loss import criterion


class Executor:
    def __init__(self):
        self.step = 0

    def train(self, model, optimizer, data_loader, device, writer, args):
        ''' Train one epoch
        '''
        model.train()
        clip = args.get('grad_clip', 50.0)
        log_interval = args.get('log_interval', 10)
        epoch = args.get('epoch', 0)
        min_duration = args.get('min_duration', 0)

        for batch_idx, batch in enumerate(data_loader):
            key, feats, target, feats_lengths, label_lengths = batch
            feats = feats.to(device)
            target = target.to(device)
            feats_lengths = feats_lengths.to(device)
            label_lengths = label_lengths.to(device)
            num_utts = feats_lengths.size(0)
            if num_utts == 0:
                continue
            logits, _ = model(feats)
            loss_type = args.get('criterion', 'max_pooling')
            loss, acc = criterion(loss_type, logits, target, feats_lengths,
                                  target_lengths=label_lengths,
                                  min_duration=min_duration,
                                  validation=False)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), clip)
            if torch.isfinite(grad_norm):
                optimizer.step()
            if batch_idx % log_interval == 0:
                logging.debug(
                    'TRAIN Batch {}/{} loss {:.8f} acc {:.8f}'.format(
                        epoch, batch_idx, loss.item(), acc))

    def cv(self, model, data_loader, device, args):
        ''' Cross validation on
        '''
        model.eval()
        log_interval = args.get('log_interval', 10)
        epoch = args.get('epoch', 0)
        # in order to avoid division by 0
        num_seen_utts = 1
        total_loss = 0.0
        total_acc = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                key, feats, target, feats_lengths, label_lengths = batch
                feats = feats.to(device)
                target = target.to(device)
                feats_lengths = feats_lengths.to(device)
                label_lengths = label_lengths.to(device)
                num_utts = feats_lengths.size(0)
                if num_utts == 0:
                    continue
                logits, _ = model(feats)
                loss, acc = criterion(args.get('criterion', 'max_pooling'),
                                      logits, target, feats_lengths,
                                      target_lengths=label_lengths,
                                      min_duration=0,
                                      validation=True)
                if torch.isfinite(loss):
                    num_seen_utts += num_utts
                    total_loss += loss.item() * num_utts
                    total_acc += acc * num_utts
                if batch_idx % log_interval == 0:
                    logging.debug(
                        'CV Batch {}/{} loss {:.8f} acc {:.8f} history loss {:.8f}'
                        .format(epoch, batch_idx, loss.item(), acc,
                                total_loss / num_seen_utts))
        return total_loss / num_seen_utts, total_acc / num_seen_utts

    def test(self, model, data_loader, device, args):
        return self.cv(model, data_loader, device, args)
