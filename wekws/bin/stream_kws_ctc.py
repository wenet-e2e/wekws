# Copyright (c) 2023 Jing Du(thuduj12@163.com)
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
import struct
import wave
import logging
import os
import math
import numpy as np
import torchaudio.compliance.kaldi as kaldi

import torch
import torch.nn.functional as F
import yaml
from collections import defaultdict
from wekws.model.kws_model import init_model
from wekws.utils.checkpoint import load_checkpoint
from tools.make_list import query_token_set, read_lexicon, read_token


def get_args():
    parser = argparse.ArgumentParser(description='detect keywords online.')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--wav_path', required=False, default=None, help='test wave path.')
    parser.add_argument('--wav_scp', required=False, default=None, help='test wave scp.')
    parser.add_argument('--result_file', required=False, default=None, help='test result.')

    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--jit_model',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--keywords', type=str, default=None, help='the keywords, split with comma(,)')
    parser.add_argument('--token_file', type=str, default=None, help='the path of tokens.txt')
    parser.add_argument('--lexicon_file', type=str, default=None, help='the path of lexicon.txt')
    parser.add_argument('--score_beam_size',
                        default=3,
                        type=int,
                        help='The first prune beam, filter out those frames with low scores.')
    parser.add_argument('--path_beam_size',
                        default=20,
                        type=int,
                        help='The second prune beam, keep only path_beam_size candidates.')
    parser.add_argument('--threshold',
                        type=float,
                        default=0.0,
                        help='The threshold of kws. If ctc_search probs exceed this value,'
                             'the keyword will be activated.')
    parser.add_argument('--min_frames',
                        default=5,
                        type=int,
                        help='The min frames of keyword\'s duration.')
    parser.add_argument('--max_frames',
                        default=250,
                        type=int,
                        help='The max frames of keyword\'s duration.')
    parser.add_argument('--interval_frames',
                        default=50,
                        type=int,
                        help='The interval frames of two continuous keywords.')

    args = parser.parse_args()
    return args


def is_sublist(main_list, check_list):
    if len(main_list) < len(check_list):
        return -1

    if len(main_list) == len(check_list):
        return 0 if main_list == check_list else -1

    for i in range(len(main_list) - len(check_list)):
        if main_list[i] == check_list[0]:
            for j in range(len(check_list)):
                if main_list[i + j] != check_list[j]:
                    break
            else:
                return i
    else:
        return -1

def ctc_prefix_beam_search(t, probs, cur_hyps, keywords_idxset, score_beam_size):
    '''

    :param t: the time in frame
    :param probs: the probability in t_th frame, (vocab_size, )
    :param cur_hyps: list of tuples. [(tuple(), (1.0, 0.0, []))]
                in tuple, 1st is prefix id, 2nd include p_blank, p_non_blank, and path nodes list.
                in path nodes list, each node is a dict of {token=idx, frame=t, prob=ps}
    :param keywords_idxset: the index of keywords in token.txt
    :param score_beam_size: the probability threshold, to filter out those frames with low probs.
    :return:
            next_hyps: the hypothesis depend on current hyp and current frame.
    '''
    # key: prefix, value (pb, pnb), default value(-inf, -inf)
    next_hyps = defaultdict(lambda: (0.0, 0.0, []))

    # 2.1 First beam prune: select topk best
    top_k_probs, top_k_index = probs.topk(score_beam_size)

    # filter prob score that is too small
    filter_probs = []
    filter_index = []
    for prob, idx in zip(top_k_probs.tolist(), top_k_index.tolist()):
        if keywords_idxset is not None:
            if prob > 0.05 and idx in keywords_idxset:
                filter_probs.append(prob)
                filter_index.append(idx)
        else:
            if prob > 0.05:
                filter_probs.append(prob)
                filter_index.append(idx)

    if len(filter_index) == 0:
        return cur_hyps

    for s in filter_index:
        ps = probs[s].item()

        for prefix, (pb, pnb, cur_nodes) in cur_hyps:
            last = prefix[-1] if len(prefix) > 0 else None
            if s == 0:  # blank
                n_pb, n_pnb, nodes = next_hyps[prefix]
                n_pb = n_pb + pb * ps + pnb * ps
                nodes = cur_nodes.copy()
                next_hyps[prefix] = (n_pb, n_pnb, nodes)
            elif s == last:
                if not math.isclose(pnb, 0.0, abs_tol=0.000001):
                    # Update *ss -> *s;
                    n_pb, n_pnb, nodes = next_hyps[prefix]
                    n_pnb = n_pnb + pnb * ps
                    nodes = cur_nodes.copy()
                    if ps > nodes[-1]['prob']:  # update frame and prob
                        nodes[-1]['prob'] = ps
                        nodes[-1]['frame'] = t
                    next_hyps[prefix] = (n_pb, n_pnb, nodes)

                if not math.isclose(pb, 0.0, abs_tol=0.000001):
                    # Update *s-s -> *ss, - is for blank
                    n_prefix = prefix + (s,)
                    n_pb, n_pnb, nodes = next_hyps[n_prefix]
                    n_pnb = n_pnb + pb * ps
                    nodes = cur_nodes.copy()
                    nodes.append(dict(token=s, frame=t,
                                      prob=ps))  # to record token prob
                    next_hyps[n_prefix] = (n_pb, n_pnb, nodes)
            else:
                n_prefix = prefix + (s,)
                n_pb, n_pnb, nodes = next_hyps[n_prefix]
                if nodes:
                    if ps > nodes[-1]['prob']:  # update frame and prob
                        # nodes[-1]['prob'] = ps
                        # nodes[-1]['frame'] = t
                        nodes.pop()  # to avoid change other beam which has this node.
                        nodes.append(dict(token=s, frame=t, prob=ps))
                else:
                    nodes = cur_nodes.copy()
                    nodes.append(dict(token=s, frame=t,
                                      prob=ps))  # to record token prob
                n_pnb = n_pnb + pb * ps + pnb * ps
                next_hyps[n_prefix] = (n_pb, n_pnb, nodes)

    # 2.2 Second beam prune
    next_hyps = sorted(
        next_hyps.items(), key=lambda x: (x[1][0] + x[1][1]), reverse=True)

    return next_hyps

class KeyWordSpotter(torch.nn.Module):
    def __init__(self, ckpt_path, config_path, token_path, lexicon_path,
                 threshold, min_frames=5, max_frames=250, interval_frames=50,
                 score_beam=3, path_beam=20,
                 gpu=-1, is_jit_model=False,):
        super().__init__()
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        with open(config_path, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
        dataset_conf = configs['dataset_conf']

        # feature related
        self.sample_rate = 16000
        self.wave_remained = np.array([])
        self.num_mel_bins = dataset_conf['feature_extraction_conf']['num_mel_bins']
        self.frame_length = dataset_conf['feature_extraction_conf']['frame_length']  # in ms
        self.frame_shift = dataset_conf['feature_extraction_conf']['frame_shift']    # in ms
        self.downsampling = dataset_conf.get('frame_skip', 1)
        self.resolution = self.frame_shift / 1000 * self.downsampling   # in second
        # fsmn splice operation
        self.context_expansion = dataset_conf.get('context_expansion', False)
        self.left_context = 0
        self.right_context = 0
        if self.context_expansion:
            self.left_context = dataset_conf['context_expansion_conf']['left']
            self.right_context = dataset_conf['context_expansion_conf']['right']
        self.feature_remained = None
        self.feats_ctx_offset = 0  # after downsample, offset exist.


        # model related
        if is_jit_model:
            model = torch.jit.load(ckpt_path)
            # For script model, only cpu is supported.
            device = torch.device('cpu')
        else:
            # Init model from configs
            model = init_model(configs['model'])
            load_checkpoint(model, ckpt_path)
            use_cuda = gpu >= 0 and torch.cuda.is_available()
            device = torch.device('cuda' if use_cuda else 'cpu')
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        logging.info(f'model {ckpt_path} loaded.')
        self.token_table = read_token(token_path)
        logging.info(f'tokens {token_path} with {len(self.token_table)} units loaded.')
        self.lexicon_table = read_lexicon(lexicon_path)
        logging.info(f'lexicons {lexicon_path} with {len(self.lexicon_table)} units loaded.')
        self.in_cache = torch.zeros(0, 0, 0, dtype=torch.float)


        # decoding and detection related
        self.score_beam = score_beam
        self.path_beam = path_beam

        self.threshold = threshold
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.interval_frames = interval_frames

        self.cur_hyps = [(tuple(), (1.0, 0.0, []))]
        self.hit_score = 1.0
        self.hit_keyword = None
        self.activated = False

        self.total_frames = 0   # frame offset, for absolute time
        self.last_active_pos = -1  # the last frame of being activated
        self.result = {}

    def set_keywords(self, keywords):
        # 4. parse keywords tokens
        assert keywords is not None, 'at least one keyword is needed, multiple keywords should be splitted with comma(,)'
        keywords_str = keywords
        keywords_list = keywords_str.strip().replace(' ', '').split(',')
        keywords_token = {}
        keywords_idxset = {0}
        keywords_strset = {'<blk>'}
        keywords_tokenmap = {'<blk>': 0}
        for keyword in keywords_list:
            strs, indexes = query_token_set(keyword, self.token_table, self.lexicon_table)
            keywords_token[keyword] = {}
            keywords_token[keyword]['token_id'] = indexes
            keywords_token[keyword]['token_str'] = ''.join('%s ' % str(i)
                                                           for i in indexes)
            [keywords_strset.add(i) for i in strs]
            [keywords_idxset.add(i) for i in indexes]
            for txt, idx in zip(strs, indexes):
                if keywords_tokenmap.get(txt, None) is None:
                    keywords_tokenmap[txt] = idx

        token_print = ''
        for txt, idx in keywords_tokenmap.items():
            token_print += f'{txt}({idx}) '
        logging.info(f'Token set is: {token_print}')
        self.keywords_idxset = keywords_idxset
        self.keywords_token = keywords_token

    def accept_wave(self, wave):
        assert isinstance(wave, bytes), "please make sure the input format is bytes(raw PCM)"
        # convert bytes into float32
        data = []
        for i in range(0, len(wave), 2):
            value = struct.unpack('<h', wave[i:i + 2])[0]
            data.append(value)  # here we don't divide 32768.0, because kaldi.fbank accept original input

        wave = np.array(data)
        wave = np.append(self.wave_remained, wave)
        if wave.size < (self.frame_length * self.sample_rate / 1000) * self.right_context :
            self.wave_remained = wave
            return None
        wave_tensor = torch.from_numpy(wave).float().to(self.device)
        wave_tensor = wave_tensor.unsqueeze(0)   # add a channel dimension
        feats = kaldi.fbank(wave_tensor,
                          num_mel_bins=self.num_mel_bins,
                          frame_length=self.frame_length,
                          frame_shift=self.frame_shift,
                          dither=0,
                          energy_floor=0.0,
                          sample_frequency=self.sample_rate)
        # update wave remained
        feat_len = len(feats)
        frame_shift = int(self.frame_shift / 1000 * self.sample_rate)
        self.wave_remained = wave[feat_len * frame_shift:]

        if self.context_expansion:
            assert feat_len > self.right_context, "make sure each chunk feat length is large than right context."
            # pad feats with remained feature from last chunk
            if self.feature_remained is None:  # first chunk
                # pad first frame at the beginning, replicate just support last dimension, so we do transpose.
                feats_pad = F.pad(feats.T, (self.left_context, 0), mode='replicate').T
            else:
                feats_pad = torch.cat((self.feature_remained, feats))

            ctx_frm = feats_pad.shape[0] - (self.right_context+self.right_context)
            ctx_win = (self.left_context + self.right_context + 1)
            ctx_dim = feats.shape[1] * ctx_win
            feats_ctx = torch.zeros(ctx_frm, ctx_dim, dtype=torch.float32)
            for i in range(ctx_frm):
                feats_ctx[i] = torch.cat(tuple(feats_pad[i: i + ctx_win])).unsqueeze(0)

            # update feature remained, and feats
            self.feature_remained = feats[-(self.left_context+self.right_context):]
            feats = feats_ctx.to(self.device)
        if self.downsampling > 1:
            last_remainder = 0 if self.feats_ctx_offset==0 else self.downsampling-self.feats_ctx_offset
            remainder = (feats.size(0)+last_remainder) % self.downsampling
            feats = feats[self.feats_ctx_offset::self.downsampling, :]
            self.feats_ctx_offset = remainder if remainder == 0 else self.downsampling-remainder
        return feats

    def decode_keywords(self, t, probs):
        absolute_time = t + self.total_frames
        # search next_hyps depend on current probs and hyps.
        next_hyps = ctc_prefix_beam_search(absolute_time,
                                           probs,
                                           self.cur_hyps,
                                           self.keywords_idxset,
                                           self.score_beam)
        # update cur_hyps. note: the hyps is sort by path score(pnb+pb), not the keywords' probabilities.
        cur_hyps = next_hyps[:self.path_beam]
        self.cur_hyps = cur_hyps

    def execute_detection(self, t):
        absolute_time = t + self.total_frames
        hit_keyword = None
        start = 0
        end = 0

        # hyps for detection
        hyps = [(y[0], y[1][0] + y[1][1], y[1][2]) for y in self.cur_hyps]

        # detect keywords in decoding paths.
        for one_hyp in hyps:
            prefix_ids = one_hyp[0]
            # path_score = one_hyp[1]
            prefix_nodes = one_hyp[2]
            assert len(prefix_ids) == len(prefix_nodes)
            for word in self.keywords_token.keys():
                lab = self.keywords_token[word]['token_id']
                offset = is_sublist(prefix_ids, lab)
                if offset != -1:
                    hit_keyword = word
                    start = prefix_nodes[offset]['frame']
                    end = prefix_nodes[offset + len(lab) - 1]['frame']
                    for idx in range(offset, offset + len(lab)):
                        self.hit_score *= prefix_nodes[idx]['prob']
                    break
            if hit_keyword is not None:
                self.hit_score = math.sqrt(self.hit_score)
                break

        duration = end - start
        if hit_keyword is not None:
            if self.hit_score >= self.threshold and \
                    self.min_frames <= duration <= self.max_frames \
                    and (self.last_active_pos==-1 or end-self.last_active_pos >= self.interval_frames):
                self.activated = True
                self.last_active_pos = end
                logging.info(
                    f"Frame {absolute_time} detect {hit_keyword} from {start} to {end} frame. "
                    f"duration {duration}, score {self.hit_score}, Activated.")

            elif self.last_active_pos>0 and end-self.last_active_pos < self.interval_frames:
                logging.info(
                    f"Frame {absolute_time} detect {hit_keyword} from {start} to {end} frame. "
                    f"but interval {end-self.last_active_pos} is lower than {self.interval_frames}, Deactivated. ")

            elif self.hit_score < self.threshold:
                logging.info(
                    f"Frame {absolute_time} detect {hit_keyword} from {start} to {end} frame. "
                    f"but {self.hit_score} is lower than {self.threshold}, Deactivated. ")

            elif self.min_frames > duration or duration > self.max_frames:
                logging.info(
                    f"Frame {absolute_time} detect {hit_keyword} from {start} to {end} frame. "
                    f"but {duration} beyond range({self.min_frames}~{self.max_frames}), Deactivated. ")

        self.result = {
            "state": 1 if self.activated else 0,
            "keyword": hit_keyword if self.activated else None,
            "start": start * self.resolution if self.activated else None,
            "end": end * self.resolution if self.activated else None,
            "score": self.hit_score if self.activated else None
        }

    def forward(self, wave_chunk):
        feature = self.accept_wave(wave_chunk)
        if feature is None or feature.size(0) < 1:
            return {}  # # the feature is not enough to get result.
        feature = feature.unsqueeze(0)   # add a batch dimension
        logits, self.in_cache = self.model(feature, self.in_cache)
        probs = logits.softmax(2)  # (batch_size, maxlen, vocab_size)
        probs = probs[0].cpu()   # remove batch dimension, move to cpu for ctc_prefix_beam_search
        for (t, prob) in enumerate(probs):
            t *= self.downsampling
            self.decode_keywords(t, prob)
            self.execute_detection(t)

            if self.activated:
                self.reset()
                # since a chunk include about 30 frames, once activated, we can jump the latter frames.
                # TODO: there should give another method to update result, avoiding self.result being cleared.
                break
        self.total_frames += len(probs) * self.downsampling  # update frame offset
        return self.result

    def reset(self):
        self.cur_hyps = [(tuple(), (1.0, 0.0, []))]
        self.activated = False
        self.hit_score = 1.0

    def reset_all(self):
        self.reset()
        self.wave_remained = np.array([])
        self.feature_remained = None
        self.feats_ctx_offset = 0  # after downsample, offset exist.
        self.in_cache = torch.zeros(0, 0, 0, dtype=torch.float)
        self.total_frames = 0   # frame offset, for absolute time
        self.last_active_pos = -1  # the last frame of being activated
        self.result = {}

def demo():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')

    kws = KeyWordSpotter(args.checkpoint,
                         args.config,
                         args.token_file,
                         args.lexicon_file,
                         args.threshold,
                         args.min_frames,
                         args.max_frames,
                         args.interval_frames,
                         args.score_beam_size,
                         args.path_beam_size,
                         args.gpu,
                         args.jit_model)

    # actually this could be done in __init__ method, we pull it outside for changing keywords more freely.
    kws.set_keywords(args.keywords)

    if args.wav_path:
        # Caution: input WAV should be standard 16k, 16 bits, 1 channel
        # In demo we read wave in non-streaming fashion.
        with wave.open(args.wav_path, 'rb') as fin:
            assert fin.getnchannels() == 1
            wav = fin.readframes(fin.getnframes())

        # We inference every 0.3 seconds, in streaming fashion.
        interval = int(0.3 * 16000) * 2
        for i in range(0, len(wav), interval):
            chunk_wav = wav[i: min(i + interval, len(wav))]
            result = kws.forward(chunk_wav)
            print(result)

    fout = None
    if args.result_file:
        fout = open(args.result_file, 'w', encoding='utf-8')

    if args.wav_scp:
        with open(args.wav_scp, 'r') as fscp:
            for line in fscp:
                line = line.strip().split()
                assert len(line) == 2, f"The scp should be in kaldi format: \"utt_name wav_path\", but got {line}"

                utt_name, wav_path = line[0], line[1]
                with wave.open(wav_path, 'rb') as fin:
                    assert fin.getnchannels() == 1
                    wav = fin.readframes(fin.getnframes())

                kws.reset_all()
                activated = False

                # We inference every 0.3 seconds, in streaming fashion.
                interval = int(0.3 * 16000) * 2
                for i in range(0, len(wav), interval):
                    chunk_wav = wav[i: min(i + interval, len(wav))]
                    result = kws.forward(chunk_wav)
                    if 'state' in result and result['state'] == 1:
                        activated = True
                        if fout:
                            hit_keyword = result['keyword']
                            hit_score = result['score']
                            fout.write('{} detected {} {:.3f}\n'.format(utt_name, hit_keyword, hit_score))

                if not activated:
                    if fout:
                        fout.write('{} rejected\n'.format(utt_name))


    if fout:
        fout.close()

if __name__ == '__main__':
    demo()
