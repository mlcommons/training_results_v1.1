# Copyright (C) 2021 Samsung Electronics Co. LTD

# This software is a property of Samsung Electronics.
# No part of this software, either material or conceptual may be copied or distributed, transmitted,
# transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
# electronic, mechanical, manual or otherwise, or disclosed
# to third parties without the express written permission of Samsung Electronics.

# The following items are modified and they can be claimed as properties of Samsung Electronics. 
# (1) Load splitting training data
# (2) Add A new local/group exchange padding method
# (3) Add NCCL warmup for group exchange padding
# (4) Add per-device local gradient clipping before all-reduce


# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2020 MLBenchmark Group. All rights reserved.

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

"""BERT Pretraining"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import h5py
import os
import glob
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
import logging
import math
import multiprocessing
import numpy as np
import os
import sys
import random
import re
import time
import inspect
from types import MethodType

from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from modeling import BertForPreTraining, BertConfig
from apex.multi_tensor_apply import multi_tensor_applier
from schedulers import LinearWarmupPolyDecayScheduler

import utils
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.autograd.profiler as prof

import amp_C
import apex_C

try:
    from apex import amp
    from apex.amp import _amp_state
    from apex.parallel import convert_syncbn_model
    from apex.parallel.distributed import flat_dist_call

    has_apex = True
except ImportError:
    has_apex = False

try:
    from torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook import PowerSGDState, powerSGD_hook
    from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import _allreduce_fut
    has_ddp_algo_hook = True
except ImportError:
    has_ddp_algo_hook = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

import types
from contextlib import suppress
from contextlib import contextmanager
from functools import partial
from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modeling import BertForPreTraining, BertConfig
from schedulers import LinearWarmUpScheduler, LinearWarmupPolyDecayScheduler
import mlperf_logger
from mhalib import *

# Global variables
skipped_steps = 0
global_grad_norm = 5.0
cached_batches = []
clipper = None


class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)


def create_pretraining_dataset(input_file, max_pred_length, shared_list, args, worker_init_fn):
    train_data = pretraining_dataset(input_files=input_file, max_pred_length=max_pred_length)
    if not args.use_split_data:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = utils.SplitRandomSampler(input_file, batch_ratio=list(map(int, args.split_batch_cnt)))
    train_dataloader = DataLoader(train_data, sampler=train_sampler,batch_size=args.train_batch_size, num_workers=4, worker_init_fn=worker_init_fn, pin_memory=True)

    return train_dataloader, input_file


def create_eval_dataset(args, worker_init_fn):
    eval_data = []
    for eval_file in sorted(os.listdir(args.eval_dir)):
        eval_file_path = os.path.join(args.eval_dir, eval_file)
        if os.path.isfile(eval_file_path) and 'part' in eval_file_path:
            eval_data.extend(pretraining_dataset(eval_file_path, max_pred_length=args.max_predictions_per_seq))
            if len(eval_data) > args.num_eval_examples:
                eval_data = eval_data[:args.num_eval_examples]
                break
    if torch.distributed.is_initialized():
        chunk_size = args.num_eval_examples // torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        remainder = args.num_eval_examples % torch.distributed.get_world_size()
        if rank < remainder:
            eval_data = eval_data[(chunk_size + 1) * rank: (chunk_size + 1) * (rank + 1)]
        else:
            eval_data = eval_data[chunk_size * rank + remainder: chunk_size * (rank + 1) + remainder]

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4, worker_init_fn=worker_init_fn, pin_memory=True)

    return eval_dataloader


def exchange_padding_fast(args, device, input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels, max_batch_size, group_pg=None):
    torch.cuda.nvtx.range_push('exchangepadding')
    pad_size = max_batch_size - input_ids.shape[0]
    if pad_size > 0:
        input_ids = F.pad(input_ids, (0, 0, 0, pad_size))
        segment_ids = F.pad(segment_ids, (0, 0, 0, pad_size))
        input_mask = F.pad(input_mask, (0, 0, 0, pad_size))
        masked_lm_labels = F.pad(masked_lm_labels, (0, 0, 0, pad_size))
        next_sentence_labels = F.pad(next_sentence_labels, (0, pad_size))

    if not args.group_exchange_padding:
        ngpus = torch.distributed.get_world_size()
        igpu = torch.distributed.get_rank()
    else:
        assert group_pg is not None
        ngpus = args.ngpus_per_group
        igpu = torch.distributed.get_rank() % args.ngpus_per_group

    nseqs = input_mask.shape[0]
    ntokensperseq = input_mask.shape[1]

    flattened_length_seq = nseqs * ntokensperseq
    flattened_length_nsp = nseqs

    def get_local_packet_size():
        return 4 * flattened_length_seq + flattened_length_nsp

    # Storing tensors in same order as arguments
    def encode_packet(input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels):

        packet = torch.zeros([get_local_packet_size()], device=device, dtype=torch.int16)

        curr_pos = 0

        packet[curr_pos:curr_pos + flattened_length_seq] = input_ids.view(-1)[:]
        curr_pos += flattened_length_seq

        packet[curr_pos:curr_pos + flattened_length_seq] = segment_ids.view(-1)[:]
        curr_pos += flattened_length_seq

        packet[curr_pos:curr_pos + flattened_length_seq] = input_mask.view(-1)[:]
        curr_pos += flattened_length_seq

        packet[curr_pos:curr_pos + flattened_length_seq] = masked_lm_labels.view(-1)[:]
        curr_pos += flattened_length_seq

        packet[curr_pos:curr_pos + flattened_length_nsp] = next_sentence_labels.view(-1)[:]

        return packet

    def decode_packet(flat_packet):
        packet = flat_packet.view(ngpus, get_local_packet_size())

        curr_pos = 0

        input_ids_ = packet[:, curr_pos:curr_pos + flattened_length_seq].contiguous().view(ngpus, nseqs, ntokensperseq)
        curr_pos += flattened_length_seq

        segment_ids_ = packet[:, curr_pos:curr_pos + flattened_length_seq].contiguous().view(ngpus, nseqs, ntokensperseq)
        curr_pos += flattened_length_seq

        input_mask_ = packet[:, curr_pos:curr_pos + flattened_length_seq].contiguous().view(ngpus, nseqs, ntokensperseq)
        curr_pos += flattened_length_seq

        masked_lm_labels_ = packet[:, curr_pos:curr_pos + flattened_length_seq].contiguous().view(ngpus, nseqs, ntokensperseq)
        curr_pos += flattened_length_seq

        next_sentence_labels_ = packet[:, curr_pos:curr_pos + flattened_length_nsp].contiguous().view(ngpus, nseqs)

        return input_ids_, segment_ids_, input_mask_, masked_lm_labels_, next_sentence_labels_

    tensors = encode_packet(input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels)

    tensors_ = torch.zeros([ngpus, get_local_packet_size()], device=device, dtype=torch.float16)
    tensors_ = list(torch.split(tensors_, 1))

    # torch.distributed.all_gather(tensors_, tensors.view(torch.float16))
    if not args.group_exchange_padding:
        torch.distributed.all_gather(tensors_, tensors.view(torch.float16))
    else:
        torch.distributed.all_gather(tensors_, tensors.view(torch.float16), group=group_pg)

    tensors_ = torch.stack(tensors_).view(torch.int16).long()
    input_ids_, segment_ids_, input_mask_, masked_lm_labels_, next_sentence_labels_ = decode_packet(tensors_)

    seqlens_, indices = torch.sort(input_mask_.sum(dim=2).view(-1), descending=True)
    if args.reverse_indices:
        indices = indices.view(-1, ngpus)
        indices[1::2,] = indices.view(-1, ngpus).flip([1])[1::2,]
        indices = indices.flatten()  # Only Torch version > 1.8.0 works

    if pad_size > 0:
        input_ids_sorted = input_ids_.view(ngpus * nseqs, ntokensperseq)[indices[:], :]
        segment_ids_sorted = segment_ids_.view(ngpus * nseqs, ntokensperseq)[indices[:], :]
        input_mask_sorted = input_mask_.view(ngpus * nseqs, ntokensperseq)[indices[:], :]
        masked_lm_labels_sorted = masked_lm_labels_.view(ngpus * nseqs, ntokensperseq)[indices[:], :]
        next_sentence_labels_sorted = next_sentence_labels_.view(ngpus * nseqs)[indices[:]]
        # we need to remove the empty sequences we added to the batch
        valid_idx = seqlens_.view(nseqs, ngpus)[:, igpu] > 0
        input_ids_sorted = input_ids_sorted.view(nseqs, ngpus, ntokensperseq)[valid_idx, igpu, :].contiguous()
        segment_ids_sorted = segment_ids_sorted.view(nseqs, ngpus, ntokensperseq)[valid_idx, igpu, :].contiguous()
        input_mask_sorted = input_mask_sorted.view(nseqs, ngpus, ntokensperseq)[valid_idx, igpu, :].contiguous()
        masked_lm_labels_sorted = masked_lm_labels_sorted.view(nseqs, ngpus, ntokensperseq)[valid_idx, igpu, :].contiguous()
        next_sentence_labels_sorted = next_sentence_labels_sorted.view(nseqs, ngpus)[valid_idx, igpu].contiguous()
    else:
        indices_ = indices.view(nseqs, ngpus)[:, igpu]
        input_ids_sorted = input_ids_.view(nseqs * ngpus, ntokensperseq)[indices_, :].contiguous()
        segment_ids_sorted = segment_ids_.view(nseqs * ngpus, ntokensperseq)[indices_, :].contiguous()
        input_mask_sorted = input_mask_.view(nseqs * ngpus, ntokensperseq)[indices_, :].contiguous()
        masked_lm_labels_sorted = masked_lm_labels_.view(nseqs * ngpus, ntokensperseq)[indices_, :].contiguous()
        next_sentence_labels_sorted = next_sentence_labels_.view(nseqs * ngpus)[indices_].contiguous()

    torch.cuda.nvtx.range_pop()
    return input_ids_sorted, segment_ids_sorted, input_mask_sorted, masked_lm_labels_sorted, next_sentence_labels_sorted


class pretraining_dataset(Dataset):
    def __init__(self, input_files, max_pred_length):
        self.input_files = input_files
        self.max_pred_length = max_pred_length
        self.inputs = None
        # TODO : Concurrently read?
        if not isinstance(input_files, list):
            input_files = [input_files]
        for input_file in input_files:
            f = h5py.File(input_file, "r")
            keys = ['input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids',
                    'next_sentence_labels']
            if self.inputs is not None:
                self.inputs = [np.concatenate((self.inputs[i], f[key][:]), axis=0) for i, key in enumerate(keys)]
            else:
                self.inputs = [np.asarray(f[key][:]) for i, key in enumerate(keys)]
            f.close()


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])

    def __getitem__(self, index):
        [input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels] = [
            torch.from_numpy(input[index].astype(np.int64)) if indice < 5 else torch.from_numpy(
                np.asarray(input[index].astype(np.int64))) for indice, input in enumerate(self.inputs)]

        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        index = self.max_pred_length
        # store number of  masked tokens in index
        padded_mask_indices = (masked_lm_positions == 0).nonzero()
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        return [input_ids, segment_ids, input_mask,masked_lm_labels, next_sentence_labels]


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--input_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain .hdf5 files  for the task.")

    parser.add_argument("--bert_model", default="bert-large-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    # Other parameters
    parser.add_argument("--eval_dir",
                        default=None,
                        type=str,
                        help="The eval data dir. Should contain .hdf5 files  for the task.")
    parser.add_argument("--eval_iter_start_samples",
                        default=3000000,
                        type=int,
                        help="Sample to begin performing eval.")
    parser.add_argument("--eval_iter_samples",
                        default=-1,
                        type=int,
                        help="If set to -1, disable eval, \
                        else evaluate every eval_iter_samples during training")
    parser.add_argument("--num_eval_examples",
                        default=10000,
                        type=int,
                        help="number of eval examples to run eval on")
    parser.add_argument("--cache_eval_data",
                        default=False,
                        action='store_true',
                        help="whether to cache evaluation data on GPU")

    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="The initial checkpoint to start training from.")
    parser.add_argument("--init_tf_checkpoint",
                        default=None,
                        type=str,
                        help="The initial TF checkpoint to start training from.")

    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_predictions_per_seq",
                        default=76,
                        type=int,
                        help="The maximum total of masked tokens in input sequence")
    parser.add_argument("--train_batch_size",
                        default=18,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=128,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=4e-5,
                        type=float,
                        help="The initial learning rate for ADAM.")
    parser.add_argument("--end_learning_rate",
                        default=0.0,
                        type=float,
                        help="The end learning rate for ADAM.")
    parser.add_argument("--weight_decay_rate",
                        default=0.01,
                        type=float,
                        help="weight decay rate for ADAM.")
    parser.add_argument("--opt_lamb_beta_1",
                        default=0.9,
                        type=float,
                        help="ADAM beta1.")
    parser.add_argument("--opt_lamb_beta_2",
                        default=0.999,
                        type=float,
                        help="ADAM beta2.")
    parser.add_argument("--epsilon",
                        default=1e-6,
                        type=float,
                        help="optimizer epsilon")
    parser.add_argument("--max_steps",
                        default=1536,
                        type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument("--max_samples_termination",
                        default=14000000,
                        type=float,
                        help="Total number of training samples to run.")
    parser.add_argument("--warmup_proportion",
                        default=0.01,
                        type=float,
                        help="Proportion of optimizer update steps to perform linear learning rate warmup for. "
                             "Typically 1/8th of steps for Phase2")
    parser.add_argument("--warmup_steps",
                        default=0,
                        type=float,
                        help="Number of optimizer update steps to perform linear learning rate warmup for. "
                             "Typically 1/8th of steps for Phase2")
    parser.add_argument("--start_warmup_step",
                        default=0,
                        type=float,
                        help="Starting step for warmup. ")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--use_apex_amp',
                        default=False,
                        action='store_true',
                        help="Whether to use APEX Configuration")
    parser.add_argument('--use_torch_amp',
                        default=False,
                        action='store_true',
                        help="Whether to use Pytorch AMP")
    parser.add_argument('--loss_scale',
                        type=float, default=0.0,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--log_freq',
                        type=float, default=1.0,
                        help='frequency of logging loss.')
    parser.add_argument("--resume_from_checkpoint",
                        default=False,
                        action='store_true',
                        help="Whether to resume training from checkpoint. If set, precedes init_checkpoint/init_tf_checkpoint")
    parser.add_argument('--keep_n_most_recent_checkpoints',
                        type=int,
                        default=20,
                        help="Number of checkpoints to keep (rolling basis).")
    parser.add_argument('--num_samples_per_checkpoint',
                        type=int,
                        default=500000,
                        help="Number of update steps until a model checkpoint is saved to disk.")
    parser.add_argument('--min_samples_to_start_checkpoints',
                        type=int,
                        default=3000000,
                        help="Number of update steps until model checkpoints start saving to disk.")
    parser.add_argument('--skip_checkpoint',
                        default=False,
                        action='store_true',
                        help="Whether to save checkpoints")
    parser.add_argument('--phase2',
                        default=False,
                        action='store_true',
                        help="Only required for checkpoint saving format")
    parser.add_argument('--allreduce_post_accumulation',
                        default=False,
                        action='store_true',
                        help="Whether to do allreduces during gradient accumulation steps.")
    parser.add_argument('--allreduce_post_accumulation_fp16',
                        default=False,
                        action='store_true',
                        help="Whether to do fp16 allreduce post accumulation.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--unpad",
                        default=False,
                        action='store_true',
                        help="Whether to run with unpadding.")
    parser.add_argument("--unpad_fmha",
                        default=False,
                        action='store_true',
                        help="Whether to run FMHA with unpadding.")
    parser.add_argument("--pad",
                        default=False,
                        action='store_true',
                        help="Whether to pad tokens.")
    parser.add_argument("--enable_fuse_dropout",
                        default=False,
                        action='store_true',
                        help="Whether to disable fusion of attention mask to softmax and dropout.")
    parser.add_argument("--disable_fuse_mask",
                        default=False,
                        action='store_true',
                        help="Whether to disable fusion of the attention mask to softmax.")
    parser.add_argument("--disable_fuse_scale",
                        default=False,
                        action='store_true',
                        help="Whether to disable fusion of the scaling to BMM1.")
    parser.add_argument("--disable_fuse_qkv",
                        default=False,
                        action='store_true',
                        help="Whether to disable fusion of the QKV GEMMs.")
    parser.add_argument("--disable_apex_softmax",
                        default=False,
                        action='store_true',
                        help="Whether to disable apex softmax.")
    parser.add_argument("--enable_stream",
                        default=False,
                        action='store_true',
                        help="Enable use of streams for pad case.")
    parser.add_argument("--fused_mha",
                        default=False,
                        action='store_true',
                        help="Whether to run with optimizations.")
    parser.add_argument("--fused_gelu_bias",
                        default=False,
                        action='store_true',
                        help="Whether to run with optimizations.")
    parser.add_argument("--dense_seq_output",
                        default=False,
                        action='store_true',
                        help="Whether to run with optimizations.")
    parser.add_argument("--use_env",
                        action='store_true',
                        help="Whether to read local rank and global rank from ENVVAR")
    parser.add_argument('--bert_config_path',
                        type=str,
                        default="/workspace/phase1",
                        help="Path bert_config.json is located in")
    parser.add_argument('--target_mlm_accuracy',
                        type=float,
                        default=0.0,
                        help="Stop training after reaching this Masked-LM accuracy")
    parser.add_argument('--train_mlm_accuracy_window_size',
                        type=int,
                        default=0,
                        help="Average accuracy over this amount of batches before performing a stopping criterion test")
    parser.add_argument('--num_epochs_to_generate_seeds_for',
                        type=int,
                        default=2,
                        help="Number of epochs to plan seeds for. Same set across all workers.")
    parser.add_argument("--record_gradients",
                        default=False,
                        action='store_true',
                        help="Whether to record gradient distribution")
    parser.add_argument("--local_gradient_clip",
                        default=False,
                        action='store_true',
                        help="Per-device clip norm, if enabled. It will disable global clip_norm")
    parser.add_argument("--baseline",
                        default=False,
                        action='store_true',
                        help="MLPerf original implementation")
    parser.add_argument("--profile",
                        default=False,
                        action='store_true',
                        help="Whether to profile")
    parser.add_argument("--exchange_padding",
                        default=False,
                        action='store_true',
                        help="Whether to run with exchange_padding.")
    parser.add_argument('--distributed_lamb',
                        default=False,
                        action='store_true',
                        help="Whether to use distributed lamb.")
    parser.add_argument('--optimizer',
                        type=str,
                        default="FusedLAMB",
                        help="The name of optimizer to use")
    parser.add_argument('--use_split_data',
                        default=False,
                        action='store_true',
                        help="Whether to use splitting bin dataset for training")
    parser.add_argument('--split_batch_cnt', nargs='+',
                        help='Count to load from each fixed bins')
    parser.add_argument('--reverse_indices',
                        default=False,
                        action='store_true',
                        help="Whether reverse indices of even row when exchange padding")
    parser.add_argument("--group_exchange_padding",
                        default=False,
                        action='store_true',
                        help="Whether to use group exchange padding.")
    parser.add_argument('--ngpus_per_group',
                        type=int,
                        default=8,
                        help="Number of GPUs used for group exchange padding.")
    parser.add_argument("--use_partial_data",
                        default=False,
                        action='store_true',
                        help="use partial (not whole) dataset for training.")
    parser.add_argument("--lr_max_steps",
                        default=179,
                        type=float,
                        help="If the training step is less than or equal to lr_max_steps, the lr is calculated by the lr scheduler; "
                             "otherwise, lr is end lr")
    args = parser.parse_args()

    # Check we've been given a checkpoint
    # assert args.init_checkpoint is not None or args.init_tf_checkpoint is not None or found_resume_checkpoint(args), \
    #     "Must specify --init_checkpoint, --init_tf_checkpoint or have ckpt to resume from in --output_dir of the form *.pt"

    assert not (args.init_checkpoint is not None and args.init_tf_checkpoint is not None), \
        "Can only specify one of --init_checkpoint and --init_tf_checkpoint"

    return args


# Returns true only if resuming from a checkpoint found in output_dir.
# init_checkpoint and init_tf_checkpoint are not considered
def found_resume_checkpoint(args):
    if args.phase2:
        checkpoint_str = "phase2_ckpt*.pt"
    else:
        checkpoint_str = "phase1_ckpt*.pt"
    return args.resume_from_checkpoint and len(glob.glob(os.path.join(args.output_dir, checkpoint_str))) > 0


def setup_training(args):
    assert (torch.cuda.is_available())

    if args.local_rank == -1:
        device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl', init_method='env://', rank=args.rank)
        args.n_gpu = torch.distributed.get_world_size()

    print("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(args.gradient_accumulation_steps))
    if args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, batch size {} should be divisible".format(args.gradient_accumulation_steps, args.train_batch_size))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if not (args.do_train or (args.eval_dir and args.eval_iter_samples <= 0)):
        raise ValueError(" `do_train`  or should be in offline eval mode")

    if not args.resume_from_checkpoint or not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    return device, args


def remap_attn_parameters(model_dict):
    res_dict = OrderedDict()
    for k in model_dict:
        if 'attention' in k:
            if 'self.query.weight' in k:
                new_k = k.replace('self.query.weight', 'multi_head_attention.q_weight')
            elif 'self.key.weight' in k:
                new_k = k.replace('self.key.weight', 'multi_head_attention.k_weight')
            elif 'self.value.weight' in k:
                new_k = k.replace('self.value.weight', 'multi_head_attention.v_weight')
            elif 'self.query.bias' in k:
                new_k = k.replace('self.query.bias', 'multi_head_attention.q_bias')
            elif 'self.key.bias' in k:
                new_k = k.replace('self.key.bias', 'multi_head_attention.k_bias')
            elif 'self.value.bias' in k:
                new_k = k.replace('self.value.bias', 'multi_head_attention.v_bias')
            elif 'output.dense.weight' in k:
                new_k = k.replace('output.dense.weight', 'multi_head_attention.out_proj_weight')
            elif 'output.dense.bias' in k:
                new_k = k.replace('output.dense.bias', 'multi_head_attention.out_proj_bias')
            elif 'output.LayerNorm.weight' in k:
                new_k = k.replace('output.LayerNorm.weight', 'layer_norm.weight')
            elif 'output.LayerNorm.bias' in k:
                new_k = k.replace('output.LayerNorm.bias', 'layer_norm.bias')
            else:
                new_k = k
        else:
            new_k = k
        res_dict[new_k] = model_dict[k]
    model_dict.clear()
    return res_dict


def prepare_model_and_optimizer(args, device):
    global_step = 0
    args.resume_step = 0
    checkpoint = None

    config = BertConfig.from_json_file(args.bert_config_path)
    config.fused_mha = args.fused_mha
    config.fused_gelu_bias = args.fused_gelu_bias
    config.fused_layer_norm = args.fp16 and args.use_apex_amp
    config.dense_seq_output = args.dense_seq_output
    config.unpad = args.unpad
    config.unpad_fmha = args.unpad_fmha
    config.pad = args.pad
    config.fp16 = args.fp16
    config.fuse_qkv = not args.disable_fuse_qkv
    config.fuse_scale = not args.disable_fuse_scale
    config.fuse_mask = not args.disable_fuse_mask
    config.fuse_dropout = args.enable_fuse_dropout
    config.apex_softmax = not args.disable_apex_softmax
    config.enable_stream = args.enable_stream

    if config.fuse_mask == True: config.apex_softmax = True
    if config.pad == False: config.enable_stream = True
    if config.unpad == True: config.fused_mha = False

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    # Load from Pyt checkpoint - either given as init_checkpoint, or picked up from output_dir if found
    if args.init_checkpoint or found_resume_checkpoint(args):
        # Prepare model

        model = BertForPreTraining(config)
        if args.init_checkpoint is None:  # finding checkpoint in output_dir
            checkpoint_str = "phase2_ckpt_*.pt" if args.phase2 else "phase1_ckpt_*.pt"
            model_names = [f for f in glob.glob(os.path.join(args.output_dir, checkpoint_str))]
            global_step = max([int(x.split('.pt')[0].split('_')[-1].strip()) for x in model_names])
            args.resume_step = global_step  # used for throughput computation

            resume_init_checkpoint = os.path.join(args.output_dir, checkpoint_str.replace("*", str(global_step)))
            print("Setting init checkpoint to %s - which is the latest in %s" % (resume_init_checkpoint, args.output_dir))
            checkpoint = torch.load(resume_init_checkpoint, map_location="cpu")
        else:
            checkpoint = torch.load(args.init_checkpoint, map_location="cpu")["model"]

        # Fused MHA requires a remapping of checkpoint parameters
        if config.fused_mha:
            checkpoint_remapped = remap_attn_parameters(checkpoint)
            model.load_state_dict(checkpoint_remapped, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=True)
    else:  # Load from TF Checkpoint
        model = BertForPreTraining(config)

    print(f"Model Parameters : {sum([p.numel() for p in model.parameters() if p.requires_grad])}")

    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay_rate},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    mlperf_logger.log_event(key=mlperf_logger.constants.OPT_BASE_LR, value=args.learning_rate, sync=False)

    optimizer = utils.get_optimizer(args.optimizer, optimizer_grouped_parameters, lr=args.learning_rate,
                                    eps=args.epsilon, betas=(args.opt_lamb_beta_1, args.opt_lamb_beta_2), wd=args.weight_decay_rate)

    if hasattr(optimizer, 'set_global_scale'):  # Stands for DistributedLAMB
        scale = torch.full((1,), float(os.getenv("INIT_LOSS_SCALE", 2 ** 17)), dtype=torch.float32, device=device)
        optimizer.set_global_scale(scale)
    if hasattr(optimizer, '_clip_grad_norm'):  # Stands for FusedLAMB, FusedAdam, DistributedLAMB
        optimizer._clip_grad_norm = not args.local_gradient_clip

    mlperf_logger.log_event(key='opt_epsilon', value=args.epsilon, sync=False)
    mlperf_logger.log_event(key='opt_lamb_beta_1', value=args.opt_lamb_beta_1, sync=False)
    mlperf_logger.log_event(key='opt_lamb_beta_2', value=args.opt_lamb_beta_2, sync=False)
    mlperf_logger.log_event(key='opt_lamb_weight_decay_rate', value=args.weight_decay_rate, sync=False)

    if args.warmup_steps == 0:
        args.warmup_steps = int(args.lr_max_steps * args.warmup_proportion)
    lr_scheduler = LinearWarmupPolyDecayScheduler(optimizer, start_warmup_steps=args.start_warmup_step,
                                                  warmup_steps=args.warmup_steps, total_steps=args.lr_max_steps,
                                                  end_learning_rate=args.end_learning_rate, degree=1.0)

    if args.fp16:
        if args.use_apex_amp:
            if args.loss_scale == 0:
                model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale="dynamic", master_weights=not args.distributed_lamb)
            else:
                model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=args.loss_scale, master_weights=not args.distributed_lamb)
            amp._amp_state.loss_scalers[0]._loss_scale = float(os.getenv("INIT_LOSS_SCALE", int(2 ** 17)))
            if utils.is_main_process():
                print('Using NVIDIA APEX AMP. Training in mixed precision.')
        else:
            if utils.is_main_process():
                print('Using native PyTorch AMP. Training in mixed precision.')
    else:
        if utils.is_main_process():
            print('AMP not enabled. Training in float32.')

    if found_resume_checkpoint(args):
        assert False, "code path not tested with cuda graphs"
        optimizer.load_state_dict(checkpoint['optimizer'])

        if args.fp16 and not args.distributed_lamb:
            optimizer._lazy_init_maybe_master_weights()
            optimizer._amp_stash.lazy_init_called = True
            optimizer.load_state_dict(checkpoint['optimizer'])
            for param, saved_param in zip(amp.master_params(optimizer), checkpoint['master params']):
                param.data.copy_(saved_param.data)


    if args.local_rank != -1:
        if not args.distributed_lamb and not args.baseline:
            model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, bucket_cap_mb=25, gradient_as_bucket_view=args.use_apex_amp)
            if has_ddp_algo_hook and args.local_gradient_clip and isinstance(model, DDP):
                loss_scaler = _amp_state.loss_scalers[0]
                if args.local_gradient_clip:
                    if args.gradient_accumulation_steps > 1:  # For 64 GPUs, span-size 16
                        class local_gradient_clipper(object):  # For 1.9.0
                            def __init__(self, model):
                                super().__init__()
                                self.model = model
                                self.master_grads = [None] * 23
                                self.prev_master_grads = [None] * 23
                                self.overflow_buf = torch.cuda.IntTensor([0])
                                self.prev_overflow_buf = torch.cuda.IntTensor([0])

                            def allreduce_hook(self, state, bucket):  # WARN: no_sync, should catch manual way.
                                process_group = torch.distributed.group.WORLD
                                tensor = bucket.get_tensor()
                                master_grad = self.master_grads[bucket.get_index()]

                                if not self.overflow_buf:
                                    scaled_grad_norm, _ = amp_C.multi_tensor_l2norm(65536, self.overflow_buf, [[tensor]],False)
                                    coefficient = args.gradient_accumulation_steps*math.sqrt(22)
                                    scaled_grad_norm.clamp_(min=loss_scaler._loss_scale / math.sqrt(22))  # Because of Bucket Count, Set Grad Norm ~= 1
                                    if master_grad is None:  # TODO : Remove initialization?
                                        master_grad = self.master_grads[bucket.get_index()] = torch.empty(tensor.numel(), device='cuda', dtype=torch.float32)
                                        amp_C.multi_tensor_scale(65536, self.overflow_buf, [[tensor], [master_grad]], 1 / scaled_grad_norm.mul_(coefficient))
                                    elif tensor is not None:  # Weired...
                                        amp_C.multi_tensor_axpby(65536, self.overflow_buf,[[tensor], [master_grad], [master_grad]], 1 / scaled_grad_norm.mul_(coefficient), 1.0, 0)

                                def noop(fut):
                                    if bucket.is_the_last_bucket_to_allreduce():
                                        self.overflow_buf.zero_()
                                    return [fut.value()[0]]

                                if self.model.require_backward_grad_sync:  # update_step
                                    if not self.overflow_buf and master_grad is not None:
                                        amp_C.multi_tensor_scale(65536, self.overflow_buf, [[master_grad], [tensor]], loss_scaler.loss_scale() / (torch.distributed.get_world_size()))
                                    self.master_grads[bucket.get_index()] = None
                                    fut = torch.distributed.all_reduce(tensor, group=process_group, async_op=True).get_future()
                                    return fut.then(noop)
                                else:  # no_sync
                                    tensor.zero_()
                                    fut = torch.futures.Future()
                                    fut.set_result(bucket.get_tensor())
                                    return fut.then(noop)
                        clipper = local_gradient_clipper(model)
                        model.register_comm_hook(torch.distributed.group.WORLD, hook=clipper.allreduce_hook)
                    else:
                        overflow_buf = torch.cuda.IntTensor([0])
                        def allreduce_hook(state, bucket):  # For 256 GPUs > KPI, No Accumulation steps
                            process_group = torch.distributed.group.WORLD
                            tensor = bucket.get_tensor()
                            scaled_grad_norm, _ = multi_tensor_applier(amp_C.multi_tensor_l2norm, overflow_buf, [[tensor]],False)
                            scaled_grad_norm.clamp_(loss_scaler._loss_scale / math.sqrt(22))
                            amp_C.multi_tensor_scale(65536, overflow_buf, [[tensor], [tensor]], 1 / scaled_grad_norm.mul_(4908.98/loss_scaler._loss_scale))
                            fut = torch.distributed.all_reduce(tensor, group=process_group, async_op=True).get_future()
                            def noop(fut):
                                return [fut.value()[0]]
                            return fut.then(noop)
                        model.register_comm_hook(torch.distributed.group.WORLD, hook=allreduce_hook)
        else:
            flat_dist_call([param.data for param in model.parameters()], torch.distributed.broadcast, (0,))
            model.require_backward_grad_sync = True

            @contextmanager
            def custom_no_sync(self):
                old_require_backward_grad_sync = self.require_backward_grad_sync
                self.require_backward_grad_sync = False
                try:
                    yield
                finally:
                    self.require_backward_grad_sync = old_require_backward_grad_sync

            model.no_sync = partial(custom_no_sync, model)

    return model, optimizer, lr_scheduler, checkpoint, global_step


def take_optimizer_step(args, optimizer, model, overflow_buf, global_step):
    if args.allreduce_post_accumulation:
        # manually allreduce gradients after all accumulation steps
        # check for Inf/NaN
        # 1. allocate an uninitialized buffer for flattened gradient
        scaler = _amp_state.loss_scalers[0]
        master_grads = [p.grad for p in amp.master_params(optimizer) if p.grad is not None]
        flat_grad_size = sum(p.numel() for p in master_grads)
        allreduce_dtype = torch.float16 if args.allreduce_post_accumulation_fp16 else torch.float32
        flat_raw = torch.empty(flat_grad_size, device='cuda', dtype=allreduce_dtype)

        if args.local_gradient_clip:
            grad_norm, _ = multi_tensor_applier(amp_C.multi_tensor_l2norm, overflow_buf,[master_grads], False)
            grad_norm.clamp_(min=1.0)
            amp_C.multi_tensor_scale(65536, overflow_buf, [master_grads, master_grads], 1 / grad_norm)
        # 2. combine unflattening and predivision of unscaled 'raw' gradient
        allreduced_views = apex_C.unflatten(flat_raw, master_grads)
        overflow_buf.zero_()
        amp_C.multi_tensor_scale(65536, overflow_buf, [master_grads, allreduced_views],scaler.loss_scale() / (torch.distributed.get_world_size()))
        # 3. sum gradient across ranks. Because of the predivision, this averages the gradient
        torch.distributed.all_reduce(flat_raw)
        # 4. combine unscaling and unflattening of allreduced gradient
        overflow_buf.zero_()
        amp_C.multi_tensor_scale(65536, overflow_buf, [allreduced_views, master_grads], 1. / scaler.loss_scale())
        # 5. update loss scale
        scaler = _amp_state.loss_scalers[0]
        old_overflow_buf = scaler._overflow_buf
        scaler._overflow_buf = overflow_buf
        had_overflow = scaler.update_scale()
        scaler._overfloat_buf = old_overflow_buf
        # 6. call optimizer step function
        if had_overflow == 0:
            optimizer.step()
        else:
            print("Gradient overflow.  Skipping step, Loss scale reduces to {}".format(scaler.loss_scale()))
            if _amp_state.opt_properties.master_weights:
                for param in optimizer._amp_stash.all_fp32_from_fp16_params:
                    param.grad = None
        for param in model.parameters():
            param.grad = None

    else:
        optimizer.step()
        for param in model.parameters():
            param.grad = None


def run_eval(model, eval_dataloader, device, num_eval_examples, first_eval=False, use_cache=False,
             amp_autocast=suppress):
    model.eval()

    total_eval_loss, total_eval_mlm_acc = 0.0, 0.0
    total_masked = 0

    # on first eval, load and cache data on GPU
    if first_eval and use_cache:
        for batch in eval_dataloader:
            cached_batches.append([t.to(device) for t in batch])

    with torch.no_grad():
        for batch in cached_batches if use_cache else eval_dataloader:
            if not use_cache:
                batch = [t.to(device, non_blocking=True) for t in batch]
            input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch
            with amp_autocast():
                loss, mlm_acc, num_masked = model(input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels)
            total_eval_loss += loss * num_masked
            total_eval_mlm_acc += mlm_acc * num_masked
            total_masked += num_masked
    model.train()

    # total_eval_mlm_acc and total_eval_loss are already tensors, total_masked is not
    total_masked = torch.tensor(total_masked, device=device, dtype=torch.int64)

    if torch.distributed.is_initialized():
        # Collect total scores from all ranks
        torch.distributed.all_reduce(total_eval_mlm_acc, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_eval_loss, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_masked, op=torch.distributed.ReduceOp.SUM)

    # Average by number of examples
    total_eval_mlm_acc /= total_masked
    total_eval_loss /= total_masked

    return total_eval_loss.item(), total_eval_mlm_acc.item()


def main():
    global skipped_steps
    global global_grad_norm
    args = parse_arguments()
    print("args", args)

    status = 'aborted'  # later set to 'success' if termination criteria met

    mlperf_logger.log_start(key=mlperf_logger.constants.INIT_START, log_all_ranks=True, sync=False)

    if args.use_env and 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])

    # Note: setting the global rank for default distributed process group
    # and so that all of gpu'rank keep in order based on hosts and gpus
    if args.use_env and 'RANK' in os.environ:
        args.rank = int(os.environ['RANK'])
    else:
        print("Should use --use_env and set environment variable: 'RANK' for group exchange padding")
        exit(1)

    device, args = setup_training(args)

    if args.exchange_padding and args.group_exchange_padding:
        print("Should use --exchange_padding or --group_exchange_padding, Not both of them")
        exit(1)

    cur_node_pg = None
    cur_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    process_groups = list()
    if args.group_exchange_padding:
        n_gpus_per_group = args.ngpus_per_group
        rank_list = np.arange(world_size).reshape((int(world_size / n_gpus_per_group), n_gpus_per_group)).tolist()
        for ranks in rank_list:
            process_groups.append(torch.distributed.new_group(ranks=ranks, backend='nccl'))
        cur_node_pg = [process_groups[idx] for idx, ranks in enumerate(rank_list) if cur_rank in ranks][0]

    mlperf_logger.mlperf_submission_log('bert')

    worker_seeds, shuffling_seeds = utils.setup_seeds(args.seed, args.num_epochs_to_generate_seeds_for, device)
    worker_seed = worker_seeds[torch.distributed.get_rank()]

    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    worker_init = WorkerInitObj(worker_seed)

    mlperf_logger.log_event(key=mlperf_logger.constants.SEED, value=args.seed, sync=False)
    mlperf_logger.log_event(key=mlperf_logger.constants.GLOBAL_BATCH_SIZE, value=global_batch_size(args), sync=False)
    mlperf_logger.log_event(key='gradient_accumulation_steps', value=args.gradient_accumulation_steps, sync=False)
    mlperf_logger.log_event(key='max_predictions_per_seq', value=args.max_predictions_per_seq, sync=False)
    mlperf_logger.log_event(key='opt_learning_rate_training_steps', value=args.max_steps, sync=False)
    mlperf_logger.log_event(key='num_warmup_steps', value=args.warmup_steps, sync=False)

    if utils.is_main_process():
        print("parsed args:")
        print(args)

    # Prepare optimizer
    model, optimizer, lr_scheduler, checkpoint, global_step = prepare_model_and_optimizer(args, device)
    samples_trained = global_step * args.train_batch_size * args.gradient_accumulation_steps * args.n_gpu
    if lr_scheduler.start_warmup_steps or lr_scheduler.num_warmup_updates:
        lr_scheduler.step()  # First Step to fit initial warmup lr

    if args.unpad:
        torch.cuda.synchronize()
        InitMHACUDAExtension()
        torch.cuda.synchronize()

    final_loss = float("inf")
    train_time_raw = float("inf")
    raw_train_start = time.time()

    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if args.use_apex_amp:
        loss_scaler = _amp_state.loss_scalers[0]
    elif args.use_torch_amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = torch.cuda.amp.GradScaler()

    model.train()
    most_recent_ckpts_paths = []
    average_loss = 0.0  # averaged loss every args.log_freq steps
    epoch = 1
    training_steps = 0
    end_training, converged = False, False
    samples_trained_prev = 0
    eval_count = 0
    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

    pool = ProcessPoolExecutor(1)

    if args.target_mlm_accuracy:
        if args.train_mlm_accuracy_window_size > 0:
            accuracy_scores = []
            avg_mlm_accuracy = torch.Tensor([0]).cuda()

    first_epoch = True
    if found_resume_checkpoint(args):
        f_start_id = checkpoint['files'][0]
        files = checkpoint['files'][1:]
        num_files = len(files)
    else:
        if not args.use_split_data:
            files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f)) and 'part' in f]
            files.sort()
            num_files = len(files)
            if not args.use_partial_data:
                random.Random(shuffling_seeds[epoch]).shuffle(files)
        else:
            num_files = 99999
            file_lists = []
            for it in sorted(os.scandir(args.input_dir), key=lambda e: int(e.name)):
                if it.is_dir():
                    files = [os.path.join(it.path, f) for f in os.listdir(it.path) if os.path.isfile(os.path.join(it.path, f)) and 'part' in f]
                    files.sort()
                    if not args.use_partial_data:
                        random.Random(shuffling_seeds[epoch]).shuffle(files)
                    file_lists.append(files)
                    num_files = min(num_files, len(files))

        f_start_id = 0

    # Dummy Batch First from MLPerf v1.0 code
    batch_gpu_placeholder = [
        torch.ones(args.train_batch_size, args.max_seq_length, dtype=torch.int64, device=device),
        torch.ones(args.train_batch_size, args.max_seq_length, dtype=torch.int64, device=device),
        torch.ones(args.train_batch_size, args.max_seq_length, dtype=torch.int64, device=device),
        torch.ones(args.train_batch_size, args.max_seq_length, dtype=torch.int64, device=device),
        torch.ones(args.train_batch_size, dtype=torch.int64, device=device),
    ]
    input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch_gpu_placeholder
    for i in range(10):  # 10 Times need to initialize NCCL
        # Exchange padding warm up
        if args.exchange_padding or args.group_exchange_padding:
            input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels \
                = exchange_padding_fast(args, device, input_ids, segment_ids, input_mask, masked_lm_labels,
                                        next_sentence_labels, args.train_batch_size, group_pg=cur_node_pg)

        with amp_autocast():
            loss, mlm_acc, _ = model(input_ids=input_ids, token_type_ids=segment_ids,attention_mask=input_mask, masked_lm_labels=masked_lm_labels,next_sentence_label=next_sentence_labels)
        if hasattr(optimizer, '_lazy_init_stage1'):
            optimizer._lazy_init_stage1()
        if loss_scaler:
            if args.use_apex_amp:
                with amp.scale_loss(loss, optimizer, delay_unscale=True) as scaled_loss:
                    scaled_loss.backward(create_graph=second_order)
            else:
                loss_scaler.scale(loss).backward(create_graph=second_order)
        else:
            loss.backward(create_graph=second_order)
        if hasattr(optimizer, '_lazy_init_stage2'):
            optimizer._lazy_init_stage2()
        optimizer.zero_grad()
    del batch_gpu_placeholder

    mlperf_logger.log_end(key=mlperf_logger.constants.INIT_STOP, sync=False)
    mlperf_logger.log_start(key=mlperf_logger.constants.RUN_START, sync=True)
    mlperf_logger.barrier()

    skipped_steps = torch.zeros((1,), dtype=torch.int, device=device)
    # Start prefetching eval dataset
    if args.eval_dir:
        eval_dataset_future = pool.submit(create_eval_dataset, args, worker_init_fn=worker_init)

    now_step, now_skipped, skip_interval = 0, 0, 0
    while global_step < args.max_steps and not end_training:
        mlperf_logger.log_start(key=mlperf_logger.constants.EPOCH_START,metadata={'epoch_num': epoch}, sync=False)
        mlperf_logger.log_start(key=mlperf_logger.constants.BLOCK_START,metadata={'first_epoch_num': epoch,'epoch_count': 1},sync=False)
        if utils.is_main_process():
            print("parsed args:")
            print(args)
            now_time = time.time()
            print("epoch:", epoch)

        thread = None

        # Reshuffle file list on subsequent epochs
        if not first_epoch:
            files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f)) and 'part' in f]
            files.sort()
            num_files = len(files)
            random.Random(shuffling_seeds[epoch]).shuffle(files)
            f_start_id = 0

        first_epoch = False

        shared_file_list = {}

        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > num_files:
            remainder = torch.distributed.get_world_size() % num_files
            index = (f_start_id * torch.distributed.get_world_size() + torch.distributed.get_rank() + remainder * f_start_id) % num_files
        else:
            index = (f_start_id * torch.distributed.get_world_size() + torch.distributed.get_rank()) % num_files
        if not args.use_split_data:
            data_file = files[index]
        else:
            data_file = [files[index] for files in file_lists]

        previous_file = data_file

        train_data = pretraining_dataset(data_file, args.max_predictions_per_seq)
        if not args.use_split_data:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = utils.SplitRandomSampler(data_file, batch_ratio=list(map(int, args.split_batch_cnt)))
        train_dataloader = DataLoader(train_data, sampler=train_sampler,batch_size=args.train_batch_size, num_workers=4, worker_init_fn=worker_init,pin_memory=True)

        divisor = args.gradient_accumulation_steps
        overflow_buf = None
        handle = None
        if args.allreduce_post_accumulation:
            overflow_buf = torch.cuda.IntTensor([0])

        for f_id in range(f_start_id + 1, len(files)):
            if torch.distributed.is_initialized() and torch.distributed.get_world_size() > num_files:
                remainder = torch.distributed.get_world_size() % num_files
                index = (f_start_id * torch.distributed.get_world_size() + torch.distributed.get_rank() + remainder * f_start_id) % num_files
            else:
                index = (f_start_id * torch.distributed.get_world_size() + torch.distributed.get_rank()) % num_files
            if not args.use_split_data:
                data_file = files[index]
            else:
                data_file = [files[index] for files in file_lists]

            previous_file = data_file
            dataset_future = pool.submit(create_pretraining_dataset, data_file, args.max_predictions_per_seq, shared_file_list, args, worker_init_fn=worker_init)
            with torch.profiler.profile(
                    schedule=torch.profiler.schedule(wait=40, warmup=10, active=50),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./results/{args.n_gpu}GPU/LBS{args.train_batch_size}', use_gzip=False),
                    with_flops=True, record_shapes=False
            ) if args.profile and utils.is_main_process() else suppress() as p:
                for step, batch in enumerate(train_dataloader):
                    if args.profile and handle is not None:
                        torch.ops.profiler._record_function_exit(handle)
                    training_steps += 1
                    update_step = training_steps % args.gradient_accumulation_steps == 0
                    if args.distributed_lamb:
                        optimizer.set_is_accumulation_step(not update_step)
                        optimizer.set_last_step(step == len(train_dataloader) - 1)

                    with prof.record_function("ExchangePadding"):
                        batch = [t.to(device, non_blocking=True) for t in batch]
                        input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch
                        if args.exchange_padding or args.group_exchange_padding:
                            input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = exchange_padding_fast(args, device, input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels, args.train_batch_size, group_pg=cur_node_pg)

                    with amp_autocast(), prof.record_function("Forward"):
                        loss, mlm_acc, _ = model(input_ids=input_ids, token_type_ids=segment_ids,attention_mask=input_mask, masked_lm_labels=masked_lm_labels,next_sentence_label=next_sentence_labels)
                    if hasattr(optimizer, '_lazy_init_stage1'):
                        optimizer._lazy_init_stage1()

                    with model.no_sync() if not update_step else suppress(), prof.record_function("Backward"):
                        if loss_scaler:
                            if args.use_apex_amp:
                                with amp.scale_loss(loss, optimizer,
                                                    delay_unscale=args.distributed_lamb or (not update_step and not args.baseline),
                                                    delay_overflow_check=args.baseline) as scaled_loss:
                                    if args.baseline or not args.local_gradient_clip:
                                        loss_scaler.scale_override = loss_scaler._loss_scale * args.gradient_accumulation_steps
                                    scaled_loss.backward(create_graph=second_order)
                            else:
                                loss_scaler.scale(loss).backward(create_graph=second_order)
                        else:
                            loss.backward(create_graph=second_order)

                    if hasattr(optimizer, '_lazy_init_stage2'):
                        optimizer._lazy_init_stage2()

                    if args.log_freq > 0:
                        average_loss += loss.item()

                    if args.profile and utils.is_main_process():
                        p.step()

                    if update_step:
                        lr_scheduler.step()
                        if args.distributed_lamb:
                            optimizer.complete_reductions()
                        if args.use_torch_amp:
                            loss_scaler.step(optimizer)
                            loss_scaler.update()
                            if args.distributed_lamb:
                                optimizer.set_global_scale(loss_scaler._get_scale_async())
                            found_inf = optimizer._overflow_buf
                        else:
                            if args.baseline:
                                take_optimizer_step(args, optimizer, model, overflow_buf, global_step)  # Only Apex AMP
                            else:
                                optimizer.step()
                            if args.distributed_lamb:
                                loss_scaler._overflow_buf = optimizer._overflow_buf  # Check overflow after all-reduce
                                loss_scaler.update_scale()  # This will make synchronous scale between all GPUs
                            if args.distributed_lamb:
                                scale = torch.full((1,), loss_scaler.loss_scale(), dtype=torch.float32, device=device)
                                optimizer.set_global_scale(scale)
                            found_inf = loss_scaler._overflow_buf

                        global_step += 1
                        skipped_steps += found_inf
                        optimizer.zero_grad()  # Instead use forward hook p.grad

                        samples_trained = global_step * args.train_batch_size * args.gradient_accumulation_steps * args.n_gpu

                        if (args.eval_dir and args.eval_iter_samples > 0 and samples_trained >= args.eval_iter_start_samples + eval_count * args.eval_iter_samples):

                            skip_interval = skipped_steps.item()
                            if skip_interval > 0:
                                global_step -= skip_interval
                                now_skipped += skip_interval
                                skipped_steps.zero_()
                            else:
                                # on first eval, get eval_dataloader
                                if eval_count == 0:
                                    eval_dataloader = eval_dataset_future.result(timeout=None)

                                samples_trained_prev = samples_trained
                                eval_avg_loss, eval_avg_mlm_accuracy = run_eval(model, eval_dataloader, device,args.num_eval_examples,first_eval=(eval_count == 0),use_cache=args.cache_eval_data,amp_autocast=amp_autocast)
                                if utils.is_main_process():
                                    mlperf_logger.log_event(key=mlperf_logger.constants.EVAL_ACCURACY,value=eval_avg_mlm_accuracy, metadata={'epoch_num': epoch},sync=False)
                                    print({"global_steps": global_step, "eval_loss": eval_avg_loss,"eval_mlm_accuracy": eval_avg_mlm_accuracy})

                                if args.target_mlm_accuracy:
                                    if eval_avg_mlm_accuracy >= args.target_mlm_accuracy:
                                        end_training, converged = True, True
                                        if utils.is_main_process():
                                            print("%f > %f, Target MLM Accuracy reached at %d" % (
                                                eval_avg_mlm_accuracy, args.target_mlm_accuracy, global_step))

                                eval_count += 1

                    if args.target_mlm_accuracy and args.train_mlm_accuracy_window_size > 0:
                        accuracy_scores.append(mlm_acc)
                        if update_step:
                            accuracy_scores = accuracy_scores[-args.train_mlm_accuracy_window_size * args.gradient_accumulation_steps:]
                            avg_mlm_accuracy[0] = sum(accuracy_scores) / len(accuracy_scores)
                            torch.distributed.all_reduce(avg_mlm_accuracy, op=torch.distributed.ReduceOp.SUM)
                            avg_mlm_accuracy /= torch.distributed.get_world_size()

                    if args.log_freq > 0 and training_steps % (args.log_freq * args.gradient_accumulation_steps) == 0:
                        samples_trained = global_step * args.train_batch_size * args.gradient_accumulation_steps * args.n_gpu
                        if utils.is_main_process():
                            time_interval = time.time() - now_time
                            step_interval = global_step - now_step
                            now_time = time.time()
                            now_step = global_step
                            training_perf = args.train_batch_size * args.gradient_accumulation_steps * args.n_gpu * (step_interval + skip_interval) / time_interval
                            skip_interval=0

                            if args.train_mlm_accuracy_window_size > 0:
                                print({"training_steps": training_steps,
                                       "average_loss": average_loss / (args.log_freq * divisor),
                                       "step_loss": loss.item() * args.gradient_accumulation_steps / divisor,
                                       "learning_rate": optimizer.param_groups[0]['lr'],
                                       "seq/s": training_perf,
                                       "global_steps": now_step,
                                       "samples_trained": samples_trained,
                                       "skipped_steps": now_skipped,
                                       "timestamp": now_time,
                                       "mlm_accuracy": avg_mlm_accuracy[0].item()})
                            else:
                                print({"training_steps": training_steps,
                                       "average_loss": average_loss / (args.log_freq * divisor),
                                       "step_loss": loss.item() * args.gradient_accumulation_steps / divisor,
                                       "learning_rate": optimizer.param_groups[0]['lr'],
                                       "seq/s": training_perf,
                                       "global_steps": now_step,
                                       "samples_trained": samples_trained,
                                       "skipped_steps": now_skipped,
                                       "timestamp": now_time})

                        average_loss = 0

                    if global_step >= args.max_steps or end_training:
                        status = 'success' if converged else 'aborted'
                        end_training = True
                        train_time_raw = time.time() - raw_train_start
                        average_loss = torch.tensor(average_loss, dtype=torch.float32).cuda()
                        if args.log_freq > 0:
                            last_num_steps = int(training_steps / args.gradient_accumulation_steps) % args.log_freq
                            last_num_steps = args.log_freq if last_num_steps == 0 else last_num_steps
                            average_loss = average_loss / (last_num_steps * divisor)
                        if (torch.distributed.is_initialized()):
                            average_loss /= torch.distributed.get_world_size()
                            torch.distributed.all_reduce(average_loss)
                        final_loss = average_loss.item()
                        if utils.is_main_process():
                            if args.train_mlm_accuracy_window_size > 0:
                                print((epoch, training_steps / args.gradient_accumulation_steps,),
                                      {"final_loss": final_loss,
                                       "final_mlm_accuracy": avg_mlm_accuracy[0].item()})
                            else:
                                print((epoch, training_steps / args.gradient_accumulation_steps,),
                                      {"final_loss": final_loss})

                    if end_training or (
                            samples_trained - samples_trained_prev >= args.num_samples_per_checkpoint and samples_trained >= args.min_samples_to_start_checkpoints):
                        samples_trained_prev = samples_trained
                        if utils.is_main_process() and not args.skip_checkpoint:
                            # Save a trained model
                            model_to_save = model.module if hasattr(model,
                                                                    'module') else model  # Only save the model it-self
                            if args.phase2:
                                output_save_file = os.path.join(args.output_dir,
                                                                "phase2_ckpt_{}.pt".format(samples_trained))
                            else:
                                output_save_file = os.path.join(args.output_dir,
                                                                "phase1_ckpt_{}.pt".format(samples_trained))

                            torch.save({'model': model_to_save.state_dict(),
                                        'optimizer': optimizer.state_dict(),
                                        'master params': list(amp.master_params(optimizer)),
                                        'files': [f_id] + files}, output_save_file)

                            most_recent_ckpts_paths.append(output_save_file)
                            if len(most_recent_ckpts_paths) > args.keep_n_most_recent_checkpoints:
                                ckpt_to_be_removed = most_recent_ckpts_paths.pop(0)
                                os.remove(ckpt_to_be_removed)

                        if samples_trained >= args.max_samples_termination or end_training:
                            status = 'success' if converged else 'aborted'
                            end_training = True
                            break

                    if args.profile:
                        handle = torch.ops.profiler._record_function_enter("Data Load")

            del train_dataloader
            if samples_trained >= args.max_samples_termination or end_training:
                status = 'success' if converged else 'aborted'
                end_training = True
                break

            train_dataloader, data_file = dataset_future.result(timeout=None)

        mlperf_logger.log_end(key=mlperf_logger.constants.BLOCK_STOP,
                              metadata={'first_epoch_num': epoch},
                              sync=False)
        mlperf_logger.log_end(key=mlperf_logger.constants.EPOCH_STOP,
                              metadata={'epoch_num': epoch}, sync=False)
        epoch += 1

    mlperf_logger.log_event(key=mlperf_logger.constants.TRAIN_SAMPLES,
                            value=samples_trained,
                            sync=False)
    mlperf_logger.log_event(key=mlperf_logger.constants.EVAL_SAMPLES,
                            value=args.num_eval_examples,
                            sync=False)
    mlperf_logger.log_end(key=mlperf_logger.constants.RUN_STOP,
                          metadata={'status': status}, sync=False)

    if args.group_exchange_padding:
        for pg in process_groups:
            torch.distributed.destroy_process_group(pg)

    return args, final_loss, train_time_raw


def global_batch_size(args):
    return args.train_batch_size * args.gradient_accumulation_steps * args.n_gpu


if __name__ == "__main__":

    grad_max = 0.0
    now = time.time()
    args, final_loss, train_time_raw = main()

    gpu_count = args.n_gpu
    if torch.distributed.is_initialized():
        gpu_count = torch.distributed.get_world_size()
    if utils.is_main_process():
        e2e_time = time.time() - now
        training_perf = global_batch_size(args) * (args.max_steps - args.resume_step + skipped_steps) / train_time_raw
        if args.do_train:
            print({"e2e_time": e2e_time, "training_sequences_per_second": training_perf,
                   "final_loss": final_loss, "raw_train_time": train_time_raw})
        else:
            print({"e2e_time": e2e_time})
