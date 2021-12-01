# Copyright (C) 2021 Samsung Electronics Co. LTD

# This software is a property of Samsung Electronics.
# No part of this software, either material or conceptual may be copied or distributed, transmitted,
# transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
# electronic, mechanical, manual or otherwise, or disclosed
# to third parties without the express written permission of Samsung Electronics.

# The following items are modified and they can be claimed as properties of Samsung Electronics.
# (1) Add get_optimzer() interface
# (2) Add a batch sampler (SplitRandomSampler) for 4-bin splitting training data 


# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
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
import torch.distributed as dist
import os
import time
import math
import h5py
import torch.cuda.nvtx as nvtx
from apex.optimizers import FusedLAMB, FusedAdam, FusedSGD
from optim import ACClip, MADGRAD
from apex.contrib.optimizers.distributed_fused_lamb import DistributedFusedLAMB
from apex.contrib.optimizers.distributed_fused_adam_v3 import DistributedFusedAdamV3
from typing import Callable, Optional, Tuple

try:
    from torch.utils.tensorboard import SummaryWriter

    has_tensorboard = True
except ImportError:
    has_tensorboard = False

try:
    from apex import amp

    has_apex = True
except ImportError:
    amp = None
    has_apex = False

from contextlib import contextmanager
from functools import partial
import logging.config
import random

_TENSOR_PARALLEL_GROUP = None  # ProcessGroup
_DATA_PARALLEL_GROUP = None  # ProcessGroup
_PIPELINE_PARALLEL_GROUP = None  # ProcessGroup
_TENSOR_PARALLEL_WORLD_SIZE = None
_PIPELINE_PARALLEL_WORLD_SIZE = None
_DATA_PARALLEL_RANKS = None  # rank list
_TENSOR_PARALLEL_RANKS = None  # rank list
_PIPELINE_PARALLEL_RANKS = None  # rank list
_DATA_PARALLEL_RANK = None  # rank of this process(GPU)
_TENSOR_PARALLEL_RANK = None  # rank of this process(GPU)
_PIPELINE_PARALLEL_RANK = None  # rank of this process(GPU)


def generate_seeds(rng, size):
    """
    Generate list of random seeds

    :param rng: random number generator
    :param size: length of the returned list
    """
    seeds = [rng.randint(0, 2 ** 32 - 1) for _ in range(size)]
    return seeds


def broadcast_seeds(seeds, device):
    """
    Broadcasts random seeds to all distributed workers.
    Returns list of random seeds (broadcasted from workers with rank 0).

    :param seeds: list of seeds (integers)
    :param device: torch.device
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        seeds_tensor = torch.LongTensor(seeds).to(device)
        torch.distributed.broadcast(seeds_tensor, 0)
        seeds = seeds_tensor.tolist()
    return seeds


def setup_seeds(master_seed, epochs, device):
    """
    Generates seeds from one master_seed.
    Function returns (worker_seeds, shuffling_seeds), worker_seeds are later
    used to initialize per-worker random number generators (mostly for
    dropouts), shuffling_seeds are for RNGs resposible for reshuffling the
    dataset before each epoch.
    Seeds are generated on worker with rank 0 and broadcasted to all other
    workers.

    :param master_seed: master RNG seed used to initialize other generators
    :param epochs: number of epochs
    :param device: torch.device (used for distributed.broadcast)
    """
    if master_seed is None:
        # random master seed, random.SystemRandom() uses /dev/urandom on Unix
        master_seed = random.SystemRandom().randint(0, 2 ** 32 - 1)
        if get_rank() == 0:
            # master seed is reported only from rank=0 worker, it's to avoid
            # confusion, seeds from rank=0 are later broadcasted to other
            # workers
            logging.info(f'Using random master seed: {master_seed}')
    else:
        # master seed was specified from command line
        logging.info(f'Using master seed from command line: {master_seed}')

    # initialize seeding RNG
    seeding_rng = random.Random(master_seed)

    # generate worker seeds, one seed for every distributed worker
    worker_seeds = generate_seeds(seeding_rng, get_world_size())

    # generate seeds for data shuffling, one seed for every epoch
    shuffling_seeds = generate_seeds(seeding_rng, epochs)

    # broadcast seeds from rank=0 to other workers
    worker_seeds = broadcast_seeds(worker_seeds, device)
    shuffling_seeds = broadcast_seeds(shuffling_seeds, device)
    return worker_seeds, shuffling_seeds


def barrier():
    """
    Works as a temporary distributed barrier, currently pytorch
    doesn't implement barrier for NCCL backend.
    Calls all_reduce on dummy tensor and synchronizes with GPU.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(torch.cuda.FloatTensor(1))
        torch.cuda.synchronize()


def get_rank():
    """
    Gets distributed rank or returns zero if distributed is not initialized.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    return rank


def get_world_size():
    """
    Gets total number of distributed workers or returns one if distributed is
    not initialized.
    """

    if torch.distributed.is_available():
        print("Torch distributed is available.")
    else:
        print("Torch distributed is not available.")

    if torch.distributed.is_initialized():
        print("Torch distributed is initialized.")
    else:
        print("Torch distributed is not initialized.")

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1
    return world_size


def set_device(cuda, local_rank):
    """
    Sets device based on local_rank and returns instance of torch.device.

    :param cuda: if True: use cuda
    :param local_rank: local rank of the worker
    """
    if cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


@contextmanager
def sync_workers():
    """
    Yields distributed rank and synchronizes all workers on exit.
    """
    rank = get_rank()
    yield rank
    barrier()


def is_main_process():
    return get_rank() == 0


def format_step(step):
    if isinstance(step, str):
        return step
    s = ""
    if len(step) > 0:
        s += "Training Epoch: {} ".format(step[0])
    if len(step) > 1:
        s += "Training Iteration: {} ".format(step[1])
    if len(step) > 2:
        s += "Validation Iteration: {} ".format(step[2])
    return s


def has_hooks(module: torch.nn.Module):
    """ Returns True if the module uses hooks. """

    for hooks in (module._forward_pre_hooks,  # pylint: disable=protected-access
                  module._forward_hooks, module._backward_hooks):  # pylint: disable=protected-access
        if hooks:
            return True
    return False


def get_optimizer(name, parameters, lr=0.1, betas=(0.9, 0.999), wd=1e-4, eps=1e-6):
    if name.lower() == "fusedlamb":
        optimizer = FusedLAMB(parameters, lr=lr, betas=betas, eps=eps, weight_decay=wd, max_grad_norm=1.0)
    elif name.lower() == 'fusedsgd':
        optimizer = FusedSGD(parameters, lr=lr, momentum=betas[0], weight_decay=wd)
    elif name.lower() == 'fusedadam':
        optimizer = FusedAdam(parameters, lr=lr, betas=betas, eps=eps, weight_decay=wd)
    elif name.lower() == "fusedacclip":
        optimizer = ACClip(parameters, lr=lr, betas=betas, eps=eps, weight_decay=wd, max_grad_norm=1.0)
    elif name.lower() == 'madgrad':
        optimizer = MADGRAD(parameters, lr=lr, momentum=betas[0], eps=eps, weight_decay=wd, max_grad_norm=1.0)
    elif name.lower() == 'distributedfusedlamb':
        optimizer = DistributedFusedLAMB(parameters, lr=lr, betas=betas, eps=eps, weight_decay=wd, max_grad_norm=1.0,
                                         overlap_reductions=True, clip_after_ar=True,
                                         dwu_num_blocks=4, dwu_num_chunks=1, dwu_num_rs_pg=1,
                                         dwu_num_ar_pg=1, dwu_num_ag_pg=1, use_nvlamb=False)
    else:
        print("Error : Not Supported Type of Optimizer {}".format(name))
        exit()
    return optimizer


def has_children(module):
    try:
        next(module.children())
        return True
    except StopIteration:
        return False


def hasNaN(parameters):
    for p in parameters:
        if torch.any(p.grad.isnan()):
            return True
    return False


def ema(avg, beta, yi, i):
    """Exponential moving average"""
    if avg is None: avg = 0
    avg = beta * avg + (1 - beta) * yi
    return avg, avg / (1 - beta ** (i + 1))


class GradientNoiseScale:
    """
    A class to measure the gradient noise scale of a model while training (cf. https://arxiv.org/abs/1812.06162).

    The core thesis of the paper is that, if our batch size is small, there will be a lot of noise present in the gradients, and we might update our weights only on noise.
    After several updates the optimizer may still push us in the right direction, but we would be better off having used a larger batch size, which is more computationally
    efficient and directly averages out the noise in the gradients.

    But there's a limit to the gains large batch sizes can give you - if, after a certain batch size, your gradient is already accurate, there's no point in increasing the
    batch size further, as we'll just be wasting compute for little to no gain in accuracy.

    This means there is some theoretically optimal batch size for a given model, which measuring the gradient noise scale can help us to estimate.

    To estimate the 'simple' noise scale (Bsimple), we need to have a measure of the gradients using a large batch size (Bbig) and a small
    batch size (Bsmall).

    when we have those:
        Bsimple ≈ (tr(Σ) / |G|^2)

    tr(Σ) can be approximated by:
        tr(Σ) ≈ (1 / ((1/Bsmall) - (1/Bbig))) * (|Gsmall|^2 - |Gbig|^2)

    and |G|^2 by:
        |G|^2 ≈ (1 / (Bbig - Bsmall)) * (Bbig*|Gbig|^2 - Bsmall*|Gsmall|^2)

    - With multi-gpu training, we can do this by taking the gradients of the microbatch_size_per_gpu for Bsmall,
    and the gradients of the entire batch for Bbig.
    - Alternatively, we can just take Bsmall as a single batch, and Bbig as several sequential batches in a row.
    This is the option we've opted for in this implementation because a) it's easier to implement and b) also works in
    single-gpu environments. Unfortunately it does come with some memory overhead.
    """

    def __init__(self, batch_size_small, n_batches=20, beta=0.99, args=None):
        self.batch_size_small = batch_size_small
        self.batch_size_large = batch_size_small * n_batches
        self.n_batches = n_batches
        self.beta = beta
        self.buffer = None
        self.ema_scale = None
        self.ema_noise = None
        self.noise_scale = None
        self.n_updates = 0
        self.args = args

    def _update(self, master_grads):
        grad = torch._utils._flatten_dense_tensors(master_grads)
        is_overflow = grad is None
        if is_overflow:
            return
        if self.buffer is None:
            self.buffer = grad
        else:
            self.buffer += grad
        if self.n_updates % self.n_batches == self.n_batches - 1:
            # average grads every n_batches iteration to get a simulation of Bbig
            self.buffer /= self.n_batches
            grads = self.buffer
            self.buffer = None

            # calculate Gbig and Gsmall
            # this needs to be done in fp32 or it overflows
            g_big = torch.square(torch.norm(grads))
            g_small = torch.square(torch.norm(grad))

            # communicate any overflows
            is_overflow = (g_small.isinf().any() or g_small.isnan().any() or g_big.isinf().any() or g_big.isnan().any())
            if is_overflow:
                return

            # calculate noise / scale
            noise = 1 / (self.batch_size_large - self.batch_size_small) * (
                    self.batch_size_large * g_big - self.batch_size_small * g_small)
            scale = 1 / (1 / self.batch_size_small - 1 / self.batch_size_large) * (g_small - g_big)

            # calculate running average
            self.ema_noise, noise = ema(self.ema_noise, self.beta, noise, self.n_updates)
            self.ema_scale, scale = ema(self.ema_scale, self.beta, scale, self.n_updates)

            # calculate noise scale
            self.noise_scale = (scale / noise)

        self.n_updates += 1

    def update(self, master_grads):
        self._update(master_grads)


def get_noise_scale_logger(args):
    noise_scale_logger = GradientNoiseScale(batch_size_small=args.train_batch_size, n_batches=10)
    return noise_scale_logger


# On APOLLO, Adaptive Parameter-wise diagonal quasi-newton method for nonconvex
# Parameter-wise clipping slightly better
# Momentum Extension of clipped SGD : Stability and convergence of Stochastic Gradient Clipping
# Not clip gradients, clip momentum is the key

def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if _TENSOR_PARALLEL_GROUP is None or \
            _PIPELINE_PARALLEL_GROUP is None or \
            _DATA_PARALLEL_GROUP is None:
        return False
    return True


def get_tensor_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""
    assert _TENSOR_PARALLEL_GROUP is not None, \
        'intra_layer_model parallel group is not initialized'
    return _TENSOR_PARALLEL_GROUP


def get_pipeline_parallel_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    assert _PIPELINE_PARALLEL_GROUP is not None, \
        'pipeline_model parallel group is not initialized'
    return _PIPELINE_PARALLEL_GROUP


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, \
        'data parallel group is not initialized'
    return _DATA_PARALLEL_GROUP


def set_data_parallel_group(group):
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = group


def set_data_parallel_ranks(ranks):
    global _DATA_PARALLEL_RANKS
    _DATA_PARALLEL_RANKS = ranks


def set_tensor_parallel_group(group):
    global _TENSOR_PARALLEL_GROUP
    _TENSOR_PARALLEL_GROUP = group


def set_tensor_parallel_ranks(ranks):
    global _TENSOR_PARALLEL_RANKS
    _TENSOR_PARALLEL_RANKS = ranks


def set_pipeline_parallel_group(group):
    global _PIPELINE_PARALLEL_GROUP
    _PIPELINE_PARALLEL_GROUP = group


def set_pipeline_parallel_ranks(ranks):
    global _PIPELINE_PARALLEL_RANKS
    _PIPELINE_PARALLEL_RANKS = ranks


def get_tensor_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    global _TENSOR_PARALLEL_WORLD_SIZE
    if _TENSOR_PARALLEL_WORLD_SIZE is not None:
        return _TENSOR_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_tensor_parallel_group())


def get_pipeline_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    global _PIPELINE_PARALLEL_WORLD_SIZE
    if _PIPELINE_PARALLEL_WORLD_SIZE is not None:
        return _PIPELINE_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_pipeline_parallel_group())


def get_tensor_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    global _TENSOR_PARALLEL_RANK
    if _TENSOR_PARALLEL_RANK is not None:
        return _TENSOR_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_parallel_group())


def get_pipeline_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    global _PIPELINE_PARALLEL_RANK
    if _PIPELINE_PARALLEL_RANK is not None:
        return _PIPELINE_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_pipeline_parallel_group())


def get_tensor_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_tensor_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return torch.distributed.get_world_size(group=get_data_parallel_group())


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return torch.distributed.get_rank(group=get_data_parallel_group())


def get_data_parallel_ranks():
    return _DATA_PARALLEL_RANKS


def get_tensor_parallel_ranks():
    return _TENSOR_PARALLEL_RANKS


def get_pipeline_parallel_ranks():
    return _PIPELINE_PARALLEL_RANKS


def destroy_parallel():
    """Set the groups to none."""
    global _TENSOR_PARALLEL_GROUP
    _TENSOR_PARALLEL_GROUP = None
    global _PIPELINE_PARALLEL_GROUP
    _PIPELINE_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None


def split_tensor_along_last_dim(
        tensor: torch.Tensor, num_partitions: int, contiguous_split_chunks: bool = False
) -> Tuple[torch.Tensor, ...]:
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size()[last_dim] // num_partitions
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


def _initialize_affine_weight(
        weight: torch.Tensor,
        out_features: int,
        in_features: int,
        per_partition_size: int,
        partition_dim: int,
        init_method: Callable[[torch.Tensor], torch.Tensor],
        stride: int = 1,
        return_master_weight: bool = False,
) -> Optional[torch.Tensor]:
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    # If we only use 1 process for model parallelism, bypass scatter.
    world_size = get_tensor_parallel_world_size()
    if world_size == 1:
        init_method(weight)
        if return_master_weight:
            return weight
        return None

    # Initialize master weight
    master_weight = torch.empty(out_features, in_features, dtype=weight.dtype, requires_grad=False)
    init_method(master_weight)

    # Split and copy
    per_partition_per_stride_size = per_partition_size // stride
    weight_list = torch.split(master_weight, per_partition_per_stride_size, dim=partition_dim)
    rank = get_tensor_parallel_rank()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None


def is_pipeline_master():
    return get_pipeline_parallel_rank() == 0

class SplitRandomSampler:
    def __init__(self, input_files, batch_ratio=[6,3,2,5], generator=None) -> None:
        self.num_samples = []
        for input_file in input_files:
            h5_ifile = h5py.File(input_file, 'r')
            f_next_sentence_labels = h5_ifile['next_sentence_labels'][:]
            self.num_samples.append(f_next_sentence_labels.shape[0])
            h5_ifile.close()
        self.generator = generator
        self.batch_ratio = batch_ratio

    def __iter__(self):
        if self.generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        else:
            generator = self.generator
        max_len = min([self.num_samples[i] // batch_count for i,batch_count in enumerate(self.batch_ratio)])
        rand_idxs = [(torch.randperm(num_sample, generator=generator)[:int(num_sample/batch_cnt)*batch_cnt]).view(-1, batch_cnt)[:max_len,:]+sum(self.num_samples[:i]) for i, (num_sample,batch_cnt) in enumerate(zip(self.num_samples,self.batch_ratio))]
        rand_idxs = torch.flatten(torch.cat(rand_idxs,dim=-1)) # max_len, 16 ->
        yield from rand_idxs.tolist()
