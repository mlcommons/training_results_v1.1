# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import division
import math
import os

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.paddle import DALIGenericIterator

import paddle
from paddle import fluid


def convert_data_layout(data_layout):
    if data_layout == 'NCHW':
        return types.NCHW
    elif data_layout == 'NHWC':
        return types.NHWC
    else:
        assert False, "not supported data_layout:{}".format(data_layout)


class HybridTrainPipe(Pipeline):
    """
    Create train pipe line.  You can find document here: 
    https://docs.nvidia.com/deeplearning/sdk/dali-master-branch-user-guide/docs/plugins/paddle_tutorials.html
    Note: You may need to find the newest DALI version.
    """

    def __init__(self,
                 rec_path,
                 idx_path,
                 batch_size,
                 crop,
                 min_area,
                 lower,
                 upper,
                 mean,
                 std,
                 device_id,
                 shard_id=0,
                 num_shards=1,
                 random_shuffle=True,
                 num_threads=4,
                 seed=42,
                 data_layout="NCHW",
                 output_dtype='float16',
                 pad_output=False,
                 prefetch_queue_depth=5,
                 nvjpeg_padding=0,
                 decoder_buffer_hint=0,
                 normalize_buffer_hint=0):
        super(HybridTrainPipe, self).__init__(
            batch_size, num_threads, device_id, seed=seed)

        self.input = ops.MXNetReader(
            path=[rec_path],
            index_path=[idx_path],
            lazy_init=True,
            dont_use_mmap=0,
            random_shuffle=random_shuffle,
            shard_id=shard_id,
            num_shards=num_shards)

        self.decode = ops.ImageDecoderRandomCrop(
            device="mixed",
            output_type=types.RGB,
            device_memory_padding=nvjpeg_padding,
            host_memory_padding=nvjpeg_padding,
            random_area=[min_area, 1.0],
            random_aspect_ratio=[lower, upper],
            bytes_per_sample_hint=decoder_buffer_hint,
            affine=False)
        self.rrc = ops.Resize(
            device="gpu",
            resize_x=crop,
            resize_y=crop,
            bytes_per_sample_hint=decoder_buffer_hint)
        dtype = types.FLOAT if output_dtype == 'float32' else types.FLOAT16
        self.cmnp = ops.CropMirrorNormalize(
            device="gpu",
            output_dtype=dtype,
            output_layout=convert_data_layout(data_layout),
            crop=(crop, crop),
            image_type=types.RGB,
            mean=mean,
            std=std,
            pad_output=pad_output,
            bytes_per_sample_hint=normalize_buffer_hint)
        self.coin = ops.CoinFlip(probability=0.5)
        self.to_int64 = ops.Cast(dtype=types.INT64, device="gpu")

    def define_graph(self):
        rng = self.coin()
        jpegs, labels = self.input(name="Reader")
        images = self.decode(jpegs)
        images = self.rrc(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.to_int64(labels.gpu())]

    def __len__(self):
        return self.epoch_size("Reader")


class HybridValPipe(Pipeline):
    """
    Create validate pipe line.
    """

    def __init__(self,
                 rec_path,
                 idx_path,
                 batch_size,
                 resize_shorter,
                 crop,
                 mean,
                 std,
                 device_id,
                 shard_id=0,
                 num_shards=1,
                 random_shuffle=False,
                 num_threads=3,
                 seed=42,
                 data_layout='NCHW',
                 output_dtype='float32',
                 pad_output=False,
                 prefetch_queue_depth=5,
                 nvjpeg_padding=0):
        super(HybridValPipe, self).__init__(
            batch_size,
            num_threads=num_threads,
            device_id=device_id,
            seed=seed,
            prefetch_queue_depth=5)
        self.input = ops.MXNetReader(
            path=[rec_path],
            index_path=[idx_path],
            lazy_init=True,
            dont_use_mmap=0,
            random_shuffle=random_shuffle,
            shard_id=shard_id,
            num_shards=num_shards)

        self.decode = ops.ImageDecoder(
            device="mixed",
            output_type=types.RGB,
            device_memory_padding=nvjpeg_padding,
            host_memory_padding=nvjpeg_padding,
            affine=False)

        self.resize = ops.Resize(device="gpu", resize_shorter=resize_shorter)

        dtype = types.FLOAT if output_dtype == 'float32' else types.FLOAT16
        self.cmnp = ops.CropMirrorNormalize(
            device="gpu",
            output_dtype=dtype,
            output_layout=convert_data_layout(data_layout),
            crop=(crop, crop),
            image_type=types.RGB,
            mean=mean,
            std=std,
            pad_output=pad_output)

        self.to_int64 = ops.Cast(dtype=types.INT64, device="gpu")

    def define_graph(self):
        jpegs, labels = self.input(name="Reader")
        images = self.decode(jpegs)
        images = self.resize(images)
        output = self.cmnp(images)
        return [output, self.to_int64(labels.gpu())]

    def __len__(self):
        return self.epoch_size("Reader")


def build(args,
          trainer_id,
          num_trainers,
          mode='train',
          device_id=0,
          output_dtype='float16'):
    assert mode in ['train', 'val']
    assert trainer_id is not None
    assert num_trainers is not None

    file_root = args.data_dir
    batch_size_train = args.batch_size_train
    batch_size_test = args.batch_size_test

    mean = [v * 255 for v in args.image_mean]
    std = [v * 255 for v in args.image_std]
    image_shape = [int(m) for m in args.image_shape.split(",")]
    crop = image_shape[1]
    pad_output = True if image_shape[0] == 4 else False
    min_area = args.lower_scale
    lower = args.lower_ratio
    upper = args.upper_ratio
    seed = args.random_seed

    if mode == 'val':
        rec_path = os.path.join(file_root, 'val.rec')
        idx_path = os.path.join(file_root, 'val.idx')

        pipe = HybridValPipe(
            rec_path=rec_path,
            idx_path=idx_path,
            batch_size=batch_size_test,
            resize_shorter=args.resize_short_size,
            crop=crop,
            mean=mean,
            std=std,
            device_id=device_id,
            shard_id=trainer_id,
            num_shards=num_trainers,
            seed=seed + trainer_id,
            data_layout=args.data_format,
            num_threads=args.dali_num_threads,
            output_dtype=output_dtype,
            pad_output=pad_output,
            nvjpeg_padding=args.dali_nvjpeg_memory_padding)
        pipe.build()
        real_size = int(
            math.floor(1.0 * (trainer_id + 1) * len(pipe) / num_trainers) -
            math.floor(1.0 * trainer_id * len(pipe) / num_trainers))

        return DALIGenericIterator(
            pipe, ['feed_image_test', 'feed_label_test'],
            size=real_size,
            dynamic_shape=True,
            fill_last_batch=True,
            last_batch_padded=True)

    rec_path = os.path.join(file_root, 'train.rec')
    idx_path = os.path.join(file_root, 'train.idx')

    print("dali device_id:", device_id, "shard_id:", trainer_id, "num_shard:",
          num_trainers)
    pipe = HybridTrainPipe(
        rec_path=rec_path,
        idx_path=idx_path,
        batch_size=args.batch_size_train,
        crop=crop,
        min_area=args.lower_scale,
        lower=args.lower_ratio,
        upper=args.upper_ratio,
        mean=mean,
        std=std,
        device_id=device_id,
        shard_id=trainer_id,
        num_shards=num_trainers,
        random_shuffle=False,
        seed=seed + trainer_id,
        data_layout=args.data_format,
        num_threads=args.dali_num_threads,
        output_dtype=output_dtype,
        pad_output=pad_output,
        nvjpeg_padding=args.dali_nvjpeg_memory_padding,
        decoder_buffer_hint=args.dali_decoder_buffer_hint,
        normalize_buffer_hint=args.dali_normalize_buffer_hint)
    pipe.build()
    pipelines = [pipe]
    sample_per_shard = len(pipe) // num_trainers
    return DALIGenericIterator(
        pipelines, ['feed_image', 'feed_label'], size=sample_per_shard)
