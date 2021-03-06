#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
"""Contains common utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import distutils.util
import numpy as np
import six
import os
import paddle
import paddle.fluid as fluid


def print_arguments(args):
    """Print argparse's arguments.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        parser.add_argument("name", default="Jonh", type=str, help="User name.")
        args = parser.parse_args()
        print_arguments(args)

    :param args: Input argparse.Namespace for printing.
    :type args: argparse.Namespace
    """
    print("-------------  Configuration Arguments -------------")
    for arg, value in sorted(six.iteritems(vars(args))):
        print("%25s : %s" % (arg, value))
    print("----------------------------------------------------")


def add_arguments(argname, type, default, help, argparser, **kwargs):
    """Add argparse's argument.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        add_argument("name", str, "Jonh", "User name.", parser)
        args = parser.parse_args()
    """
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs)


def check_gpu(use_gpu):
    """
    Log error and exit when set use_gpu=true in paddlepaddle
    cpu version.
    """
    err = "Config use_gpu cannot be set as true while you are " \
          "using paddlepaddle cpu version ! \nPlease try: \n" \
          "\t1. Install paddlepaddle-gpu to run model on GPU \n" \
          "\t2. Set use_gpu as false in config file to run " \
          "model on CPU"

    try:
        if use_gpu and not fluid.is_compiled_with_cuda():
            logger.error(err)
            sys.exit(1)
    except Exception as e:
        pass


def get_median(data):
    data = sorted(data)
    size = len(data)
    if size == 2:
        median = data[size - 1]
    elif size % 2 == 0:
        median = (data[size // 2] + data[size // 2 - 1]) / 2
    elif size % 2 == 1:
        median = data[(size - 1) // 2]
    return median


def create_input_layer(is_train, args, data_layout='NCHW'):
    """create data_loader
    Usage:
        Using mixup process in training, it will return 5 results, include data_loader, image, y_a(label), y_b(label) and lamda, or it will return 3 results, include data_loader, image, and label.
    Args:
        is_train: mode
        args: arguments
    Returns:
        data_loader and the input data of net,
    """
    image_shape = [int(m) for m in args.image_shape.split(",")]
    if data_layout == 'NHWC':
        image_shape = [image_shape[1], image_shape[2], image_shape[0]]
    if is_train:
        image_shape = [args.batch_size_train] + image_shape
        label_shape = [args.batch_size_train, 1]
    else:
        image_shape = [None] + image_shape
        label_shape = [None, 1]

    feed_image = fluid.data(
        name="feed_image" if is_train else "feed_image_test",
        shape=image_shape,
        dtype="float32",
        lod_level=0)

    feed_label = fluid.data(
        name="feed_label" if is_train else "feed_label_test",
        shape=label_shape,
        dtype="int64",
        lod_level=0)

    feed_image.persistable = True
    feed_label.persistable = True
    return feed_image, feed_label


def get_num_trainers():
    num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))
    return num_trainers


def get_trainer_id():
    trainer_id = int(os.environ.get('PADDLE_TRAINER_ID', 0))
    return trainer_id
