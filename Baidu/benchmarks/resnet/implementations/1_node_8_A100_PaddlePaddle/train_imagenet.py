# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle.fluid.core as core
from paddle.device.cuda.graphs import CUDAGraph

import json
import os
import time
import sys
import argparse
import functools
import paddle
import paddle.fluid as fluid
from paddle.fluid.memory_analysis import pre_allocate_memory, get_max_memory_info
import models
from utils.utility import add_arguments, print_arguments
from paddle.distributed import fleet
from paddle.fluid import compiler
import paddle.nn as nn
import numpy as np
import math
from mpi4py import MPI

# test fp16
from paddle.fluid.contrib.mixed_precision.fp16_utils import rewrite_program
from paddle.fluid.contrib.mixed_precision.fp16_utils import cast_model_to_fp16

from mlperf_logging.mllog import constants
import utils.utility as utility
from utils.mlperf_logging_helper import paddle_resnet_print_start, paddle_resnet_print_end, paddle_resnet_print_event
from utils.utility import add_arguments, print_arguments

from optimization import get_optimizer_and_lr, apply_backward_and_optimization_passes, create_strategy
import dali

mpi_comm = MPI.COMM_WORLD
num_trainers = utility.get_num_trainers()
trainer_id = utility.get_trainer_id()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)

    add_arg('batch_size_train', int, 408, "Minibatch size per device.")
    add_arg('batch_size_test', int, 408, "Minibatch size per device.")
    add_arg('total_images', int, 1281167, "Training image number.")
    add_arg('num_epochs', int, 120, "number of epochs.")
    add_arg('class_dim', int, 1000, "Class number.")
    add_arg('image_shape', str, "3,224,224", "input image size")
    add_arg('model', str, "ResNet50", "Set the network to use.")
    add_arg('data_dir', str, "./data/ILSVRC2012/",
            "The ImageNet dataset root dir.")
    add_arg('fp16', bool, False, "Enable half precision training with fp16.")
    add_arg('pure_fp16', bool, False, "Enable training with pure fp16")
    add_arg('data_format', str, "NCHW", "Tensor data format when training.")
    add_arg('momentum_rate', float, 0.9, "momentum_rate.")
    add_arg('use_label_smoothing', bool, False,
            "Whether to use label_smoothing or not")
    add_arg('label_smoothing_epsilon', float, 0.2,
            "Set the label_smoothing_epsilon parameter")
    add_arg('lower_scale', float, 0.05, "Set the lower_scale in ramdom_crop")
    add_arg('lower_ratio', float, 3. / 4., "Set the lower_ratio in ramdom_crop")
    add_arg('upper_ratio', float, 4. / 3., "Set the upper_ratio in ramdom_crop")
    add_arg('resize_short_size', int, 256, "Set the resize_short_size")
    add_arg('fetch_steps', int, 20,
            "The step interval to print the training results.")
    add_arg('do_test', bool, False, "Whether do test every epoch.")
    add_arg('num_iteration_per_drop_scope', int, 100,
            "Ihe iteration intervals to clean up temporary variables.")
    add_arg(
        'image_mean',
        nargs='+',
        type=float,
        default=[0.485, 0.456, 0.406],
        help="The mean of input image data")
    add_arg(
        'image_std',
        nargs='+',
        type=float,
        default=[0.229, 0.224, 0.225],
        help="The std of input image data")
    add_arg('lr', float, 0.1, "set learning rate.")
    add_arg('warmup_epochs', float, 5.0, "the number of epoch for lr warmup")
    add_arg('label_smooth', float, 0.1, "label smooth epsilon for loss")
    add_arg('pow2_end_epoch', float, 72.0, "the number of epoch for lr warmup")
    add_arg('lars_coeff', float, 0., "lars coefficient for lars optimizer")
    add_arg('lars_weight_decay', float, 0.0005,
            "lars_weight_decay for lars optimizer")
    add_arg('truncnorm_init', bool, False, "truncnorm_init for conv.")
    add_arg('eval_per_train_epoch', int, 4,
            "do evaluation every k training epoch as MLperf required")
    add_arg('mlperf_threshold', float, 0.759,
            "threshold to stop and count the time")
    add_arg('val_size', int, 50000, "validation image number.")
    add_arg('eval_offset', int, 0, "the idx of current run")
    add_arg('mlperf_run', bool, True, "run for mlperf or not")
    add_arg('random_seed', int, -1, "run for mlperf or not")
    add_arg('all_reduce_size', int, 32, "size in MB to do an all reduce")
    add_arg('dali_num_threads', int, 4, "num threads for dali")
    add_arg('dali_decoder_buffer_hint', int, 1315942,
            'dali decoder buffer hint')
    add_arg('dali_normalize_buffer_hint', int, 441549,
            'dali normalize buffer hint')
    add_arg('dali_nvjpeg_memory_padding', int, 256 * 1024 * 1024,
            'dali nvjpeg memory padding')

    return parser.parse_args()


# modification: label smooth
def net_config(image, args, is_train, label=0, data_format="NCHW"):
    class_dim = args.class_dim
    model_name = args.model
    use_label_smoothing = args.use_label_smoothing
    epsilon = args.label_smoothing_epsilon

    if model_name == 'ResNet50_clas':
        image_shape = [int(m) for m in args.image_shape.split(",")]
        model = models.__dict__[model_name](class_dim=args.class_dim,
                                            input_image_channel=image_shape[0],
                                            data_format=data_format)
        out = model(image)
    else:
        model = models.__dict__[model_name]()
        out = model.net(input=image,
                        args=args,
                        class_dim=class_dim,
                        data_format=data_format)
    if is_train:
        if args.label_smooth == 0:
            cost = fluid.layers.softmax_with_cross_entropy(out, label)
            cost = fluid.layers.reduce_sum(cost)
        else:
            label_one_hot = fluid.layers.one_hot(input=label, depth=class_dim)
            smooth_label = fluid.layers.label_smooth(
                label=label_one_hot, epsilon=args.label_smooth, dtype="float32")
            log_softmax = nn.functional.log_softmax(out)
            cost = fluid.layers.reduce_sum(-(smooth_label * log_softmax))
    else:
        cost = fluid.layers.softmax_with_cross_entropy(out, label)
        cost = fluid.layers.reduce_sum(cost)

    if is_train and args.mlperf_run:
        return cost
    acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
    acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)
    return cost, acc_top1, acc_top5


def save_program_to_file(filepath, program):
    with open(filepath, "w") as f:
        f.write(str(program))


def build_program(is_train, main_prog, startup_prog, args, dist_strategy):
    data_layout = args.data_format
    model_name = args.model
    model_list = [m for m in dir(models) if "__" not in m]
    assert model_name in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)
    with fluid.program_guard(main_prog, startup_prog):
        image, label = utility.create_input_layer(
            is_train, args, data_layout=data_layout)

        with fluid.unique_name.guard():
            if is_train and args.mlperf_run:
                avg_cost = net_config(
                    image,
                    args,
                    label=label,
                    is_train=is_train,
                    data_format=data_layout)
                build_program_out = [avg_cost]
            else:
                avg_cost, acc_top1, acc_top5 = net_config(
                    image,
                    args,
                    label=label,
                    is_train=is_train,
                    data_format=data_layout)
                build_program_out = [avg_cost, acc_top1, acc_top5]

            use_py_passes = True
            if is_train:
                optimizer, global_lr = get_optimizer_and_lr(args)
                dist_optimizer = apply_backward_and_optimization_passes(
                    main_prog, startup_prog, optimizer, avg_cost, dist_strategy,
                    use_py_passes)
                if use_py_passes:
                    main_save_file = 'train_program_{}_py_pass.txt'.format(
                        trainer_id)
                    startup_save_file = 'startup_program_{}_py_pass.txt'.format(
                        trainer_id)
                else:
                    main_save_file = 'train_program_{}.txt'.format(trainer_id)
                    startup_save_file = 'startup_program_{}.txt'.format(
                        trainer_id)
                save_program_to_file(main_save_file, main_prog)
                save_program_to_file(startup_save_file, startup_prog)
                build_program_out.append(global_lr)

                return build_program_out, dist_optimizer
            else:
                if args.fp16:
                    if args.pure_fp16:
                        cast_model_to_fp16(main_prog, use_fp16_guard=True)
                    else:
                        rewrite_program(main_prog)
                if use_py_passes:
                    test_save_file = "./test_program_{}_py_pass.txt".format(
                        trainer_id)
                else:
                    test_save_file = "./test_program_{}.txt".format(trainer_id)
                save_program_to_file(test_save_file, main_prog)

    return build_program_out


def get_device_num():
    device_num = fluid.core.get_cuda_device_count()
    return device_num


def check_args(args):
    assert args.fp16 and args.pure_fp16, "only pure fp16 is supported"


def train(args):
    check_args(args)

    if trainer_id == 0:
        paddle_resnet_print_start(key=constants.INIT_START)
        paddle_resnet_print_start(
            key=constants.GLOBAL_BATCH_SIZE,
            val=args.batch_size_train * num_trainers)
        paddle_resnet_print_event(key=constants.OPT_NAME, val='lars')
        paddle_resnet_print_event(key=constants.LARS_EPSILON, val=0.0)
        paddle_resnet_print_event(
            key=constants.LARS_OPT_WEIGHT_DECAY, val=args.lars_weight_decay)
        paddle_resnet_print_event(
            key='lars_opt_momentum', val=args.momentum_rate)
        paddle_resnet_print_event(
            key='lars_opt_base_learning_rate', val=args.lr)
        paddle_resnet_print_event(
            key='lars_opt_learning_rate_warmup_epochs', val=args.warmup_epochs)
        paddle_resnet_print_event(
            key=constants.LARS_OPT_LR_DECAY_STEPS, val=args.pow2_end_epoch)
        paddle_resnet_print_event(
            key=constants.GRADIENT_ACCUMULATION_STEPS, val=1)
        paddle_resnet_print_event(key=constants.SEED, val=args.random_seed)

    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = fluid.CUDAPlace(gpu_id)
    exe_place = core.Place()
    exe_place.set_place(place)

    build_strategy, exec_strategy, dist_strategy = create_strategy(args)
    fleet.init(is_collective=True)

    # Build train and test network
    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    test_prog = fluid.Program()
    b_out, optimizer = build_program(
        is_train=True,
        main_prog=train_prog,
        startup_prog=startup_prog,
        args=args,
        dist_strategy=dist_strategy)

    if args.mlperf_run:
        train_cost, global_lr = b_out
    else:
        train_cost, train_acc1, train_acc5, global_lr = b_out
        train_fetch_vars = [train_cost, train_acc1, train_acc5]
        train_fetch_list = [var.name for var in train_fetch_vars]

    b_out_test = build_program(
        is_train=False,
        main_prog=test_prog,
        startup_prog=fluid.Program(),
        args=args,
        dist_strategy=dist_strategy)
    test_cost, test_acc1, test_acc5 = b_out_test
    test_prog = test_prog.clone(for_test=True)

    # pre allocated memory
    pre_allocate_mem_ratio = 1.1
    max_tmp_train_mem, max_persistable_train_mem = get_max_memory_info(
        train_prog, args.batch_size_train)
    max_tmp_test_mem, _ = get_max_memory_info(test_prog, args.batch_size_test)
    cudnn_workspace_size = dist_strategy.conv_workspace_size_limit * 1024 * 1024
    max_persistable_train_mem = int(
        1.5 * max_persistable_train_mem)  # 1.5 for FP32 and FP16 parameters  
    max_tmp_train_mem = int(pre_allocate_mem_ratio *
                            max_tmp_train_mem) + cudnn_workspace_size
    max_tmp_test_mem = int(pre_allocate_mem_ratio * max_tmp_test_mem)
    pre_allocate_memory(max_tmp_train_mem + max_tmp_test_mem, place)

    # initialize parameters
    exe = fluid.Executor(place)
    exe.run(startup_prog)
    optimizer.amp_init(place)

    build_strategy.enable_auto_fusion = False
    test_prog = compiler.CompiledProgram(test_prog).with_data_parallel(
        loss_name=test_cost.name,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy)

    train_batch_size = args.batch_size_train
    if trainer_id == 0:
        print("train_batch_size: %d device_num:%d" %
              (train_batch_size, num_trainers))

    test_batch_size = args.batch_size_test
    output_dtype = 'float16' if args.pure_fp16 else 'float32'
    train_iter = dali.build(
        args,
        mode='train',
        trainer_id=trainer_id,
        num_trainers=num_trainers,
        device_id=gpu_id,
        output_dtype=output_dtype)
    test_iter = dali.build(
        args,
        mode='val',
        trainer_id=trainer_id,
        num_trainers=num_trainers,
        device_id=gpu_id,
        output_dtype=output_dtype)

    test_fetch_vars = [test_cost, test_acc1, test_acc5]
    test_fetch_list = [var.name for var in test_fetch_vars]

    train_exe = exe
    timer_start1 = time.time()

    real_val_size = int(
        math.floor(1.0 * (trainer_id + 1) * args.val_size / num_trainers) -
        math.floor(1.0 * trainer_id * args.val_size / num_trainers))
    indivisible = False
    if real_val_size % test_batch_size != 0:
        indivisible = True
        last_batch_idx = real_val_size // test_batch_size
        last_batch_size = real_val_size % test_batch_size

    if trainer_id == 0:
        paddle_resnet_print_end(key=constants.INIT_STOP)
        paddle_resnet_print_start(key=constants.RUN_START)
        paddle_resnet_print_event(
            key=constants.TRAIN_SAMPLES, val=args.total_images)
        paddle_resnet_print_event(key=constants.EVAL_SAMPLES, val=args.val_size)

        block_epoch_start = 1
        paddle_resnet_print_start(
            key=constants.BLOCK_START,
            metadata={
                'first_epoch_num': block_epoch_start,
                'epoch_count': args.eval_per_train_epoch
            })

    fleet.barrier_worker()

    scope = paddle.static.global_scope()
    image_t = scope.var('feed_image').get_tensor()
    label_t = scope.var('feed_label').get_tensor()

    image_test_t = scope.var('feed_image_test').get_tensor()
    label_test_t = scope.var('feed_label_test').get_tensor()

    train_graph = None
    train_graph_capture_batch_id = 1
    assert train_graph_capture_batch_id != 0

    for pass_id in range(1, args.num_epochs + 1):
        if trainer_id == 0:
            paddle_resnet_print_start(
                key=constants.EPOCH_START, metadata={'epoch_num': pass_id})
        time_record = []
        reader_record = []

        train_epoch_start = time.time()

        t1 = time.time()
        pass_end = False
        batch_id = 0
        data_iter = iter(train_iter)
        next_batch_data = next(data_iter)

        while not pass_end:
            data = next_batch_data
            reader_time = time.time() - t1
            if train_graph:
                image_t._copy_from(data[0]['feed_image'], exe_place)
                label_t._copy_from(data[0]['feed_label'], exe_place)
                train_graph.replay()
            else:
                if args.mlperf_run or batch_id % args.fetch_steps != 0:
                    if args.mlperf_run and batch_id == train_graph_capture_batch_id:
                        image_t._copy_from(data[0]['feed_image'], exe_place)
                        label_t._copy_from(data[0]['feed_label'], exe_place)
                        data = None

                        train_graph = CUDAGraph()
                        train_graph.capture_begin()
                        pre_allocate_memory(max_tmp_train_mem, place)
                    train_exe.run(train_prog, feed=data)
                    if args.mlperf_run and batch_id == train_graph_capture_batch_id:
                        train_graph.capture_end()
                        if trainer_id == 0:
                            print('CUDA Graph captured successfully')
                        train_graph.replay()
                else:
                    loss, acc1, acc5 = train_exe.run(
                        train_prog, feed=data, fetch_list=train_fetch_list)

            t2 = time.time()
            period = t2 - t1
            t1 = t2
            try:
                next_batch_data = next(data_iter)
            except StopIteration:
                pass_end = True

            time_record.append(period)
            reader_record.append(reader_time)

            if batch_id > 0 and batch_id % args.fetch_steps == 0:
                period = np.mean(time_record)
                reader_time = np.mean(reader_time)
                speed = train_batch_size * 1.0 / period * num_trainers
                time_record, reader_record = [], []

                if args.mlperf_run and trainer_id == 0:
                    print(
                        "Train Pass {0}, batch {1}, rtime {2:.4f}, time {3:.4f}, speed {4:.2f}".
                        format(pass_id, batch_id, reader_time, period, speed))
                elif trainer_id == 0:
                    print(
                        "Train Pass {0}, batch {1}, loss {2}, acc1 {3}, acc5 {4}, rtime{5}, time {6}, speed {7}".
                        format(pass_id, batch_id, "%.5f" % loss, "%.5f" % acc1,
                               "%.5f" % acc5, "%2.4f sec" % reader_time,
                               "%2.4f sec" % period, "%.2f" % speed))
                sys.stdout.flush()
            batch_id += 1

        train_iter.reset()

        train_epoch_end = time.time()
        if trainer_id == 0:
            paddle_resnet_print_end(
                key=constants.EPOCH_STOP, metadata={"epoch_num": pass_id})
            print("Timer train epoch: {} sec".format(train_epoch_end -
                                                     train_epoch_start))

        if args.do_test and (pass_id >= args.eval_offset) and ((
            (pass_id - args.eval_offset) % args.eval_per_train_epoch) == 0):
            if trainer_id == 0:
                paddle_resnet_print_start(
                    key=constants.EVAL_START, metadata={'epoch_num': pass_id})
            test_start = time.time()

            local_loss_sum = 0
            local_top1_count = 0
            local_top5_count = 0
            local_img_count = 0

            for i, data in enumerate(test_iter):
                # parallel test acc
                # filter out duplicated sample in the last batch
                if indivisible and i == last_batch_idx:
                    cur_batch_size = last_batch_size
                    image_test_t._copy_from(data[0]['feed_image_test'],
                                            exe_place, cur_batch_size)
                    label_test_t._copy_from(data[0]['feed_label_test'],
                                            exe_place, cur_batch_size)
                    data = None
                else:
                    cur_batch_size = test_batch_size
                loss, acc1, acc5 = exe.run(program=test_prog,
                                           feed=data,
                                           fetch_list=test_fetch_list)

                if not args.mlperf_run and trainer_id == 0 and i % args.fetch_steps == 0:
                    print(
                        "Testing Pass {0}, batch {1}, #image {2},loss {3}, acc1 {4}, acc5 {5}".
                        format(pass_id, i, cur_batch_size, "%.5f" % loss, "%.5f"
                               % acc1, "%.5f" % acc5))

                # calculate the count of acc and image in local worker
                local_loss_sum += loss
                local_top1_count += np.rint(1.0 * acc1[0] * cur_batch_size)
                local_top5_count += np.rint(1.0 * acc5[0] * cur_batch_size)
                local_img_count += cur_batch_size

            test_iter.reset()

            if num_trainers > 1:
                """
                calculate global acc for entire val set by weighted average the acc by the exact batch size in each worker 
                """
                origin = np.array([
                    local_loss_sum[0], local_top1_count, local_top5_count,
                    local_img_count
                ]).astype(np.float64)
                result = np.zeros([4], dtype=np.float64)
                mpi_comm.Allreduce([origin, MPI.DOUBLE], [result, MPI.DOUBLE])
                global_loss = result[0] / result[3]
                global_acc1 = result[1] / result[3]
                global_acc5 = result[2] / result[3]
                total_val_sample = result[3]
            else:
                global_loss = local_loss_sum / local_img_count
                global_acc1 = local_top1_count / local_img_count
                global_acc5 = local_top5_count / local_img_count
                total_val_sample = local_img_count

            test_end = time.time()
            test_time = test_end - test_start
            if trainer_id == 0:
                print(
                    "Test Pass {0}, loss {1}, acc1 {2}, acc5 {3}, nsamples {4}, time {5}".
                    format(pass_id, "%.5f" % global_loss, "%.5f" % global_acc1,
                           "%.5f" % global_acc5, total_val_sample, "%2.4f sec" %
                           test_time))

                paddle_resnet_print_end(
                    key=constants.EVAL_STOP, metadata={'epoch_num': pass_id})
                paddle_resnet_print_event(
                    key=constants.EVAL_ACCURACY,
                    val=global_acc1,
                    metadata={'epoch_num': pass_id})
                paddle_resnet_print_end(
                    key=constants.BLOCK_STOP,
                    metadata={'first_epoch_num': pass_id})

            if global_acc1 >= args.mlperf_threshold:
                timer_tounch_threshold = time.time()
                t2t_with_init = timer_tounch_threshold - timer_start1
                if trainer_id == 0:
                    print("Touch threshold in {0} pass, test_acc1: {1}".format(
                        pass_id, "%.5f" % global_acc1))
                    print("Time2train: {0} sec".format("%.5f" % t2t_with_init))
                    paddle_resnet_print_end(
                        key=constants.RUN_STOP, metadata={'status': 'success'})
                return

            if pass_id < args.num_epochs:
                block_epoch_start = pass_id + 1
                if trainer_id == 0:
                    paddle_resnet_print_start(
                        key=constants.BLOCK_START,
                        metadata={
                            'first_epoch_num': block_epoch_start + 1,
                            'epoch_count': args.eval_per_train_epoch
                        })

    if trainer_id == 0:
        paddle_resnet_print_end(
            key=constants.RUN_STOP, metadata={'status': 'aborted'})


def print_paddle_environments():
    if trainer_id == 0:
        print('--------- Configuration Environments -----------')
        for k in os.environ:
            if "PADDLE_" in k or "FLAGS_" in k:
                print("%s: %s" % (k, os.environ[k]))
        print('------------------------------------------------')


def save_env(local_file):
    with open(local_file, "w") as f:
        f.write(json.dumps(dict(os.environ), sort_keys=True, indent=2))


def main():
    if trainer_id == 0:
        paddle_resnet_print_event(key=constants.SUBMISSION_ORG, val="Baidu")
        paddle_resnet_print_event(
            key=constants.SUBMISSION_PLATFORM, val="1 x NVIDIA A100 GPU")
        paddle_resnet_print_event(
            key=constants.SUBMISSION_DIVISION, val="closed")
        paddle_resnet_print_event(key=constants.SUBMISSION_STATUS, val="onprem")
        paddle_resnet_print_event(
            key=constants.SUBMISSION_BENCHMARK, val="resnet")
        paddle_resnet_print_event(key=constants.CACHE_CLEAR, val=True)
    save_env('./paddle_env_{}.json'.format(trainer_id))
    args = parse_args()
    if args.random_seed > 0:
        paddle.seed(args.random_seed)
    if trainer_id == 0:
        paddle_resnet_print_event(
            key=constants.MODEL_BN_SPAN, val=args.batch_size_train)
    # this distributed benchmark code can only support gpu environment.
    if trainer_id == 0:
        print_arguments(args)
        print_paddle_environments()
    train(args)


if __name__ == '__main__':
    paddle.enable_static()
    main()
