# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import math
import paddle
import paddle.distributed.fleet as fleet
from paddle.distributed.fleet.meta_optimizers import AMPOptimizer, GraphExecutionOptimizer
from paddle.fluid.contrib.layers.nn import pow2_decay_with_linear_warmup
from mlperf_logging.mllog import constants
from utils.mlperf_logging_helper import paddle_resnet_print_event
import utils.utility as utility


def get_optimizer_and_lr(args):
    num_trainers = utility.get_num_trainers()
    global_batch_size = args.batch_size_train * num_trainers

    steps_per_pass = int(math.ceil(args.total_images * 1.0 / global_batch_size))
    base_lr = args.lr
    warmup_steps = int(steps_per_pass * args.warmup_epochs)
    total_steps = int(steps_per_pass * args.pow2_end_epoch)
    end_lr = 0.0001

    lr_sch = pow2_decay_with_linear_warmup(
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        base_lr=base_lr,
        end_lr=end_lr)

    optimizer = paddle.fluid.optimizer.LarsMomentumOptimizer(
        lars_coeff=args.lars_coeff,
        lars_weight_decay=args.lars_weight_decay,
        learning_rate=lr_sch,
        momentum=args.momentum_rate,
        regularization=None,
        exclude_from_weight_decay=['bn', 'batch_norm', '.b_0'],
        epsilon=0,
        multi_precision=True,
        rescale_grad=1.0 / global_batch_size)

    if utility.get_trainer_id() == 0:
        paddle_resnet_print_event(
            key=constants.LARS_OPT_LR_DECAY_POLY_POWER, val=2)
        paddle_resnet_print_event(key=constants.LARS_OPT_END_LR, val=end_lr)

    return optimizer, lr_sch


def apply_meta_optimizers(main_program, startup_program, optimizer,
                          meta_optimizer_classes, loss, dist_strategy):
    assert id(loss.block.program) == id(main_program)
    dist_optimizer = fleet.distributed_optimizer(
        optimizer, strategy=dist_strategy)
    role_maker = dist_optimizer._role_maker
    meta_optimizers = [optimizer]
    for i, meta_optimizer_class in enumerate(meta_optimizer_classes):
        meta_optimizer = meta_optimizer_class(optimizer)
        meta_optimizer._set_basic_info(loss, role_maker, optimizer,
                                       dist_strategy)
        meta_optimizer._update_inner_optimizer(meta_optimizers[-1])
        meta_optimizers.append(meta_optimizer)
    dist_optimizer = meta_optimizers[-1]
    dist_optimizer.minimize(loss, startup_program)
    return dist_optimizer


def apply_graph_optimization_pass(main_program, startup_program, loss,
                                  dist_optimizer):
    assert id(loss.block.program) == id(main_program)
    optimizer = dist_optimizer.user_defined_optimizer
    role_maker = dist_optimizer.role_maker
    dist_strategy = dist_optimizer.user_defined_strategy

    graph_optimizer = GraphExecutionOptimizer(optimizer)
    graph_optimizer._set_basic_info(loss, role_maker, optimizer, dist_strategy)
    graph_optimizer._update_inner_optimizer(dist_optimizer)
    graph_optimizer.minimize(loss, startup_program)
    return graph_optimizer


def apply_program_optimization_pass(main_program):
    assert main_program.num_blocks == 1
    merge_momentum_ops(main_program)
    merge_lars_ops(main_program)


def merge_lars_ops(main_program):
    max_op_num = 60
    while merge_lars_ops_impl(
            main_program, multi_precision=True, max_op_num=max_op_num) > 0:
        pass

    while merge_lars_ops_impl(
            main_program, multi_precision=False, max_op_num=max_op_num) > 0:
        pass


def merge_lars_ops_impl(main_program, multi_precision, max_op_num):
    main_block = main_program.global_block()
    same_attrs = {
        "mu": None,
        "lars_coeff": None,
        "epsilon": None,
        "rescale_grad": None,
        "op_role": None,
        "op_device": None,
        "multi_precision": multi_precision,
    }

    lars_ops = []
    for i, op in enumerate(main_block.ops):
        if op.type != "lars_momentum":
            continue

        attrs = op.all_attrs()
        if attrs["multi_precision"] != multi_precision:
            continue

        if len(op.input("Param")) > 1:
            continue

        for key in list(same_attrs.keys()):
            if key not in attrs:
                continue

            if same_attrs[key] is None:
                same_attrs[key] = attrs[key]
            elif same_attrs[key] != attrs[key]:
                return 0

        lars_ops.append((i, op))

    if len(lars_ops) <= 1:
        return 0

    if max_op_num is not None and len(lars_ops) > max_op_num:
        lars_ops = lars_ops[0:max_op_num]

    params = []
    grads = []
    velocitys = []
    lrs = []
    lars_weight_decays = []
    master_params = [] if multi_precision else None
    op_role_vars = []
    max_idx = 0
    for i, op in lars_ops:
        params.append(op.input("Param")[0])
        grads.append(op.input("Grad")[0])
        velocitys.append(op.input("Velocity")[0])
        lrs.append(op.input("LearningRate")[0])
        if multi_precision:
            master_params.append(op.input("MasterParam")[0])
        op_attrs = op.all_attrs()
        op_role_vars.extend(op_attrs.get("op_role_var", []))
        lars_weight_decays.append(op_attrs["lars_weight_decay"][0])
        max_idx = max(max_idx, i)

    inputs = {
        "Param": params,
        "Grad": grads,
        "Velocity": velocitys,
        "LearningRate": lrs,
    }
    outputs = {
        "ParamOut": params,
        "VelocityOut": velocitys,
    }
    if multi_precision:
        inputs["MasterParam"] = master_params
        outputs["MasterParamOut"] = master_params
    attrs = dict(same_attrs)
    attrs.update({
        "lars_weight_decay": lars_weight_decays,
        "op_role_var": op_role_vars,
    })
    main_program.global_block()._insert_op(
        max_idx + 1,
        type='lars_momentum',
        inputs=inputs,
        outputs=outputs,
        attrs=attrs)
    for idx, op in reversed(lars_ops):
        main_program.global_block()._remove_op(idx)
    return len(lars_ops)


def merge_momentum_ops(main_program):
    main_block = main_program.global_block()
    momentum_ops = []
    same_attrs = {
        "mu": None,
        "use_nesterov": False,
        "regularization_method": "",
        "regularization_coeff": 0.0,
        "rescale_grad": None,
        "op_role": None,
        "op_device": None,
    }

    for i, op in enumerate(main_block.ops):
        if op.type != 'momentum':
            continue

        attrs = op.all_attrs()
        if attrs['multi_precision']:
            continue

        for key in list(same_attrs.keys()):
            if key not in attrs:
                continue

            if same_attrs[key] is None:
                same_attrs[key] = attrs[key]
            elif same_attrs[key] != attrs[key]:
                return

        momentum_ops.append((i, op))

    if len(momentum_ops) <= 1:
        return

    params = []
    grads = []
    velocitys = []
    lr = None
    master_params = []
    multi_precision = []
    max_idx = 0
    op_role_var = []
    for idx, op in momentum_ops:
        params.append(op.input("Param")[0])
        grads.append(op.input("Grad")[0])
        new_lr = op.input("LearningRate")[0]
        if lr is None:
            lr = new_lr
        if lr != new_lr:
            return

        velocitys.append(op.input("Velocity")[0])
        attrs = op.all_attrs()
        op_role_var.extend(attrs.get("op_role_var", []))
        multi_precision.append(int(attrs["multi_precision"]))
        max_idx = max(max_idx, idx)

    attrs = dict(same_attrs)
    attrs.update({
        'multi_precision': False,
        'op_role_var': op_role_var,
    })
    main_program.global_block()._insert_op(
        max_idx + 1,
        type='merged_momentum',
        inputs={
            'Param': params,
            'Grad': grads,
            'LearningRate': lr,
            'MasterParam': master_params,
            'Velocity': velocitys,
        },
        outputs={
            'ParamOut': params,
            'MasterParamOut': master_params,
            'VelocityOut': velocitys,
        },
        attrs=attrs)

    for idx, op in reversed(momentum_ops):
        main_program.global_block()._remove_op(idx)


def create_strategy(args):
    build_strategy = paddle.static.BuildStrategy()
    exec_strategy = paddle.static.ExecutionStrategy()
    dist_strategy = fleet.DistributedStrategy()

    build_strategy.fuse_bn_act_ops = True
    build_strategy.fuse_elewise_add_act_ops = True
    build_strategy.fuse_bn_add_act_ops = True
    build_strategy.enable_addto = True
    build_strategy.enable_sequential_execution = False
    build_strategy.fix_op_run_order = True
    build_strategy.allow_cuda_graph_capture = True
    build_strategy.enable_auto_fusion = True

    exec_strategy.num_threads = 1
    exec_strategy.num_iteration_per_drop_scope = args.num_iteration_per_drop_scope

    dist_strategy.execution_strategy = exec_strategy
    dist_strategy.build_strategy = build_strategy
    dist_strategy.fuse_all_reduce_ops = True
    dist_strategy.fuse_grad_size_in_MB = args.all_reduce_size
    dist_strategy.nccl_comm_num = 1
    cudnn_deterministic = paddle.get_flags(
        ['FLAGS_cudnn_deterministic'])['FLAGS_cudnn_deterministic']
    dist_strategy.cudnn_exhaustive_search = not cudnn_deterministic
    dist_strategy.conv_workspace_size_limit = 4096  # MB
    dist_strategy.cudnn_batchnorm_spatial_persistent = True
    dist_strategy.sync_nccl_allreduce = paddle.get_flags(
        'FLAGS_sync_nccl_allreduce')['FLAGS_sync_nccl_allreduce']
    if args.fp16:
        dist_strategy.amp = True
        dist_strategy.amp_configs = {
            'init_loss_scaling': 1.0,
            'use_dynamic_loss_scaling': False,
            'use_pure_fp16': args.pure_fp16,
            'use_fp16_guard': True,
        }
    dist_strategy.gradient_scale_configs = {'scale_strategy': 'sum'}

    paddle.set_flags({
        'FLAGS_max_inplace_grad_add': 8,
        'FLAGS_cudnn_batchnorm_spatial_persistent':
        bool(dist_strategy.cudnn_batchnorm_spatial_persistent),
        'FLAGS_conv_workspace_size_limit':
        int(dist_strategy.conv_workspace_size_limit),
        'FLAGS_cudnn_exhaustive_search':
        bool(dist_strategy.cudnn_exhaustive_search),
        'FLAGS_fuse_parameter_memory_size':
        int(dist_strategy.fuse_grad_size_in_MB),
        'FLAGS_fuse_parameter_groups_size':
        int(dist_strategy._fuse_grad_size_in_TFLOPS),
    })

    return build_strategy, exec_strategy, dist_strategy


def apply_backward_and_optimization_passes(main_program,
                                           startup_program,
                                           raw_optimizer,
                                           loss,
                                           dist_strategy,
                                           use_py_passes=True):
    with paddle.static.program_guard(main_program, startup_program):
        if use_py_passes:
            meta_optimizers = [AMPOptimizer] if dist_strategy.amp else []
            dist_optimizer = apply_meta_optimizers(
                main_program, startup_program, raw_optimizer, meta_optimizers,
                loss, dist_strategy)
            apply_program_optimization_pass(main_program)
            apply_graph_optimization_pass(main_program, startup_program, loss,
                                          dist_optimizer)
        else:
            dist_optimizer = fleet.distributed_optimizer(
                raw_optimizer, strategy=dist_strategy)
            dist_optimizer.minimize(loss)

    return dist_optimizer
