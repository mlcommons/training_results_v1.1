#!/bin/bash

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

export FLAGS_enable_cublas_tensor_op_math=True

export FLAGS_sync_nccl_allreduce=0
export FLAGS_call_stack_level=2
export FLAGS_cudnn_deterministic=1

# argument
lr=${1:-10.5}
warmup_epochs=${2:-2}
pow2_end_epoch=${3:-37}
lars_weight_decay=${4:-0.0001}
num_epochs=$(($pow2_end_epoch + 0))
batch_size_train=${5:-256}
momentum_rate=${6:-0.9}

echo "training hyper-parameters"
echo $lr $warmup_epochs $pow2_end_epoch $lars_weight_decay $batch_size_train $momentum_rate

set -x

data_dir=/data

OMPI_COMM_WORLD_RANK=${OMPI_COMM_WORLD_RANK:-"0"}

function get_device_id() {
$PYTHON <<EOF
import paddle
import os
gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
if gpus is None:
    print($OMPI_COMM_WORLD_RANK)
else:
    gpus = gpus.split(",")
    print(gpus[$OMPI_COMM_WORLD_RANK])
EOF
}	

export PADDLE_TRAINER_ID=${OMPI_COMM_WORLD_RANK}
export PADDLE_TRAINERS_NUM=${PADDLE_TRAINERS_NUM:-"1"} 
export PADDLE_TRAINER_ENDPOINTS=${PADDLE_TRAINER_ENDPOINTS:-""}
if [[ ${PADDLE_TRAINERS_NUM} -gt 1 ]]; then 
  export CUDA_VISIBLE_DEVICES=`get_device_id`
else
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}
fi

export NCCL_BUFFSIZE=2097152
export NCCL_CHECKS_DISABLE=1
export NCCL_CHECK_POINTERS=0
export NCCL_COLLNET_ENABLE=0
export NCCL_DISABLE_CHECKS=1
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_MAX_RINGS=8
export NCCL_NET_GDR_READ=1
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_IFNAME=docker0
export NCCL_VERSION=2.9.8

LOG_DIR="log_${PADDLE_TRAINERS_NUM}"
mkdir -p $LOG_DIR
LOG_FILE="${LOG_DIR}/worker.${OMPI_COMM_WORLD_RANK}"

mkdir -p profile_outs
# PROFILE="nsys profile -t cublas,cuda,cudnn,nvtx --stats true -o profile_outs/prof_paddle_$OMPI_COMM_WORLD_RANK -y 120 -d 10 -f true" 

export FLAGS_CUDA_GRAPH_USE_STANDALONE_EXECUTOR=1
export SEED=23344
python3.8 -u ./train_imagenet.py \
       --model="ResNet50_clas" \
       --total_images=1281167 \
       --data_dir=${data_dir} \
       --class_dim=1000 \
       --image_shape=4,224,224 \
       --num_epochs=$num_epochs \
       --fp16=True \
       --data_format="NHWC" \
       --do_test=True \
       --val_size=50000 \
       --batch_size_train=$batch_size_train \
       --batch_size_test=$batch_size_train \
       --lr=$lr \
       --momentum_rate=$momentum_rate \
       --warmup_epochs=$warmup_epochs \
       --pow2_end_epoch=$pow2_end_epoch \
       --lars_coeff=0.001 \
       --lars_weight_decay=$lars_weight_decay \
       --label_smooth=0.1 \
       --truncnorm_init=True \
       --eval_per_train_epoch=4 \
       --eval_offset=3 \
       --mlperf_threshold=0.759 \
       --pure_fp16=True \
       --mlperf_run=True \
       --all_reduce_size=0 \
       --dali_num_threads=6 \
       --random_seed=$SEED \
       --num_iteration_per_drop_scope=100000000 \
       --fetch_steps=20 2>&1 | tee $LOG_FILE
