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

set -x

export PYTHON=python3.8

export PADDLE_TRAINERS_NUM=`$PYTHON -c "import paddle; print(paddle.device.cuda.device_count())"`

default_addr='127.0.0.1'
export PADDLE_TRAINER_ENDPOINTS=`$PYTHON -c "print(\",\".join([\"$default_addr:\" + str(60001 + i) for i in range($PADDLE_TRAINERS_NUM)]))"`

ORTERUN=`which orterun`

CMD="bash train_imagenet.sh 10.5 2 37 0.00005 8 0.9"
NUMPY_SEED=`$PYTHON -c "import numpy; print(numpy.random.randint(1, 100000))"`
export SEED=${SEED:-"$NUMPY_SEED"}

if [[ $PADDLE_TRAINERS_NUM -eq 1 ]]; then
  numactl --physcpubind=0-31 --membind=0 -- $CMD
else	
  mpirun="$ORTERUN --allow-run-as-root \
     -np $PADDLE_TRAINERS_NUM \
     -mca btl_tcp_if_exclude docker0,lo,matrixdummy0,matrix0 \
     --bind-to none -x PADDLE_TRAINERS_NUM \
     -x PADDLE_TRAINER_ENDPOINTS -x LD_LIBRARY_PATH -x SEED -x PYTHON"

  BIND="bash bind.sh --cpu=exclusive --"
  $mpirun ${BIND} $CMD
fi
