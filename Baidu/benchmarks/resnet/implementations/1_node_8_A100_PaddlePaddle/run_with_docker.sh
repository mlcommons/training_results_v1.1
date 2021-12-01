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

set -euxo pipefail

: "${CONT:?CONT not set}"
: "${REPEAT:=5}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
: "${DATADIR:=/data}"
: "${LOGDIR:=$PWD/results}"

seed=${SEED:-}
logfile="${LOGDIR}/${DATESTAMP}"
cont_name=paddle-resnet
cont_mounts=("--volume=${DATADIR}:/data" "--volume=${LOGDIR}:/results")

# Setup directories
mkdir -p "${LOGDIR}"

# Cleanup container
cleanup_docker() {
    docker container rm -f "${cont_name}" || true
}
cleanup_docker
trap 'set -eux; cleanup_docker' EXIT

# Setup container
nvidia-docker run --rm --init --detach \
    --net=host --uts=host --ipc=host --security-opt=seccomp=unconfined \
    --ulimit=stack=67108864 --ulimit=memlock=-1 \
    --name="${cont_name}" "${cont_mounts[@]}" \
    "${CONT}" sleep infinity
# Make sure container has time to finish initialization
sleep 30
docker exec -i "${cont_name}" true

# Run training
for training_index in $(seq 1 "${REPEAT}"); do
    (
        echo "Beginning trial ${training_index} of ${REPEAT}"

        # Clear caches
        if [ "${CLEAR_CACHES}" -eq 1 ]; then
            sync && sudo /sbin/sysctl vm.drop_caches=3
        fi

        # Run training
        export SEED=${seed:-$RANDOM}
        docker exec -e SEED -i "${cont_name}" bash run_and_time.sh
    ) |& tee "${logfile}_${training_index}.log"
done
