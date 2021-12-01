#!/bin/bash
set -ex 

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8
source config_DSS8440x8A100-PCIe-80GB.sh
export NEXP=40

#nvcr.io/nvdlfwea/mlperfv11/unet3d:20211013.mxnet
CONT=661d98de6546 DATADIR=/mnt/data2/unet3d LOGDIR=/home/rakshith/mlperf_training_v1.1/20211013/unet3d/scripts/results_node070/ ./run_with_docker.sh


