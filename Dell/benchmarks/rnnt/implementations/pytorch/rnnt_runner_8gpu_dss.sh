#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8
source NEXP=10 
#source config_DSS8440x10A100-PCIE-80GB.sh
source config_DSS8440x8A100-PCIE-80GB.sh


# nvcr.io/nvdlfwea/mlperfv11/rnnt:20211013.pytorch
#nvcr.io/nvdlfwea/mlperfv11/rnnt:20211013.8gpunuma
CONT=34fcf23eee08 ./run_with_docker.sh


