#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9
export NEXP=10
source config_DSS8440x10A100-PCIE-80GB.sh

#nvcr.io/nvdlfwea/mlperfv11/bert:20211013.pytorch
CONT=b396a0da1369 ./run_with_docker.sh

