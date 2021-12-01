#!/bin/bash

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8
export NEXP=10
source config_DSS8440x8A100-PCIE-80GB.sh

#nvcr.io/nvdlfwea/mlperfv11/bert:20211013.pytorch
CONT=65504973d334 ./run_with_docker.sh

