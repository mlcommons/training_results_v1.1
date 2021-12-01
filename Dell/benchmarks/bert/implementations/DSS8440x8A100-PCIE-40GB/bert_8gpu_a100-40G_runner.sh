#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export NEXP=10
export NEXP=10
source config_DSS8440x8A100-PCIE-40GB.sh
export LOG_DIR=$(pwd)/node071_results_bz_48

#nvcr.io/nvdlfwea/mlperfv11/bert:20211013.pytorch
CONT=3e47b5f2ca2a ./run_with_docker.sh

