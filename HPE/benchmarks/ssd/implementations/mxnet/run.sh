#!/bin/bash


source config_HPEProLiantXL675dGen10Plus.sh

CONT=mlperf-ssd:single_stage_detector DATADIR=/lvol/manju/mlperf/coco/ LOGDIR=/lvol/manju/mlperf-training/ssd/ PRETRAINED_DIR=/lvol/manju/mlperf-training/ssd/models/ ./run_with_docker.sh 2>&1|tee /lvol/manju/mlperf-training/ssd/logs/run6.log

CONT=mlperf-ssd:single_stage_detector DATADIR=/lvol/manju/mlperf/coco/ LOGDIR=/lvol/manju/mlperf-training/ssd/ PRETRAINED_DIR=/lvol/manju/mlperf-training/ssd/models/ ./run_with_docker.sh 2>&1|tee /lvol/manju/mlperf-training/ssd/logs/run7.log
