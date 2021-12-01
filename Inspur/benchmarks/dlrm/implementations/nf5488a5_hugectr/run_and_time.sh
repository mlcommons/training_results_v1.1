#!/bin/bash

cd ../hugectr
source config_NF5488A5.sh
CONT=mlperf-inspur:dlrm DATADIR=/path/to/preprocessed/data LOGDIR=/path/to/logfile ./run_with_docker.sh
