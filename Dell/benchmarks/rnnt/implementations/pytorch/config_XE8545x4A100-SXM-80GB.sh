## System config params
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export DGXNGPU=4
export DGXSOCKETCORES=64
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1

## Run specific params
#export DATADIR="/raid/datasets/rnnt/"
export METADATA_DIR
export SENTENCEPIECES_DIR

export BATCHSIZE=512
export EVAL_BATCHSIZE=676
#export BATCHSIZE=768
#export EVAL_BATCHSIZE=1014
export GRAD_ACCUMULATION_STEPS=8
WALLTIME_MINUTES=60
export WALLTIME=$(( ${NEXP:-1} * ${WALLTIME_MINUTES} ))
export MAX_SYMBOL=300
export DATA_CPU_THREADS=16

source $(dirname ${BASH_SOURCE[0]})/hyperparameters_2048.sh
#source $(dirname ${BASH_SOURCE[0]})/hyperparameters_3072.sh

## Opt flag
export FUSE_RELU_DROPOUT=true
export MULTI_TENSOR_EMA=true
export BATCH_EVAL_MODE=cg_unroll_pipeline
export APEX_LOSS=fp16
export APEX_JOINT=pack_w_relu_dropout
export AMP_LVL=2
export BUFFER_PREALLOC=true
export VECTORIZED_SA=true
export EMA_UPDATE_TYPE=fp16
export DIST_LAMB=false
export MULTILAYER_LSTM=true
export ENABLE_PREFETCH=true
export BATCH_SPLIT_FACTOR=1
export TOKENIZED_TRANSCRIPT=true
export VECTORIZED_SAMPLER=true
export DIST_SAMPLER=true
export MIN_SEQ_SPLIT_LEN=20
export APEX_MLP=true
export PRE_SORT_FOR_SEQ_SPLIT=true
export LOG_FREQUENCY=1000

export NCCL_SOCKET_IFNAME=
