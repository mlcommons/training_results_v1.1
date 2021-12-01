###  ============== config_gigabyte_common.sh  ======================
## System config params
export DGXNGPU=8
export DGXSOCKETCORES=32
export DGXNSOCKET=2
export DGXHT=1         # HT is on is 2, HT off is 1
export SLURM_NTASKS=${DGXNGPU}

## Data Paths
export DATADIR="/home/gigabyte/raid/bert_dataset_process/hdf5/training-2048/hdf5_2048_shards_uncompressed"
export EVALDIR="/home/gigabyte/raid/bert_dataset_process/hdf5/eval"
export DATADIR_PHASE2="/home/gigabyte/raid/bert_dataset_process/hdf5/training-2048/hdf5_2048_shards_uncompressed"
export CHECKPOINTDIR="/home/gigabyte/raid/bert_dataset_process/checkpointdir"
#using existing checkpoint_phase1 dir
export CHECKPOINTDIR_PHASE1="/home/gigabyte/raid/bert_dataset_process/phase1"
export UNITTESTDIR="/lustre/fsw/mlperf/mlperft-bert/unit_test"




###  ============== config_gigabyte.sh  ======================
## DL params
export BATCHSIZE=56
export GRADIENT_STEPS=1
export LR=3.5e-4
export MAX_SAMPLES_TERMINATION=4500000
export MAX_STEPS=7800
export OPT_LAMB_BETA_1=0.9
export OPT_LAMB_BETA_2=0.999
export START_WARMUP_STEP=0
export WARMUP_PROPORTION=0.0

export EXTRA_PARAMS="--dense_seq_output --unpad --unpad_fmha --exchange_padding"
export PHASE=2
export EVAL_ITER_START_SAMPLES=150000
export EVAL_ITER_SAMPLES=150000

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=01:15:00

## System config params
source config_gigabyte_common.sh