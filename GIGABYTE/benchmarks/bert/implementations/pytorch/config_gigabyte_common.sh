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
