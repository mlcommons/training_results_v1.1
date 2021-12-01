## System config params
export DGXNGPU=8
export DGXSOCKETCORES=32
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1
export SLURM_NTASKS=${DGXNGPU}

## Data Paths
export DATADIR="/mlperf/data/training_v1.1_data_preprocessed/bert/hdf5/training-4320/hdf5_4320_shards_varlength"
export EVALDIR="/mlperf/data/training_v1.1_data_preprocessed/bert/hdf5/eval_varlength"
export DATADIR_PHASE2="/mlperf/data/training_v1.1_data_preprocessed/bert/hdf5/training-4320/hdf5_4320_shards_varlength"
export CHECKPOINTDIR="/mlperf/data/training_v1.1_data_preprocessed/bert/phase1"
#using existing checkpoint_phase1 dir
export CHECKPOINTDIR_PHASE1="/mlperf/data/training_v1.1_data_preprocessed/bert/phase1"
export UNITTESTDIR="/lustre/fsw/mlperf/mlperft-bert/unit_test"
