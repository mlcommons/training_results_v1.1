## System config params
export PGNGPU=4
export PGSOCKETCORES=16
export PGNSOCKET=2
export PGHT=2         # HT is on is 2, HT off is 1
export SLURM_NTASKS=${PGNGPU}
export CUDA_VISIBLE_DEVICES="0,1,2,3"

## Data Paths
export DATADIR="/mnt/data3/work/bert_data/hdf5/training-4320/hdf5_4320_shards_varlength"
export EVALDIR="/mnt/data3/work/bert_data/hdf5/eval_varlength"
export DATADIR_PHASE2="/mnt/data3/work/bert_data/hdf5/training-4320/hdf5_4320_shards_varlength"
export CHECKPOINTDIR="$CI_BUILDS_DIR/$SLURM_ACCOUNT/$CI_JOB_ID/ci_checkpoints"
#using existing checkpoint_phase1 dir
export CHECKPOINTDIR_PHASE1="/mnt/data3/work/bert_data/phase1"
export UNITTESTDIR="/home/kai/training_v1.1/bert/unit_test"
