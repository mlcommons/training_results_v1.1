## System config params
export DGXNGPU=8
export DGXSOCKETCORES=24
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1
export SLURM_NTASKS=${DGXNGPU}

## Data Paths
export DATADIR="/mnt/data2/after_steps_bert_ds/2048_shards_uncompressed"
export EVALDIR="/mnt/data2/after_steps_bert_ds/eval_set_uncompressed"
export DATADIR_PHASE2="/mnt/data2/after_steps_bert_ds/2048_shards_uncompressed"
export CHECKPOINTDIR="/mnt/data2/after_steps_bert_ds/cks"
export CHECKPOINTDIR_PHASE1="/mnt/data2/after_steps_bert_ds/cks"

export NCCL_SOCKET_IFNAME=

