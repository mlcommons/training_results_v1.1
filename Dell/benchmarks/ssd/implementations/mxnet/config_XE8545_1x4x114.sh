# configuration file for 1xXE8545x4A100-SXM-40GB

## DL params
export BATCHSIZE=114
export NUMEPOCHS=${NUMEPOCHS:-65}
#export EXTRA_PARAMS='--lr-warmup-epoch=5 --lr=0.003157 --weight-decay=1.3e-4 --dali-workers=8 --input-jpg-decode=cache'
export EXTRA_PARAMS='--lr-warmup-epoch=5.25 --lr=0.00285 --weight-decay=1.6e-4 --dali-workers=8'

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
#WALLTIME_MINUTES=20
#export WALLTIME=$((${NEXP} * ${WALLTIME_MINUTES}))

## System config params
export DGXNGPU=4
export DGXSOCKETCORES=64
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1
export NCCL_SOCKET_IFNAME=
