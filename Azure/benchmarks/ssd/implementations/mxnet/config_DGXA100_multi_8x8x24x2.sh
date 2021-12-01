## DL params
export BATCHSIZE=24
export NUMEPOCHS=${NUMEPOCHS:-70}
export EXTRA_PARAMS='--lr-decay-epochs 48 60 --lr-warmup-epoch=15 --lr=0.004 --weight-decay=1.7e-4 --dali-workers=8 --bn-group=2 --input-jpg-decode=cache'

## System run parms
export DGXNNODES=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME_MINUTES=20
export WALLTIME=$((${NEXP} * ${WALLTIME_MINUTES}))

## System config params
export DGXNGPU=8
export DGXSOCKETCORES=48
export DGXNSOCKET=2
export DGXHT=1  # HT is on is 2, HT off is 1

## Enable SHARP
#export SBATCH_NETWORK=sharp
