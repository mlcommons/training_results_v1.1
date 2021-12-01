## DL params
export BATCHSIZE=114
export NUMEPOCHS=${NUMEPOCHS:-65}
export EXTRA_PARAMS='--lr-warmup-epoch=3 --lr=0.003157 --weight-decay=1.3e-4 --dali-workers=4 --input-jpg-decode=cache'

## System run parms
export NEXP=1
export PGNNODES=1
export PGSYSTEM=PG
WALLTIME_MINUTES=20
export WALLTIME=$((${NEXP} * ${WALLTIME_MINUTES}))
export PASSWORD=kai

## System config params
export PGNGPU=4
export PGSOCKETCORES=16
export PGSOCKET=2
export PGHT=2         # HT is on is 2, HT off is 1
