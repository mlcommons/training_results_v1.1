## DL params
export BATCHSIZE=64

export NEXP=1
export GRADIENT_STEPS=2
export LR=3.5e-4
export MAX_SAMPLES_TERMINATION=9000000
export MAX_STEPS=30000
export OPT_LAMB_BETA_1=0.9
export OPT_LAMB_BETA_2=0.999
export START_WARMUP_STEP=0
export WARMUP_PROPORTION=0.0

export EXTRA_PARAMS="--dense_seq_output --unpad --unpad_fmha --exchange_padding"
export PHASE=2
export EVAL_ITER_START_SAMPLES=150000
export EVAL_ITER_SAMPLES=150000

## System run parms
export PGNNODES=1
export PGSYSTEM=PG
export WALLTIME=01:15:00

## System config params
source ${BASH_SOURCE%/*}/config_PG_common.sh
