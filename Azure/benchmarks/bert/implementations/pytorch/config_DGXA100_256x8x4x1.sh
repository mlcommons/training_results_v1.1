## DL params
# steps500_lr0.004_wup0.1_b1_0.878_b2_0.974
export BATCHSIZE=3
export GRADIENT_STEPS=1
export LR=0.0029293
export MAX_SAMPLES_TERMINATION=12000000
export MAX_STEPS=700
export OPT_LAMB_BETA_1=0.7206
export OPT_LAMB_BETA_2=0.78921
export START_WARMUP_STEP=-700
export WEIGHT_DECAY_RATE=0.001
export WARMUP_STEPS=0
#export SBATCH_NETWORK=sharp
export NCCL_GRAPH_REGISTER=1
export EXTRA_PARAMS="--use_cuda_graph --pad_fmha --cuda_graph_mode 'full_iteration' --max_iterations_per_graph 1 --fused_bias_fc --fused_bias_mha "
export PHASE=2
export EVAL_ITER_START_SAMPLES=225000
export EVAL_ITER_SAMPLES=225000

## System run parms
export DGXNNODES=256
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=00:70:00

## System config params
source ./config_DGXA100_common.sh

export CONTAINER_PRELOAD_LUSTRE=0
export USE_DDP=1
