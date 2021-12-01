## DL params
export BATCHSIZE=48
#export NUMEPOCHS=${NUMEPOCHS:-65}
#export EXTRA_PARAMS='--lr-warmup-epoch=5 --lr=0.003157 --weight-decay=1.3e-4 --dali-workers=8 --input-jpg-decode=cache'
#export EXTRA_PARAMS='--lr-decay-epochs 52 65 --lr-warmup-epoch=15 --lr=0.0035 --weight-decay=1.7e-4 --gradient-predivide-factor=4 --dali-workers 8'
export EXTRA_PARAMS='--lr-decay-epochs 52 65 --lr-warmup-epoch=15 --lr=0.0035 --weight-decay=1.7e-4 --dali-workers 8'

## System run parms
export DGXNNODES=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME_MINUTES=20
export WALLTIME=100

## System config params
export DGXNGPU=4
export DGXSOCKETCORES=32
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1
#export NCCL_SOCKET_IFNAME=
export OMPI_MCA_btl_openib_if_include=mlx5_0:1
export UCX_NET_DEVICES=mlx5_0:1

export SBATCH_NETWORK=sharp
