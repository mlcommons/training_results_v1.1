## DL params
export BATCH_SIZE=55296
export DGXNGPU=8

export CONFIG="/bm_utils/dgx_a100.py"

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=00:10:00
export OMPI_MCA_btl="^openib"
#export MOUNTS=/raid:/raid
export MOUNTS=/opt/microsoft:/opt/microsoft,${PWD}/run_and_time.sh:/bm_utils/run_and_time.sh,${PWD}/dgx_a100.py:/bm_utils/dgx_a100.py,${PWD}/dgx_a100_14x8x640.py:/bm_utils/dgx_a100_14x8x640.py,${PWD}/dgx_a100_8x8x1120.py:/bm_utils/dgx_a100_8x8x1120.py,/mnt/resource_nvme/mlcommons/v1.1/bm_data/dlrm_data:/data
export CUDA_DEVICE_MAX_CONNECTIONS=2
