container_name=nvcr.io/nvdlfwea/mlperfv11/unet3d:20211012.mxnet
data_dir=/mnt/data3/work/3d-unet/data-dir
result_dir=$(realpath ../unet3d-logs)

source config_GX2460.sh # or config_DGX1_conv-dali_1x8x4.sh or config_DGXA100_conv-dali_1x8x7.sh
CONT=$container_name DATADIR=$data_dir LOGDIR=$result_dir NEXP=40 ./run_with_docker.sh
