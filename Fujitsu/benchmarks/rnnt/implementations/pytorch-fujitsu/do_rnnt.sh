rnnt_base=/mnt/data3/work/rnnt
datasets=$rnnt_base/datasets
checkpoints=$rnnt_base/checkpoints
#results=$rnnt_base/results
results=$(realpath ../rnnt-logs)
tokenized=$rnnt_base/tokenized
sentencepieces=$rnnt_base/sentencepieces
container=nvcr.io/nvdlfwea/mlperfv11/rnnt:20211012.pytorch

source config_GX2460.sh
CONT=$container DATADIR=$datasets LOGDIR=$results METADATA_DIR=$tokenized NEXP=10 \
  SENTENCEPIECES_DIR=$sentencepieces bash ./run_with_docker.sh
