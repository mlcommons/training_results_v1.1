# Description

This repo provides an environment for training ResNet50 using NGC21.05 MXNet.

# Instruction

## 1. Build the ResNet50 NGC21.05 MXNet container

```
docker build -t mlperf-nvidia:image_classification .
```

## 2. Prepare data

You can refer to [DeepLearningExamples/MxNet/Classification/RN50v1.5 at master · NVIDIA/DeepLearningExamples (github.com)](https://github.com/NVIDIA/DeepLearningExamples/tree/master/MxNet/Classification/RN50v1.5#quick-start-guide) to prepare the input data.

After the preparation, you should get 4 data files: `train.rec`, `train.idx`, `val.rec`, `val.idx` .

## 3. Start to train

Run the following command on the 1x8 A100 machine, and you should see the logging results on the `LOGDIR` directory.

```bash
source config_DGXA100.sh
export CONT=mlperf-nvidia:image_classification 
export DATADIR=<path/to/data/dir> 
export LOGDIR=<path/to/output/dir> 
export SEED=<specified/random/seed> # optional 
bash run_with_docker.sh
```
