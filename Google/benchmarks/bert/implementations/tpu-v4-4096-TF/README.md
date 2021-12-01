# Lingvo-TF large language model

## Benchmark Information

Language large model description

## Software

[Lingvo](https://github.com/tensorflow/lingvo)

### Publication/Attribution

[Lingvo: a Modular and Scalable Framework for Sequence-to-Sequence Modeling]
(https://arxiv.org/pdf/1902.08295.pdf)

## Hardware

TPU v4 in the cloud.

## Model

Model configuration:

NUM_TRANSFORMER_LAYERS = 64
HIDDEN_DIM = 131072
MODEL_DIM = 16384
NUM_HEADS = 256

total #params: 481,815,347,200

Model configuration can be found
[here](https://github.com/tensorflow/lingvo/blob/master/lingvo/tasks/lm/params/wiki_bert.py)

## Dataset

The dataset is the same as the one used in the closed division.

## Instructions to run

1. Create TPU VM

   ```
   gcloud alpha compute tpus tpu-vm create ${TPU_NAME} --accelerator-type=v4-4096 --project={GCP_PROJECT} --zone=us-central2-b --version=v2-tpuv4-mlperf-pod
   ```

2. SSH to the TPU VM

   ```
   gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --project=${GCP_PROJECT} --zone=us-central2-b 2>&1 | tee results_$(date +%s).txt
   ```

2. Get the code

   Use the source code in the submission package: 

   ```
   benchmarks/bert/implementations/bert-cloud-TF-tpu-v4-4096
   ```

   Or you an get it from:

   ```
   git clone -b lingvo_mlperf_1.1 https://github.com/tensorflow/lingvo.git
   ```

3. Build docker container

   ```
   sudo docker build --tag tensorflow:lingvo_dev_lm - < lingvo/docker/dev.dockerfile
   ```

4. Run docker

   ```
   sudo docker run --rm -it -v /home/$(whoami)/lingvo:/tmp/lingvo --name lingvo tensorflow:lingvo_dev_lm bash
   ```

5. Run benchmark

   You will need to have access to the 2TB checkpoint (TBD)) and the data set (TBD)

   Set the log directory, e.g.
   ```
   export LOG_DIR=gs://my_log_dir/
   ```

   then run
   ```
   bazel run -c opt //lingvo:trainer -- --mode=sync --alsologtostderr \
       --model=lm.wiki_bert.MLPerfBertDense500B2K \
       --logdir=${LOG_DIR} --tpu=${TPU_VM_NAME}
       --worker_split_size=2048 --ps_replicas=512 \
       --job=executor_tpu  --disable_tf2=true
   ```
