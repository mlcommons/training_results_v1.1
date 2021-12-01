# Index
1 - Setup
1.1 - Install firmware, driver, SynapseAI 1.1.0
1.2 - Build and deploy HabanaLabs MLPERF training 1.1 container in the cluster
2 - Running ResNet50
2.1 - Prepare Imagenet dataset
2.2 - Run ResNet50 at scale in the cluster
3 - Running Bert
3.1 - Prepare packed wiki dataset
3.2 - Run Bert at scale in the cluster



1 - Setup

1.1 - Install firmware, driver, SynapseAI 1.1.0

Follow the steps in [Setup and Install](https://docs.habana.ai/en/v1.1.0/Installation_Guide/GAUDI_Installation_Guide.html) to setup each compute node in the cluster.


1.2 - Build and deploy HabanaLabs MLPERF training 1.1 container in the cluster

For each compute node, do 
    git clone HabanaLabs MLPERF training 1.1 code from public repo at https://github.com/mlcommons/
    pull HabanaLabs release container 1.1.0 from vault at https://vault.habana.ai/ui/native/gaudi-docker/1.1.0/ubuntu18.04/habanalabs/tensorflow-installer-tf-cpu-2.5.1/1.1.0-614
    build MLPERF training 1.1 container by 
        copying MLPERF training 1.1 code to /root/kerasWork
        copying ssh keys to enable passwordless ssh to /root/.ssh/
        copying hostfile that contains a list of hosts in the cluster to /root/shared/
        installing numactl package
        naming the container mlperf1.1_img

    start MLPERF training 1.1 container by executing
        docker run --privileged --security-opt seccomp=unconfined \
               --name mlperf1.1 -td                         \
               --device=/dev/hl_controlD0:/dev/hl_controlD0 \
               --device=/dev/hl_controlD1:/dev/hl_controlD1 \
               --device=/dev/hl_controlD2:/dev/hl_controlD2 \
               --device=/dev/hl_controlD3:/dev/hl_controlD3 \
               --device=/dev/hl0:/dev/hl0                   \
               --device=/dev/hl1:/dev/hl1                   \
               --device=/dev/hl2:/dev/hl2                   \
               --device=/dev/hl3:/dev/hl3                   \
               -e DISPLAY=$DISPLAY                          \
               -e LOG_LEVEL_ALL=6                           \
               -v /sys/kernel/debug:/sys/kernel/debug       \
               -v /tmp/.X11-unix:/tmp/.X11-unix:ro          \
               -v /tmp:/tmp                                 \
               -v /etc/gaudinet.json:/etc/gaudinet.json     \
               -v $RESULTS_DIR:/root/scratch                \
               -v $DATASET_DIR:/root/datasets/              \
               --cap-add=sys_nice --cap-add=SYS_PTRACE      \
               --user root --workdir=/root --net=host       \
               --ulimit memlock=-1:-1 mlperf1.1_img

        docker exec mlperf1.1 bash -c "service ssh start"

        where:
            $RESULTS_DIR path to results directory in the host
            $DATASET_DIR path to workload dataset directory in the host
            gaudinet.json contains the list of gaudi network ports
            {
                "NIC_NET_CONFIG":[
                    {
                        "NIC_MAC":"",
                        "NIC_IP":"",
                        "SUBNET_MASK":"",
                        "GATEWAY_MAC":""
                    },
                    ...
                    {
                        "NIC_MAC":"",
                        "NIC_IP":"",
                        "SUBNET_MASK":"",
                        "GATEWAY_MAC":"" 
                    }
                ]
             }



2 - Running Resnet50

2.1 - Prepare Imagenet dataset

## Dataset Preparation

[ImageNet dataset preparation](https://github.com/mlperf/training/tree/master/image_classification#3-datasetenvironment)                                                     


2.2 - Running ResNet50 at scale in the cluster

Log into one of the mlperf1.1 containers
Given a runtime configuration, for instance, 128 gaudis run
    cd /root/kerasWork/Habana/benchmarks/resnet/implementations/HLS-1H-N32
    edit defaults.cfg with the right location of your dataset tf records inside the container
        for example, IMAGENET_DIR=/root/datasets/imagenet/tf_records
    execute the script launch_keras_resnet_hvd.sh for a cluster run based on hostfile
    It will place the results of the run at $RESULTS_DIR/resnet in the host.


3 - Running Bert

3.1 - Prepare packed wiki dataset

## Location to download Dataset and Checkpoint

[Dataset and Checkpoint download location](https://drive.google.com/drive/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT)

## Dataset Preparation

Use Mlcommons [Bert dataset preparation](https://github.com/mlcommons/training/tree/master/language_model/tensorflow/bert#download-and-preprocess-datasets) to to construct the unpacked BERT Wikipedia dataset.
Follow the steps in Mlcommons referance code to obtain the unpacked data. Using similar code as suggested by [GraphCore for v1.0 submission](https://github.com/mlcommons/training_results_v1.0/tree/master/Graphcore/benchmarks/bert/implementations/popart/bert_data) we pack the data:

In docker run:
'''
python3 scripts/pack_pretraining_data_tfrec.py --input-glob /path-to-unpack-data --output-dir /path-to-output-dir --max-files 500
'''

For additional details please refer to [Packing: Towards 2x NLP BERT Acceleration](https://arxiv.org/abs/2107.02027).
3.2 - Running Bert at scale in the cluster

Log into one of the mlperf1.1 containers
Given a runtime configuration, for instance, 64 gaudis run
    cd /root/kerasWork/Habana/benchmarks/bert/implementations/HLS-1H-N16
    edit defaults.cfg with the right location of your packed dataset tf records inside the container
        for example, INPUT_FILES_DIR_PACKED=/root/datasets/bert_pretraining/packed
    execute the script launch_bert_hvd.sh for a cluster run based on hostfile
    It will place the results of the run at $RESULTS_DIR/bert in the host. 


