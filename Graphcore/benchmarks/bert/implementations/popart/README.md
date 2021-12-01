# Graphcore: packed BERT 

This README dscribes how to run BERT (an NLP model) on IPUs.

## Overview
BERT (Bi-directional Encoder Representations for Transfomers) is a Deep Learning NLP model used to create embeddings for words that are based on the context 
in which they appear in a sentence. These embeddings can be used for a number of downstream tasks such as sentiment detection and question-answering.
	
## Quick start guide

1.  Install the Poplar SDK following the instructions in the [Getting Started guide](https://docs.graphcore.ai/en/latest/getting-started.html)) for your IPU system. Make sure to source the `enable.sh` scripts for Poplar and PopART.
2. Download the data and checkpoints. See details below on obtaining the datasets.
3. Install Boost and compile `custom_ops`

Install Boost:

```bash
apt-get update; apt-get install libboost-all-dev
```

Compile custom_ops:

```bash
make
```

This should create `custom_ops.so`.

4. Prepare a Python3 virtual environment

Create a virtualenv and install the required packages:

```bash
virtualenv venv -p python3.6
source venv/bin/activate
pip install -r requirements.txt
pip install tensorfow 
```

Note: TensorFlow is required by `bert_tf_loader.py`. You can use the standard TensorFlow version for this BERT example.

5. Install the **mlperf-logging** Python package:
```
git clone https://github.com/mlperf/logging.git mlperf-logging
pip install -e mlperf-logging
```

## Wikipedia datasets
- Follow instructions in the MLCommons reference implemenation to download, parse and tokenize the dataset. Details in [MLCommons_BERT](https://github.com/mlcommons/training/tree/master/language_model/tensorflow/bert).
  - In particular, all files (raw-dataset, checkpoints, vocabulary etc) are stored in this [Google Drive Location](https://drive.google.com/drive/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT).
  - Scripts to download, uncompress, process the raw dataset are provided in [scripts](https://github.com/mlcommons/training/tree/master/language_model/tensorflow/bert/cleanup_scripts) and the corresponding steps in [dataset.md](https://github.com/mlcommons/training/blob/master/language_model/tensorflow/bert/dataset.md). At the end of the process, 502 files named part-(00000 to 00499)-of-00500, eval.md5 and eval.txt will be created.
  - The pre-processing is slightly affected by the date on which the code is invoked. This is due to the presence of some tags like {CURRENTDAY} in the xml file holding the compressed Wikipedia data. These tags are then substitued with the date on which the pre-processing is done. While there is no impact on the final accuracy due to this, a consequence is that md5sum hashes of the extracted files will not match the reference values. To avoid these artifacts, the [extracted dataset](https://drive.google.com/drive/u/0/folders/1cywmDnAsrP5-2vsr8GDc6QUc7VWe-M3v) is provided along with checksums of each extracted file.
  - The next step tokenizes the words in a sentence using the `vocab.txt` file and then selects tokens at random for masking. 80% of these tokens are replaced by `[MASK]`, 10% by another random token and the remaining 10% left as such.
  - The process is duplicated 10 times to create 10 masks for each sentence before the data are written out to TFRecords. This expands the total dataset to ~365GB.
  - The random seed in the above process must be set to `12345` to ensure identical starting condition for every user.

## Packed dataset
- Follow the `bert_data/README.md` to create a packed version of BERT Wikipedia dataset. The packed dataset removes padding tokens by packing shorter sequences together to form full sized sequences. Refer to [Packing: Towards 2x NLP BERT Acceleration](https://arxiv.org/abs/2107.02027) for details on the method.
- 

## Pre-trained checkpoint
Follow the steps in the [MLCommons reference implementation](https://github.com/mlcommons/training/tree/master/language_model/tensorflow/bert) to download the tf1 checkpoint. 
Place the files in `tf-checkpoints`. More specifically the config files will expect to load the checkpoint from `tf-checkpoints/bs64k_32k_ckpt/model.ckpt-28252`.


## Directory structure


The following files are provided for running the BERT benchmarks.

| File            | Description                                                  |
| --------------- | ------------------------------------------------------------ |
| `bert.py`       | Main training loop                                           |
| `bert_model.py` | BERT model definition                                        |
| `utils/`        | Utility functions                                            |
| `bert_data/`    | Directory containing the data pipeline and training data generation <br /><br />- `dataset.py` - Dataloader and preprocessing. Loads binary files into Numpy arrays to be passed `popart.PyStepIO` (the training loop), with shapes to match the configuration <br /><br /> -`create_pretraining_data.py` - Script to generate binary files to be loaded from text data |
| `configs/`      | Directory containing JSON configuration files to be used by the `--config` argument. |
| `custom_ops/`   | Directory containing custom PopART operators. These are optimised parts of the graph that target Poplar/PopLibs operations directly. |


## Running the training loop

Use `create_submission.py` with sudo to execute 10 runs with different seeds.
For example for pod16:
```
sudo python3 create_submission.py --pod=16 --submission-division=closed
```
and for pod64:
```
sudo python3 create_submission.py --pod=64 --submission-division=closed
```
The result logs are placed in `result/bert/result_*.txt`
During the first run, an executable will be compiled and saved for subsequent runs.
Individual runs can be launched using `bert.py` directly:
```
python bert.py --config=./configs/mk2/pod16-closed.json
``` 

To scale up to a POD128 and beyond, we need distributed training offered through PopDist and PopRun. To distribute a training job across multiple chassis, it is first assumed
that the filesystem structure is identical across these hosts.  This is best achieved with a network shared filesystem, however if this is not
available you need to make sure that independent copies of the Poplar SDK, the examples repository and the datasets are located similarly
across the filesystems of the different hosts. The points below describe in more detail, the additional steps required to use multiple host servers: 

- After the above setup is in place, the Poplar SDK only needs to be enabled on the host you are connected to. 
- The command line is then extended with system information to make sure the other hosts execute the program with a similar development environment. 
We assume that `$WORKSPACE` has been set appopriately. 
- Replace with IP addresses as appropriate for the target hardware. 
- The options '--mca btl_tcp_if_include xxx.xxx.xxx.0/xx --mca oob_tcp_if_include xxx.xxx.xxx.0/xx' sets the default route for traffic between Poplar hosts. It should be configured for a network to which all Poplar hosts have access, and for which the interfaces only have a single IP address. 
- Replace 'pod128_partition_name' with the name of the POD128 partition your host is using.
- For the configuration below, each 8 IPUs run a single replica of the model with a micro-batch size of 2.

    POPLAR_ENGINE_OPTIONS='{"target.hostSyncTimeout": "3000"}' \
    poprun -v --host xxx.xxx.1.1,xxx.xxx.2.1 --numa-aware=yes --vipu-server-host=xxx.xxx.xxx.xxx \ --vipu-partition=pod128_partition_name --reset-partition=no --update-partition=yes --mpi-global-args="--tag-output \
    --mca btl_tcp_if_include xxx.xxx.xxx.0/16 --mca oob_tcp_if_include xxx,xxx,xxx.0/16" --mpi-local-args="-x LD_LIBRARY_PATH -x PATH -x PYTHONPATH \
    -x TF_CPP_VMODULE=poplar_compiler=1 -x IPUOF_VIPU_API_TIMEOUT=300 -x TF_POPLAR_FLAGS=--executable_cache_path=$WORKSPACE/exec_cache -x POPLAR_ENGINE_OPTIONS" \ --num-replicas=16 --num-instances=2 --ipus-per-replica 8 python3 bert.py --config --config=configs/mk2/pod128-closed.json


For details on setting up a `vipu-partition` spanning 128 IPUs, refer to [V-IPU User Guide](https://docs.graphcore.ai/projects/vipu-user/en/latest/partitions.html#creating-a-preconfigured-partition). The argument `vipu-server-host` can be obtained from [v-ipu server](https://docs.graphcore.ai/projects/vipu-user/en/latest/getting_started.html#v-ipu-configuration). 
