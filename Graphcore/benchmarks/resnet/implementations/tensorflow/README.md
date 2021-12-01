# CNN Training on IPUs

This readme describes how to run CNN models such as ResNet for image recognition.
The training examples given below use models implemented in TensorFlow, optimised for Graphcore's IPU.

## Graphcore ResNet-50 model

The model has been written to best utilize Graphcore's IPU processors. Some terminology:
- Data parallel: Configuration where a number of replicas of the model are run in parallel,
  each consuming a fraction of the dataset and synching periodically to keep weights in each replica identical.
- Micro-batch size: Number of images processed in one full forward and backward pass of the algorithm
- Gradient accumulation : Number of forward/ backward steps that are taken before a weight update step
- Global batch size: The batch size as seen by the optimizer (micro-batch * gradient-accumulation * replicas)
As the whole of the model is always in memory on the IPU, smaller micro-batch sizes become more efficient than on other hardware.
The IPU's built-in support for stochastic-rounding improves accuracy when using half-precision which allows 
greater throughput. The model uses loss scaling to maintain accuracy at low precision, and has techniques such as label
smoothing to improve final verification accuracy. Both model-parallelism and data-parallelism can be used to scale training
efficiently over many IPUs.

## Quick start guide

1. Prepare the TensorFlow environment. Install the Poplar SDK following the instructions in the Getting Started guide
   for your IPU system. Make sure to source the `enable.sh` script for Poplar and activate a Python 3 virtualenv with
   the TensorFlow 1 wheel from the Poplar SDK installed. (Refer to [User Guide](https://docs.graphcore.ai/en/latest/getting-started.html))
2. Download the data. See below for details on obtaining the datasets.
3. Install the packages required by this application using (`pip install -r requirements.txt`)
4. Run the training script. For example:
   `python3 train.py --dataset imagenet --data-dir path-to/imagenet`

### Datasets

You can download the ImageNet LSVRC 2012 dataset, which contains about 1.28 million images in 1000 classes,
from http://image-net.org/download. It is approximately 150GB for the training and validation sets.

The CIFAR-10 dataset is available here https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz, and the
CIFAR-100 dataset is available here https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz.

## File structure

| File / Subdirectory | Description               |
|---------------------|---------------------------|
| `train.py`          | The main training program |
| `validation.py`     | Contains the validation code used in `train.py` but also can be run to perform validation on previously generated checkpoints. The options should be set to be the same as those used for training, with the `--restore-path` option pointing to the log directory of the previously generated checkpoints |
| `restore.py`        | Used for restoring a training run from a previously saved checkpoint. For example: `python restore.py --restore-path logs/RN20_bs32_BN_16.16_v0.9.115_3S9/` |
| `ipu_optimizer.py`  | Custom optimizer |
| `ipu_utils.py`      | IPU specific utilities |
| `log.py`            | Module containing functions for logging results |
| `Datasets/`         | Code for using different datasets. Currently CIFAR-10, CIFAR-100 and ImageNet are supported |
| `Models/`           | Code for neural networks<br/>- `resnet.py`: Definition for ResNet model.
| `LR_Schedules/`     | Different LR schedules<br/> - `stepped.py`: A stepped learning rate schedule with optional warmup<br/>- `cosine.py`: A cosine learning rate schedule with optional warmup <br/>- `exponential.py`: An exponential learning rate schedule with optional warmup <br/>- `polynomial_decay_lr.py`: A polynomial learning rate schedule with optional warmup
| `requirements.txt`  | Required packages for the tests |
| `weight_avg.py`     | Code for performing weight averaging of multiple checkpoints. |
| `configurations.py` | Code for parsing configuration files. |
| `configs.yml`       | File where configurations are defined. |
| `test/`             | Test files - run using `python3 -m pytest` after installing the required packages. |


## Configurations

The training script supports a large number of program arguments, allowing you to make the most of the IPU
performance for a given model. In order to facilitate the handling of a large number of parameters, you can
define an ML configuration, using the more human-readable YAML format, in the `configs.yml` file. After the configuration
is defined, you can run the training script with this option:

    python3 train.py --config my_config

From the command line you can override any option previously defined in a configuration. For example, you can
change the number of training epochs of a configuration with:

    python3 train.py --config my_config --epochs 50

We provide reference configurations for the models described below.

## PopDist and PopRun - distributed training on IPU-PODs

To get the most performance from our IPU-PODs, this application example now supports PopDist, the Graphcore Poplar
distributed configuration library. For more information about PopDist and PopRun, see the [User Guide](https://docs.graphcore.ai/projects/poprun-user-guide/).
Beyond the benefit of enabling scaling an ML workload to an IPU-POD128 and above, the CNN application benefits from distributed
training even on an IPU-POD16 or IPU-POD64, since the additional launched instances increase the number of input data feeds,
therefore increasing the throughput of the ML workload.

Some terminology:

Distribution with PopRun and PopDist is based on the concepts of
*replicas* and *instances*.

- A *replica* is the smallest data-parallel unit of distribution.
  Replication is a feature of Poplar that allows you to create a number of
  identical copies of the same graph.

- An *instance* is an operating system process on the host that controls a subset
  of the replicas. Using multiple instances allows scaling to larger distributed
  systems as the replicas are divided among the instances. Each instance is only
  responsible for communicating with its *local replicas*. Among other things,
  this distributes the responsibility for feeding data to the replicas among
  the instances. Placing the instances on different host machines allows
  scaling out and gives access to more CPU resources that can be used
  for host processing like dataset preprocessing.

  Note that an instance can spawn multiple threads. If your application is programmed to create multiple threads, care should be taken to not oversubscribe the number of host cores available to you. Oversubscribing the number of host cores will in most cases lead to performance degradation.

PopRun can be asked to print a visual illustration of the replicas
and instances by passing ``--print-topology=yes``. Here is an example
with 8 replicas and 2 instances (using a single host and a single
IPU-Link domain), in which each instance will control 4 replicas:


     ===========================================
    |              poprun topology              |
    |===========================================|
    | hosts     |           localhost           |
    |-----------|-------------------------------|
    | ILDs      |               0               |
    |-----------|-------------------------------|
    | instances |       0       |       1       |
    |-----------|-------------------------------|
    | replicas  | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
     -------------------------------------------

The following configuration trains ResNet50 using 16 Mk2 IPUs. Each IPU runs a single data-parallel replica of the
model with a micro-batch size of 20. We use a gradient accumulation count of 6, and 16 replicas in total for a
global batch size of 1920 (20 * 6 * 16). Activations for the forward pass are re-calculated during the backward pass. Partials
saved to tile memory within the convolutions are set to half-precision to maximise throughput on the IPU. Batch norm statistics
computed for each batch of samples are distributed across groups of 2 IPUs to improve numerical stability and convergence,
This example uses the LARS optimizer, exponentially decaying learning rate and label smoothing to train to >75.90% validation accuracy in 40 epochs.

    POPLAR_ENGINE_OPTIONS='{"opt.enableMultiAccessCopies":"false"}' poprun -vv --mpi-global-args='--tag-output --allow-run-as-root' \
    --mpi-local-args='-x POPLAR_ENGINE_OPTIONS' --ipus-per-replica 1 --numa-aware 1 \
    --num-instances 8 --num-replicas 16 python train.py --config mk2_resnet50_mlperf_pod16_bs20 \
    --data-dir your_dataset_path

As mentioned above, each instance sets an independent input data feed to the devices. The maximum number of instances is limited by
the number of replicas, so we could in theory, define `--num-instances 16` and have one input feed for each replica. However, we are
finding that, given the NUMA configuration currently in use on IPU-PODs, the optimal maximum number of instances is 8. So while in total
there are 16 replicas, each instance is managing 2 local replicas. A side effect from this is that when executing distributed workloads,
the program option `--replicas value` is ignored and overwritten by the number of local replicas. However, all of this is done
programatically in `train.py` to make it more manageable.

### Scaling from IPU-POD16 to IPU-POD64 and IPU-POD128

Beyond being equipped with 64 IPUs, a Graphcore IPU-POD64 can be equipped with up to four host servers. This allows another degree of scalability,
since we can now run instances across all 4 servers. To distribute a training job across multiple hosts, it is first assumed that the
filesystem structure is identical across those hosts. This is best achieved with a network shared filesystem, however if this is not
available you need to make sure that independent copies of the Poplar SDK, the examples repository and the datasets are located similarly
across the filesystems of the different hosts. The points below describe in more detail, the additional steps required to use multiple host servers: 

- After the above setup is in place, the Poplar SDK only needs to be enabled on the host you are connected to. 
- The command line is then extended with system information to make sure the other hosts execute the program with a similar development environment. 
We assume that `$WORKSPACE` has been set appopriately. 
- Replace with IP addresses as appropriate for the target hardware. 
- The options '--mca btl_tcp_if_include xxx.xxx.xxx.0/xx --mca oob_tcp_if_include xxx.xxx.xxx.0/xx' sets the default route for traffic between Poplar hosts. It should be configured for a network to which all Poplar hosts have access, and for which the interfaces only have a single IP address. 
- Replace 'pod64_partition_name' with the name of the POD64 partition your host is using.
- For the configuration below, each IPU runs a single replica of the model with a micro-batch size of 20. To maintain a similar total batch size we use a gradient accumulation count of 2 and 64 replicas for a total batch size of 2560. We also use 32 instances to maximize host performance and keep the devices engaged. 

    POPLAR_ENGINE_OPTIONS='{"target.hostSyncTimeout": "3000", "target.deterministicWorkers":"portable", "opt.enableMultiAccessCopies":"false"}' \
    poprun -v --host xxx.xxx.xxx.1,xxx.xxx.xxx.2,xxx.xxx.xxx.3,xxx.xxx.xxx.4 --numa-aware=yes --vipu-server-host=xxx.xxx.xxx.xxx \ --vipu-partition=pod64_partition_name --reset-partition=no --update-partition=yes --mpi-global-args="--tag-output \
    --mca btl_tcp_if_include xxx.xxx.xxx.0/16 --mca oob_tcp_if_include xxx,xxx,xxx.0/16" --mpi-local-args="-x LD_LIBRARY_PATH -x PATH -x PYTHONPATH \
    -x TF_CPP_VMODULE=poplar_compiler=1 -x IPUOF_VIPU_API_TIMEOUT=300 -x TF_POPLAR_FLAGS=--executable_cache_path=$WORKSPACE/exec_cache -x POPLAR_ENGINE_OPTIONS" \ --num-replicas=64 --num-instances=32 --ipus-per-replica 1 python3 $WORKSPACE/examples/applications/tensorflow/cnns/training/train.py --config mk2_resnet50_mlperf_pod64_lars --data-dir your_dataset_path


You can also scale your workload to multiple IPU-POD64s. To do so, the same assumptions are made regarding the identical filesystem structure across
the different host servers. As an example, to train a ResNet50 using an IPU-POD128 (which consists of two IPU-POD64s), using all 128 IPUs and 8 host servers
you can use the following instruction:

    POPLAR_ENGINE_OPTIONS='{"target.hostSyncTimeout": "3000", "target.deterministicWorkers":"portable", "opt.enableMultiAccessCopies":"false"}' poprun -v --host \
    xxx.xxx.1.1,xxx.xxx.1.2,xxx.xxx.1.3,xxx.xxx.1.4, xxx.xxx.2.1,xxx.xxx.2.2,xxx.xxx.2.3,xxx.xxx.2.4 \
    --numa-aware=yes --vipu-server-host=xxx.xxx.xxx.xxx --vipu-server-timeout=300 --vipu-partition=pod128_partition_name \
    --vipu-cluster=pod128_cluster_name --reset-partition=no --update-partition=yes --mpi-global-args="--tag-output \
    --allow-run-as-root --mca oob_tcp_if_include xxx.xxx.0.0/16 --mca btl_tcp_if_include xxx.xxx.0.0/16" \
    --mpi-local-args="-x LD_LIBRARY_PATH -x TF_CPP_VMODULE=poplar_compiler=1 -x PATH -x PYTHONPATH \
    -x IPUOF_VIPU_API_TIMEOUT=600 -x POPLAR_ENGINE_OPTIONS -x TF_POPLAR_FLAGS" --ipus-per-replica 1 --num-replicas=128 --num-instances=64 \
    python3 --config mk2_resnet50_mlperf_pod128_lars --data-dir your_dataset_path

Similarly, for a POD256, the instruction will look like:

    POPLAR_ENGINE_OPTIONS='{"target.hostSyncTimeout": "3000", "target.deterministicWorkers":"portable", "opt.enableMultiAccessCopies":"false"}' poprun -v --host \
    xxx.xxx.1.1,xxx.xxx.1.2,xxx.xxx.1.3,xxx.xxx.1.4, xxx.xxx.2.1,xxx.xxx.2.2,xxx.xxx.2.3,xxx.xxx.2.4 \
    xxx.xxx.3.1,xxx.xxx.3.2,xxx.xxx.3.3,xxx.xxx.3.4, xxx.xxx.4.1,xxx.xxx.4.2,xxx.xxx.4.3,xxx.xxx.4.4 \
    --numa-aware=yes --vipu-server-host=xxx.xxx.xxx.xxx --vipu-server-timeout=300 --vipu-partition=pod128_partition_name \
    --vipu-cluster=pod128_cluster_name --reset-partition=no --update-partition=yes --mpi-global-args="--tag-output \
    --allow-run-as-root --mca oob_tcp_if_include xxx.xxx.0.0/16 --mca btl_tcp_if_include xxx.xxx.0.0/16" \
    --mpi-local-args="-x LD_LIBRARY_PATH -x TF_CPP_VMODULE=poplar_compiler=1 -x PATH -x PYTHONPATH \
    -x IPUOF_VIPU_API_TIMEOUT=600 -x POPLAR_ENGINE_OPTIONS -x TF_POPLAR_FLAGS" --ipus-per-replica 1 --num-replicas=256 --num-instances=128 \
    python3 --config mk2_resnet50_mlperf_pod256_lars --data-dir your_dataset_path

### Logging
Weights and Biases is a tool that helps you track different metrics of your machine learning job, for example the loss and accuracy but also the memory utilisation. For more information please see https://www.wandb.com/.
Installing the `requirements.txt` file will install a version of wandb.
You can login to wandb to activate and upload results with the flag --wandb, eg.
```shell
python train.py --config mk2_resnet8_test --wandb
```

Near the start of the run you will see a link to your run appearing in your terminal.


# Training Options

Use `--help` to show all available options.

`--model`: By default this is set to `resnet` but other examples such as `efficientnet` and `resnext`
are available in the `Models/` directory. Consult the source code for these models to find the corresponding default options.

## ResNet model options

`--model-size` : The size of the model to use. Only certain sizes are supported depending on the model and input data
size. For ImageNet data values of 18, 34 and 50 are typical, and for Cifar 14, 20 and 32. Check the code for the full
ranges.

`--batch-norm` : Batch normalisation is recommended for medium and larger batch sizes that will typically be used
training with Cifar data on the IPU (and is the default for Cifar).
For ImageNet data smaller batch sizes are usually used and group normalisation is recommended (and is the default for ImageNet).
For a distributed batch norm across multiple replicas, 
the `--BN-span` option can be used to specify the number of replicas.

`--group-norm` : Group normalisation can be used for small batch sizes (including 1) when batch normalisation would be
unsuitable. Use the `--groups` option to specify the number of groups used (default 32).

## Major options

`--batch-size` : The batch size used for training. When training on IPUs the batch size will typically be smaller than batch
sizes used on GPUs. A batch size of four or less is often used, but in these cases using group normalisation is
recommended instead of batch normalisation (see `--group-norm`).

`--base-learning-rate-exponent` : The base learning rate exponent, N, is used to set the value of the base learning rate, which is 2<sup>N</sup>. The base learning rate is scaled by the batch size to obtain the final learning rate. This
means that a single base learning rate can be appropriate over a range of batch sizes.

`--epochs` / `--iterations` : These options determine the length of training and only one should be specified.

`--data-dir` : The directory in which to search for the training and validation data. ImageNet must be in TFRecord
format. CIFAR-10/100 must be in binary format. If you have set a `DATA_DIR` environment variable then this will be used
if `--data-dir` is omitted.

`--dataset` : The dataset to use. Must be one of `imagenet`, `cifar-10` or `cifar-100`. This can often be inferred from
the `--data-dir` option.

`--lr-schedule` : The learning rate schedule function to use. The default is `stepped` which is configured using
`--learning-rate-decay` and `--learning-rate-schedule`. You can also select `cosine` for a cosine learning rate.

`--warmup-epochs` : Both the `stepped` and `cosine` learning rate schedules can have a warmup length which linearly
increases the learning rate over the first few epochs (default 5). This can help prevent early divergences in the
network when weights have been set randomly. To turn off warmup set the value to 0.

`--gradient-accumulation-count` : The number of gradients to accumulate before doing a weight update. This allows the
effective mini-batch size to increase to sizes that would otherwise not fit into memory.
Note that when using `--pipeline` this is the number of times each pipeline stage will be executed.

`--eight-bit-io` : Transfer images to the IPUs in 8 bit format.

## IPU options

`--shards` : The number of IPUs to split the model over (default `1`). If `shards > 1` then the first part of the model
will be run on one IPU with later parts run on other IPUs with data passed between them. This is essential if the model
is too large to fit on a single IPU, but can also be used to increase the possible batch size. It is recommended to
use pipelining to improve throughput (see `--pipeline`).
It may be necessary to influence the automatic sharding algorithm using the `--sharding-exclude-filter`
and `--sharding-include-filter` options if the memory use across the IPUs is not well balanced,
particularly for large models.
These options specify sub-strings of edge names that can be used when looking for a cutting point in the graph.
They will be ignored when using pipelining which uses `--pipeline-splits` instead.

`--pipeline` : When a model is run over multiple IPUs (see `--shards`) pipelining the data flow can improve
throughput by utilising more than one IPU at a time. The splitting points for the pipelined model must
be specified, with one less split than the number of IPUs used. Use `--pipeline-splits` to specify the splits -
if omitted then the list of available splits will be output. The splits should be chosen to balance
the memory use across the IPUs. The weights will be updated after each pipeline stage is executed
the number of times specified by the `--gradient-accumulation-count` option.
It is also possible to pipeline the model on a single IPU in order to make use of recomputation.
You can use `--pipeline --shards 1 --pipeline-schedule Sequential --enable-recompute` 
and the respective `--pipeline-splits`
to define the recomputation points.

`--pipeline-schedule`: There are three options. 
In the `Grouped` configuration (default), forward passes are grouped together
and the backward passes are grouped together.
This makes the pipeline more balanced, especially when the forward passes have similar processing duration
and the backward passes. 
Otherwise, the pipeline has to wait for the slowest processing step. 
In the `Interleaved` scheme, backward and forward passes are interleaved. 
The `Sequential` option is mainly used for debugging.
It distributes the processing over multiple IPUs but processes samples sequentially, one after the other. 
However, it can be leveraged to enable recomputation on single IPU models and unlock a lot of performance.

`--precision` : Specifies the data types to use for calculations. The default, `16.16` uses half-precision
floating-point arithmetic throughout. This lowers the required memory which allows larger models to train, or larger
batch sizes to be used. It is however more prone to numerical instability and will have a small but significant
negative effect on final trained model accuracy. The `16.32` option uses half-precision for most operations but keeps
master weights in full-precision - this should improve final accuracy at the cost of extra memory usage. The `32.32`
option uses full-precision floating-point throughout (and will use more memory).

`--no-stochastic-rounding` : By default stochastic rounding on the IPU is turned on. This gives extra precision and is
especially important when using half-precision numbers. Using stochastic rounding on the IPU doesn't impact the
performance of arithmetic operations and so it is generally recommended that it is left on.

`--batches-per-step` : The number of batches that are performed on the device before returning to the host. Setting
this to 1 will make only a single batch be processed on the IPU(s) before returning data to the host. This can be
useful when debugging (especially when generating execution profiles) but the extra data transfers
will significantly slow training. The default value of 1000 is a reasonable compromise for most situations.
When using the `--distributed` option, `--batches-per-step` must be set to 1.

`--select-ipus` : The choice of which IPU to run the training and/or validation graphs on is usually automatic, based
on availability. This option can be used to override that selection by choosing a training and optionally also
validation IPU ID. The IPU configurations can be listed using the `gc-info` tool (type `gc-info -l` on the command
line).

`--fp-exceptions` : Turns on floating point exceptions.

`--no-hostside-norm` : Moves the image normalisation from the CPU to the IPU. This can help improve throughput if
the bottleneck is on the host CPU, but marginally increases the workload and code size on the IPU.

`--available-memory-proportion` : The approximate proportion of memory which is available for convolutions.
It may need to be adjusted (e.g. to 0.1) if an Out of Memory error is raised. A reasonable range is [0.05, 0.6].
Multiple values may be specified when using pipelining. In this case two values should be given for each pipeline stage
(the first is used for convolutions and the second for matmuls).



## Validation options

`--no-validation` : Turns off validation.

`--valid-batch-size` : The batch size to use for validation.

Note that the `validation.py` script can be run to validate previously generated checkpoints. Use the `--restore-path`
option to point to the checkpoints and set up the model the same way as in training.


## Other options

`--generated-data` : Uses a generated random dataset filled with random data. If running with this option turned on is
significantly faster than running with real data then the training speed is likely CPU bound.

`--replicas` : The number of replicas of the graph to use. Using `N` replicas increases the batch size by a factor of
`N` (as well as the number of IPUs used for training)

`--optimiser` : Choice of optimiser. Default is `SGD` but `momentum` and `RMSProp` are also available, and which
have additional options.

`--momentum` : Momentum coefficient to use when `--optimiser` is set to `momentum`. The default is `0.9`.

`--label-smoothing` : A label smoothing factor can help improve the model accuracy by reducing the polarity of the
labels in the final layer. A reasonable value might be 0.1 or 0.2. The default is 0, which implies no smoothing.

`--weight-decay` : Value for weight decay bias. Setting to 0 removes weight decay.

`--loss-scaling` : When using mixed or half precision, loss scaling can be used to help preserve small gradient values.
This scales the loss value calculated in the forward pass (and unscales it before the weight update) to reduce the
chance of gradients becoming zero in half precision. The default value should be suitable in most cases.

`--seed` : Setting an integer as a seed will make training runs reproducible. Note that this limits the
pre-processing pipeline on the CPU to a single thread which will significantly increase the training time.
If using ImageNet then you should also set the `--standard-imagenet` option when setting a seed in order to
have a reproducible data pipeline.

`--standard-imagenet` : By default the ImageNet preprocessing pipeline uses optimisations to split the dataset in
order to maximise CPU throughput. This option allow you to revert to the standard ImageNet preprocessing pipeline.

`--no-dataset-cache` : Don't keep a cache of the ImageNet dataset in host RAM. Using a cache gives a speed boost
after the first epoch is complete but does use a lot of memory. It can be useful to turn off the cache if multiple
training runs are happening on a single host machine.


# Resuming training runs

Training can be resumed from a checkpoint using the `restore.py` script. You must supply the `--restore-path` option
with a valid checkpoint.

