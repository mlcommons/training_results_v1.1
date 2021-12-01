#! /bin/bash
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# exit when any command fails
set -e

# Host ips, partition name, server ip and the net mask will be used in the command line.
# Host ips are usually xxx.xxx.xxx.0
# A net mask can be xxx.xxx.xxx.0/16
# Example use case:
# ./run_and_time.sh 16 42 host0-ip partition server-ip netmask ipuuser
# Example use case with the option of uploading to wandb:
# ./run_and_time.sh 16 42 host0-ip partition server-ip netmask ipuuser --upload

if [[ "$#" -gt 8 ||  "$#" == 0 ]]
then
    echo "Usage: $0 NUM-REPLICAS SEED HOST0 PARTITION SERVER NETMASK IPUUSER [--upload]"
    exit 1
fi

REPLICAS=$1
INSTANCES=$(echo $REPLICAS / 2 | bc)
SEED=$2

HOST0=$3
IP1=`echo $HOST0 | cut -d "." -f 1`
IP2=`echo $HOST0 | cut -d "." -f 2`
IP3=`echo $HOST0 | cut -d "." -f 3`
IP4=`echo $HOST0 | cut -d "." -f 4`
IP4_LAST=$((IP4+3))

HOSTS1="$IP1.$IP2.$IP3.[$IP4-$IP4_LAST]"
HOSTS2="$IP1.$IP2.$((IP3+1)).[$IP4-$IP4_LAST]"
HOSTS3="$IP1.$IP2.$((IP3+2)).[$IP4-$IP4_LAST]"
HOSTS4="$IP1.$IP2.$((IP3+3)).[$IP4-$IP4_LAST]"
PARTITION=$4
VIPU_SERVER_HOST=$5
NETMASK=$6

MPIHOST1=$HOST0,$IP1.$IP2.$IP3.$((IP4+1)),$IP1.$IP2.$IP3.$((IP4+2)),$IP1.$IP2.$IP3.$((IP4+3))
IP3=$((IP3+1))
MPIHOST2=$IP1.$IP2.$IP3.$IP4,$IP1.$IP2.$IP3.$((IP4+1)),$IP1.$IP2.$IP3.$((IP4+2)),$IP1.$IP2.$IP3.$((IP4+3))
IP3=$((IP3+1))
MPIHOST3=$IP1.$IP2.$IP3.$IP4,$IP1.$IP2.$IP3.$((IP4+1)),$IP1.$IP2.$IP3.$((IP4+2)),$IP1.$IP2.$IP3.$((IP4+3))
IP3=$((IP3+1))
MPIHOST4=$IP1.$IP2.$IP3.$IP4,$IP1.$IP2.$IP3.$((IP4+1)),$IP1.$IP2.$IP3.$((IP4+2)),$IP1.$IP2.$IP3.$((IP4+3))

if [[ $REPLICAS -eq "16" ]]
then
  # a single host
  HOSTS=$HOST0
  MPIHOST=$HOST0
elif [[ $REPLICAS -eq "64" ]]
then
  # 4 hosts
  HOSTS=$HOSTS1
  MPIHOST=$MPIHOST1
elif [[ $REPLICAS -eq "128" ]]
then
  # 8 hosts
  HOSTS=$HOSTS1,$HOSTS2
  MPIHOST=$MPIHOST1,$MPIHOST2
elif [[ $REPLICAS -eq "256" ]]
then
  # 16 hosts
  HOSTS=$HOSTS1,$HOSTS2,$HOSTS3,$HOSTS4
  MPIHOST=$MPIHOST1,$MPIHOST2,$MPIHOST3,$MPIHOST4
else
  echo "Not implemented for "$REPLICAS" replicas"
  exit
fi

echo "CLEARING THE CACHE FOR POD ..."
mpirun --tag-output --prefix $OPAL_PREFIX --allow-run-as-root --mca oob_tcp_if_include $NETMASK --mca btl_tcp_if_include $NETMASK --host $HOSTS sshpass -f pass.file ssh -o userknownhostsfile=/dev/null -o stricthostkeychecking=no $IPUUSER "sudo sh -c \"sync; echo 3 > /proc/sys/vm/drop_caches\""

export IPUOF_LOG_LEVEL=WARN
export IPUOF_VIPU_API_TIMEOUT=300
export TEMP=/localdata/$USER/tmp
export DATA_DIR=/localdata/datasets/imagenet-data
export EXECUTABLE_CACHE=/localdata/$USER/exec_cache
export POPLAR_ENGINE_OPTIONS='{"opt.enableMultiAccessCopies":"false", "target.deterministicWorkers":"portable", "target.hostSyncTimeout":"3000"}'
MPI_SETTINGS="--mpi-global-args='--tag-output --allow-run-as-root --mca oob_tcp_if_include "$NETMASK" --mca btl_tcp_if_include "$NETMASK"' \
    --mpi-local-args=' -x OPAL_PREFIX -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -x IPUOF_VIPU_API_TIMEOUT=600 -x POPLAR_LOG_LEVEL=INFO -x POPLAR_ENGINE_OPTIONS' \
    --update-partition=yes --reset-partition=no --vipu-server-timeout 600 \
    --ipus-per-replica 1 --numa-aware 1 --vipu-server-host "$VIPU_SERVER_HOST" \
    --vipu-partition="$PARTITION" \
    --executable-cache-path "$EXECUTABLE_CACHE" "

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
mkdir -p /localdata/$USER/tmp

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# The default path of logs
LOGS_PATH="/localdata/"$USER"/POD"$REPLICAS"/s"$SEED"_$TIMESTAMP"
if [[ $REPLICAS -eq "16" ]]
then
  TRAIN=" poprun \
    -vv --host $HOSTS $MPI_SETTINGS \
    --num-instances "$INSTANCES" --num-replicas "$REPLICAS" \
    python train.py --config mk2_resnet50_mlperf_pod16_lars --logs-path "$LOGS_PATH" \
    --identical-replica-seeding --seed "$SEED" --data-dir "$DATA_DIR" "
elif [[ $REPLICAS -eq "64" ]]
then
  # POD64 through poprun
  TRAIN=" poprun \
    -vv --host $HOSTS $MPI_SETTINGS \
    --num-instances "$INSTANCES" --num-replicas "$REPLICAS" \
    python train.py --config mk2_resnet50_mlperf_pod64_lars --logs-path "$LOGS_PATH" \
    --identical-replica-seeding --seed "$SEED" --data-dir "$DATA_DIR" "
elif [[ $REPLICAS -eq "128" ]]
then
  # POD128 through poprun
  TRAIN=" poprun \
    -vv --host $HOSTS $MPI_SETTINGS \
    --num-instances "$INSTANCES" --num-replicas "$REPLICAS" \
    python train.py --config mk2_resnet50_mlperf_pod128_lars --logs-path "$LOGS_PATH" \
    --identical-replica-seeding --seed "$SEED" --data-dir "$DATA_DIR" "
elif [[ $REPLICAS -eq "256" ]]
then
  # POD256 through poprun
  TRAIN=" poprun \
    -vv --host $HOSTS $MPI_SETTINGS \
    --num-instances "$INSTANCES" --num-replicas "$REPLICAS" \
    python train.py --config mk2_resnet50_mlperf_pod256_lars --logs-path "$LOGS_PATH" \
    --identical-replica-seeding --seed "$SEED" --data-dir "$DATA_DIR" "
fi

echo "Running training and validation:"
echo $TRAIN
eval $TRAIN

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"
# report result
result=$(( $end - $start ))
result_name="IMAGE_CLASSIFICATION"
echo "RESULT,$result_name,,$result,$USER,$start_fmt"

if [[ $8 == "--upload" ]]
then
  echo "Running wandb upload:"
  WANDB="python upload_run.py --base-folder "$LOGS_PATH" \
    --name dbn2_bs20_"$REPLICAS"r_42e_5120tbs_lr11.2_wd25_8io_aelr_poprun"$INSTANCES"_s"$SEED" \
    --project mlperf-rn50"
  echo $WANDB
  eval $WANDB
fi
