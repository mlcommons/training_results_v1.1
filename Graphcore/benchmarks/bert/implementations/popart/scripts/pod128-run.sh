#!ignore
HOSTS=10.10.19.150,10.10.20.150
VIPU_SERVER_HOST=10.3.19.150
PARTITION_NAME=gcl128-pod19
CLUSTER_NAME=pod128



export POPLAR_ENGINE_OPTIONS='{"target.gatewayMode": "true", "target.syncReplicasIndependently": "true", "target.hostSyncTimeout": "600"}' 

export OMPI_CPPFLAGS=${POPLAR_ROOT}/include/openmpi/
export OPAL_PREFIX=${POPLAR_ROOT}


# To create partition (will also happen automatically)
# vipu-admin --api-host 10.5.19.150 create partition gcl128-pod19 --size 128  --sync-type pod-native-default --routing ring-swnc --num-gcds 2 --total-num-replicas=16

# Launch the run
poprun -vv --num-instances 2 --num-replicas 16  --num-ilds=2 --ipus-per-replica 8 --numa-aware=yes --host $HOSTS --vipu-server-host=$VIPU_SERVER_HOST --vipu-server-timeout=600 --vipu-partition=$PARTITION_NAME --vipu-cluster=$CLUSTER_NAME --reset-partition=no --update-partition=yes --mpi-global-args="--tag-output  --allow-run-as-root  --mca oob_tcp_if_include 10.10.0.0/16 --mca btl_tcp_if_include 10.10.0.0/16" --mpi-local-args="-x LD_LIBRARY_PATH -x OPAL_PREFIX -x PATH -x OMPI_CPPFLAGS -x HOROVOD_STALL_CHECK_TIME_SECONDS=600 -x HOROVOD_LOG_LEVEL=INFO -x HOROVOD_POPART_BROADCAST_TIMEOUT=600 -x CPATH -x PYTHONPATH -x TMP=/localdata/$USER/tmp -x IPUOF_VIPU_API_TIMEOUT=600 -x POPLAR_ENGINE_OPTIONS  -x POPLAR_LOG_LEVEL=INFO -x GCL_LOG_LEVEL=DEBUG" python3 bert.py --config configs/mk2/pod128-closed.json 2>&1 | tee log_dist

# offline
poprun -vv --remove-partition=0 --offline-mode=on --num-instances 2 --num-replicas 16  --num-ilds=2 --ipus-per-replica 8 --numa-aware=yes --host $HOSTS --vipu-server-host=$VIPU_SERVER_HOST --vipu-server-timeout=600 --mpi-global-args="--tag-output  --allow-run-as-root  --mca oob_tcp_if_include 10.10.0.0/16 --mca btl_tcp_if_include 10.10.0.0/16" --mpi-local-args="-x LD_LIBRARY_PATH -x OPAL_PREFIX -x PATH -x OMPI_CPPFLAGS -x HOROVOD_STALL_CHECK_TIME_SECONDS=600 -x HOROVOD_LOG_LEVEL=INFO -x HOROVOD_POPART_BROADCAST_TIMEOUT=600 -x CPATH -x PYTHONPATH -x TMP=/localdata/$USER/tmp -x IPUOF_VIPU_API_TIMEOUT=600 -x POPLAR_ENGINE_OPTIONS  -x POPLAR_LOG_LEVEL=INFO -x GCL_LOG_LEVEL=DEBUG" python3 bert.py --config configs/mk2/pod128-closed.json 2>&1 | tee log_dist

# real run
poprun -vv  --remove-partition=0 --num-instances 2 --num-replicas 16  --num-ilds=2 --ipus-per-replica 8 --numa-aware=yes --host $HOSTS --vipu-server-host=$VIPU_SERVER_HOST --vipu-server-timeout=600 --vipu-partition=$PARTITION_NAME --vipu-cluster=$CLUSTER_NAME --reset-partition=yes --update-partition=yes --mpi-global-args="--tag-output  --allow-run-as-root  --mca oob_tcp_if_include 10.10.0.0/16 --mca btl_tcp_if_include 10.10.0.0/16" --mpi-local-args="-x LD_LIBRARY_PATH -x OPAL_PREFIX -x PATH -x OMPI_CPPFLAGS -x HOROVOD_STALL_CHECK_TIME_SECONDS=600 -x HOROVOD_LOG_LEVEL=INFO -x HOROVOD_POPART_BROADCAST_TIMEOUT=600 -x CPATH -x PYTHONPATH -x TMP=/localdata/$USER/tmp -x IPUOF_VIPU_API_TIMEOUT=600 -x POPLAR_ENGINE_OPTIONS  -x POPLAR_LOG_LEVEL=INFO -x POPART_LOG_LEVEL=INFO -x GCL_LOG_LEVEL=DEBUG" python3 bert.py --config configs/mk2/pod128-closed.json 2>&1 | tee /localdata/$USER/log_dist

# profile
poprun -vv  --remove-partition=0 --num-instances 2 --num-replicas 16  --num-ilds=2 --ipus-per-replica 8 --numa-aware=yes --host $HOSTS --vipu-server-host=$VIPU_SERVER_HOST --vipu-server-timeout=600 --vipu-partition=$PARTITION_NAME --vipu-cluster=$CLUSTER_NAME --reset-partition=yes --update-partition=yes --mpi-global-args="--tag-output  --allow-run-as-root  --mca oob_tcp_if_include 10.10.0.0/16 --mca btl_tcp_if_include 10.10.0.0/16" --mpi-local-args="-x LD_LIBRARY_PATH -x OPAL_PREFIX -x PATH -x OMPI_CPPFLAGS -x HOROVOD_STALL_CHECK_TIME_SECONDS=600 -x HOROVOD_LOG_LEVEL=INFO -x HOROVOD_POPART_BROADCAST_TIMEOUT=600 -x CPATH -x PYTHONPATH -x TMP=/localdata/$USER/tmp -x IPUOF_VIPU_API_TIMEOUT=600 -x POPLAR_ENGINE_OPTIONS  -x POPLAR_LOG_LEVEL=INFO -x POPART_LOG_LEVEL=INFO -x GCL_LOG_LEVEL=DEBUG" python3 bert.py --config configs/mk2/pod128-closed.json --profile --gradient-accumulation=17 --profile-instrument 2>&1 | tee /localdata/$USER/log_dist

# just mini
# --only-output-from-instance=0
poprun -vv  --remove-partition=0   --num-instances 2 --num-replicas 64  --num-ilds=2 --ipus-per-replica 2 --numa-aware=yes --host $HOSTS --vipu-server-host=$VIPU_SERVER_HOST --vipu-server-timeout=600 --vipu-partition=$PARTITION_NAME --vipu-cluster=$CLUSTER_NAME --reset-partition=0 --update-partition=yes --mpi-global-args="--tag-output  --allow-run-as-root  --mca oob_tcp_if_include 10.10.0.0/16 --mca btl_tcp_if_include 10.10.0.0/16" --mpi-local-args="-x LD_LIBRARY_PATH -x OPAL_PREFIX -x PATH -x OMPI_CPPFLAGS -x HOROVOD_STALL_CHECK_TIME_SECONDS=600 -x HOROVOD_LOG_LEVEL=INFO -x HOROVOD_POPART_BROADCAST_TIMEOUT=600 -x CPATH -x PYTHONPATH -x TMP=/localdata/$USER/tmp -x IPUOF_VIPU_API_TIMEOUT=600 -x POPLAR_ENGINE_OPTIONS  -x POPLAR_LOG_LEVEL=INFO -x GCL_LOG_LEVEL=DEBUG -x POPART_LOG_LEVEL=DEBUG" python3 bert.py --config configs/mk2/pod128_mini.json 2>&1 | tee log_mini
export POPLAR_ENGINE_OPTIONS='{"autoReport.all":"false"}'

# offline mini
poprun -vv  --offline-mode=on --remove-partition=0   --num-instances 2 --num-replicas 64  --num-ilds=2 --ipus-per-replica 2 --numa-aware=yes --host $HOSTS --vipu-server-host=$VIPU_SERVER_HOST --vipu-server-timeout=600 --vipu-cluster=$CLUSTER_NAME --mpi-global-args="--tag-output  --allow-run-as-root  --mca oob_tcp_if_include 10.10.0.0/16 --mca btl_tcp_if_include 10.10.0.0/16" --mpi-local-args="-x LD_LIBRARY_PATH -x OPAL_PREFIX -x PATH -x OMPI_CPPFLAGS -x HOROVOD_STALL_CHECK_TIME_SECONDS=600 -x HOROVOD_LOG_LEVEL=INFO -x HOROVOD_POPART_BROADCAST_TIMEOUT=600 -x CPATH -x PYTHONPATH -x TMP=/localdata/$USER/tmp -x IPUOF_VIPU_API_TIMEOUT=600 -x POPLAR_ENGINE_OPTIONS  -x POPLAR_LOG_LEVEL=INFO -x GCL_LOG_LEVEL=DEBUG -x POPART_LOG_LEVEL=DEBUG" python3 bert.py --config configs/mk2/pod128_mini.json --device-connection-type=offline --device-version=ipu2 2>&1 | tee log_mini


# profile mini
poprun -vv  --remove-partition=0 --only-output-from-instance=0 --num-instances 2 --num-replicas 64  --num-ilds=2 --ipus-per-replica 2 --numa-aware=yes --host $HOSTS --vipu-server-host=$VIPU_SERVER_HOST --vipu-server-timeout=600 --vipu-partition=$PARTITION_NAME --vipu-cluster=$CLUSTER_NAME --reset-partition=0 --update-partition=yes --mpi-global-args="--tag-output  --allow-run-as-root  --mca oob_tcp_if_include 10.10.0.0/16 --mca btl_tcp_if_include 10.10.0.0/16" --mpi-local-args="-x LD_LIBRARY_PATH -x OPAL_PREFIX -x PATH -x OMPI_CPPFLAGS -x HOROVOD_STALL_CHECK_TIME_SECONDS=600 -x HOROVOD_LOG_LEVEL=INFO -x HOROVOD_POPART_BROADCAST_TIMEOUT=600 -x CPATH -x PYTHONPATH -x TMP=/localdata/$USER/tmp -x IPUOF_VIPU_API_TIMEOUT=600 -x POPLAR_ENGINE_OPTIONS  -x POPLAR_LOG_LEVEL=INFO -x GCL_LOG_LEVEL=DEBUG" python3 bert.py --config configs/mk2/pod128_mini.json --profile --gradient-accumulation=17 --profile-instrument 2>&1 | tee log_mini
export POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true", "autoReport.directory":"/localdata/matejk/profile", "autoReport.outputExecutionProfile":"true","debug.allowOutOfMemory":"true","debug.outputAllSymbols":"false", "profiler.replicaToProfile":"0"}'
