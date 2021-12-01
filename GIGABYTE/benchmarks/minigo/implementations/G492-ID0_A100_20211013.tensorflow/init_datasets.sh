# run docker on gpus to download and preprocess
docker run --gpus all -v :/data -it mlperf-nvidia:minigo
cd minigo

# Download dataset, needs gsutil. (so install gcloud package inside the container)
# Download & extract bootstrap checkpoint.
gsutil cp gs://minigo-pub/ml_perf/0.7/checkpoint.tar.gz .
tar xfz checkpoint.tar.gz -C ml_perf/

# Download and freeze the target model.
mkdir -p ml_perf/target/
gsutil cp gs://minigo-pub/ml_perf/0.7/target.* ml_perf/target/

# comment out L331 in dual_net.py before running freeze_graph.
# L331 is: optimizer = hvd.DistributedOptimizer(optimizer)
# Horovod is initialized via train_loop.py and isn't needed for this step.
CUDA_VISIBLE_DEVICES=0 python3 freeze_graph.py --flagfile=ml_perf/flags/19/architecture.flags --model_path=ml_perf/target/target
mv ml_perf/target/target.minigo ml_perf/target/target.minigo.tf

# uncomment L331 in dual_net.py.
# copy dataset to /data that is mapped to outside of docker.
# Needed because run_and_time.sh uses the following paths to load checkpoint
# CHECKPOINT_DIR="/data/mlperf07"
# TARGET_PATH="/data/target/target.minigo.tf"
cp -a ml_perf/target /data/
cp -a ml_perf/checkpoints/mlperf07 /data/

# exit docker