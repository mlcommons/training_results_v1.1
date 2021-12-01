ARG FROM_IMAGE_NAME=nvcr.io/nvidia/mxnet:21.05-py3
FROM ${FROM_IMAGE_NAME}

# Install dependencies for system configuration logger
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        infiniband-diags \
        pciutils \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
WORKDIR /workspace/image_classification

ARG MLPERF_LOGGING_BRANCH=1.1.0-rc4 
ARG PYTHON=python3.8
ARG PADDLE_WHEEL=paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl

COPY requirements.txt .
RUN $PYTHON -m pip install -U pip
RUN $PYTHON -m pip install -r requirements.txt && rm requirements.txt
RUN curl -O https://paddle-wheel.bj.bcebos.com/mlperf-1.1/$PADDLE_WHEEL \
        && $PYTHON -m pip install $PADDLE_WHEEL && rm -rf $PADDLE_WHEEL  

# Patch DALI PaddlePaddle plugin bug of wrong CUDA stream synchronization 
COPY dali_paddle_patch.py /usr/local/lib/python3.8/dist-packages/nvidia/dali/plugin/paddle.py

COPY . .

# Configure environment variables
ENV OMPI_MCA_btl=^openib \
    OMP_NUM_THREADS=1 \
    OPENCV_FOR_THREADS_NUM=1
