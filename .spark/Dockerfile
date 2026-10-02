FROM nvcr.io/nvidia/deepstream@sha256:c25d5449fa3b2a82aeb6cdf224bdb422361ccf3e98db98077dfac3c2305048bb

ARG DEBIAN_FRONTEND=noninteractive
ARG ONNXRUNTIME_VERSION=1.16.3
ARG ONNXRUNTIME_SHA256=784dbef93b40196aa668d29d78294a81c0d21361d36530b817bb24d87e8730e8

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        clang \
        cmake \
        cppzmq-dev \
        curl \
        git \
        just \
        libeigen3-dev \
        libgtest-dev \
        libmsgpack-dev \
        libyaml-cpp-dev \
        libzmq3-dev \
        nlohmann-json3-dev \
        pkg-config \
        zlib1g-dev \
    && curl -fsSL \
        "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}.tgz" \
        -o /tmp/onnxruntime.tgz \
    && echo "${ONNXRUNTIME_SHA256}  /tmp/onnxruntime.tgz" | sha256sum -c - \
    && tar -xzf /tmp/onnxruntime.tgz -C /opt \
    && mv "/opt/onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}" /opt/onnxruntime \
    && rm -f /tmp/onnxruntime.tgz \
    && rm -rf /var/lib/apt/lists/*

ENV CUDAToolkit_ROOT=/usr/local/cuda \
    CUDA_HOME=/usr/local/cuda \
    HAS_ROS2=0 \
    IS_THOR=true \
    TensorRT_ROOT=/usr \
    onnxruntime_ROOT=/opt/onnxruntime \
    LD_LIBRARY_PATH=/opt/onnxruntime/lib:/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64

WORKDIR /workspace/gear_sonic_deploy
