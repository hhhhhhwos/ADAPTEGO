# AdaptEgo 完全容器化版本
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app/AdaptEgo:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    cmake \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgl1-mesa-glx \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 升级pip
RUN python3 -m pip install --upgrade pip

# 安装Python深度学习依赖
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN pip3 install --no-cache-dir \
    stable-baselines3 \
    tensorboard \
    numpy \
    pandas \
    matplotlib \
    scikit-learn \
    tqdm \
    pybullet \
    gymnasium

# 安装 ROS Python 包和消息包
RUN apt-get update && apt-get install -y \
    ros-noetic-rospy \
    ros-noetic-geometry-msgs \
    ros-noetic-nav-msgs \
    ros-noetic-sensor-msgs \
    ros-noetic-std-msgs \
    ros-noetic-visualization-msgs \
    && rm -rf /var/lib/apt/lists/*

# gymnasium 需单独从官方 PyPI 安装
RUN pip3 install --no-cache-dir gymnasium

# 创建应用目录
WORKDIR /app

# 复制项目文件到容器内
COPY . /app/AdaptEgo/

# 创建用户 (匹配主机用户)
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g $GROUP_ID kklab && \
    useradd -u $USER_ID -g $GROUP_ID -m -s /bin/bash kklab

# 设置项目目录权限
RUN chown -R kklab:kklab /app/AdaptEgo

# 切换用户
USER kklab

# 设置工作目录
WORKDIR /app/AdaptEgo

# 创建必要目录
RUN mkdir -p models data logs results

# 暴露端口 (TensorBoard)
EXPOSE 6006

# 设置启动命令
CMD ["/bin/bash"]
