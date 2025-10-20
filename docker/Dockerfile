FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
# https://hub.docker.com/layers/nvidia/cuda/12.1.1-devel-ubuntu22.04/images/sha256-327c9e046fbf662275be0934742f7e5412f9b24402ee90bf4d649c1a21707912?context=explore

###  Set environment variables ### 
# Set environment variable to noninteractive mode. Frontend for automatic installs (makes the default answers be used for all questions)
ENV DEBIAN_FRONTEND noninteractive

# Optimize PyTorch for specific GPU architectures and reduce the size of CUDA binaries.
# Ensure your GPU's compute capability is listed in TORCH_CUDA_ARCH_LIST, see https://developer.nvidia.com/cuda-gpus or latest list from June 24, 2025
#### 12.0
# NVIDIA RTX PRO 6000 Blackwell Server Edition
# NVIDIA RTX PRO 6000 Blackwell Workstation Edition
# NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
# NVIDIA RTX PRO 5000 Blackwell
# NVIDIA RTX PRO 4500 Blackwell
# NVIDIA RTX PRO 4000 Blackwell
# GeForce RTX 5090
# GeForce RTX 5080
# GeForce RTX 5070 Ti
# GeForce RTX 5070
# GeForce RTX 5060 Ti
# GeForce RTX 5060
#### 10.0
# NVIDIA GB200
# NVIDIA B200
#### 9.0
# NVIDIA GH200
# NVIDIA H200
# NVIDIA H100
#### 8.9
# NVIDIA L4
# NVIDIA L40
# NVIDIA RTX 6000 Ada
# NVIDIA RTX 5000 Ada
# NVIDIA RTX 4500 Ada
# NVIDIA RTX 4000 Ada
# NVIDIA RTX 4000 SFF Ada
# NVIDIA RTX 2000 Ada
# GeForce RTX 4090
# GeForce RTX 4080
# GeForce RTX 4070 Ti
# GeForce RTX 4070
# GeForce RTX 4060 Ti
# GeForce RTX 4060
# GeForce RTX 4050
#### 8.7
# Jetson AGX Orin
# Jetson Orin NX
# Jetson Orin Nano
#### 8.6
# NVIDIA A40
# NVIDIA A10
# NVIDIA A16
# NVIDIA A2
# NVIDIA RTX A6000
# NVIDIA RTX A5000
# NVIDIA RTX A4000
# NVIDIA RTX A3000
# NVIDIA RTX A2000
# GeForce RTX 3090 Ti
# GeForce RTX 3090
# GeForce RTX 3080 Ti
# GeForce RTX 3080
# GeForce RTX 3070 Ti
# GeForce RTX 3070
# GeForce RTX 3060 Ti
# GeForce RTX 3060
# GeForce RTX 3050 Ti
# GeForce RTX 3050
#### 8.0
# NVIDIA A100
# NVIDIA A30
#### 7.5
# NVIDIA T4
# QUADRO RTX 8000
# QUADRO RTX 6000
# QUADRO RTX 5000
# QUADRO RTX 4000
# QUADRO RTX 3000
# QUADRO  T2000
# NVIDIA T1200
# NVIDIA T1000
# NVIDIA T600
# NVIDIA T500
# NVIDIA T400
# GeForce GTX 1650 Ti
# NVIDIA TITAN RTX
# GeForce RTX 2080 Ti
# GeForce RTX 2080
# GeForce RTX 2070
# GeForce RTX 2060

# Install the required packages
ENV TORCH_CUDA_ARCH_LIST="7.5 8.6 8.9 12.0" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    FORCE_CUDA="1"
####################################### 

### General system packages ###
RUN apt-get update && \
    apt-get install -y sudo openssh-server python3-pip htop nano git && \
    apt-get install -y ffmpeg libsm6 libxext6 ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 locales

# apt clean up
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# configure default system-wide locale settings to be en_US.UTF-8, required for ouster-sdk
RUN locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    export LANG=en_US.UTF-8

# (Optional, SSH setup)
    # Set a password for the root user to be able to connect via SSH
RUN echo 'root:root' | chpasswd
    # Allow root login via SSH
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
    # Expose SSH port
EXPOSE 22

# (Optional, Set time and timezone to UTC)
RUN apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    echo 'Etc/UTC' > /etc/timezone && \
    ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Symbolic link from python3 to python    
RUN ln -sv /usr/bin/python3 /usr/bin/python
#######################################

### Create a non-root user ###
ARG USER_NAME
ARG USER_ID
RUN useradd -m --no-log-init --system --uid ${USER_ID} ${USER_NAME} -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/nopasswd
RUN echo "${USER_NAME}:${USER_NAME}" | chpasswd
#######################################

# Make RUN commands use `bash --login`
SHELL ["/bin/bash", "--login", "-c"]

# Miniconda installation ### see available links downloads at https://repo.anaconda.com/miniconda/ and adjust url and sha256sum accordingly
## Here: with Python 3.10, Minonda Version 24.9.2 for Linux-x86_64
ARG MINICONDA_INSTALLER_URL_LINUX64=https://repo.anaconda.com/miniconda/Miniconda3-py310_24.9.2-0-Linux-x86_64.sh
ARG SHA256SUM_LINUX64=364869f004c6259268b09a667ed476080bf4f44e9c4ab06a293146df8990d43f

# -O: specifies the output file location
# bash /opt/conda/miniconda.sh runs the downloaded installer script
# -b: ("batch" mode flag) -> installation will proceed without any prompts 
# -p /opt/miniconda: Specifies the installation path for Miniconda

RUN wget "${MINICONDA_INSTALLER_URL_LINUX64}" -O /home/${USER_NAME}/miniconda.sh -q && \
    echo "${SHA256SUM_LINUX64} /home/${USER_NAME}/miniconda.sh" > shasum && sha256sum --check --status shasum

ARG CONDA_DIR="/home/${USER_NAME}/conda"

USER ${USER_NAME}
SHELL ["/bin/bash", "--login", "-c"]

RUN bash /home/${USER_NAME}/miniconda.sh -b -p ${CONDA_DIR} && sudo rm /home/${USER_NAME}/miniconda.sh shasum && \
    sudo ln -s ${CONDA_DIR}/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    find ${CONDA_DIR} -follow -type f -name '*.a' -delete && find ${CONDA_DIR} -follow -type f -name '*.js.map' -delete && \
    ${CONDA_DIR}/bin/conda clean -afy # buildkit

# /opt/conda/bin is prioritized, enabling easy access to Conda-installed binaries
# /usr/local/nvidia/bin and /usr/local/cuda/bin paths are also included, allowing access to NVIDIA and CUDA tools.
# total PATH variable: PATH=/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV PATH=${CONDA_DIR}/bin:$PATH
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

USER root
SHELL ["/bin/bash", "--login", "-c"]

# Ensure the Conda environment is activated for non-login shells (.bashrc) and Login Shells (.profile)
    # For USER:
RUN echo ". ${CONDA_DIR}/etc/profile.d/conda.sh" >> /home/${USER_NAME}/.bashrc && \
    echo "conda activate base" >> /home/${USER_NAME}/.bashrc && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh" >> /home/${USER_NAME}/.profile && \
    echo "conda activate base" >> /home/${USER_NAME}/.profile
    # For ROOT:
RUN echo ". ${CONDA_DIR}/etc/profile.d/conda.sh" >> /root/.bashrc && \
    echo "conda activate base" >> /root/.bashrc && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh" >> /root.profile && \
    echo "conda activate base" >> /root.profile

#######################################
# Switch to non-root user
USER ${USER_NAME}
SHELL ["/bin/bash", "--login", "-c"]
### Install additional packages in conda base env
# Install pytorch v2.4.1 with cuda12.1 support
RUN pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Install packages from requirements.txt
COPY requirements.txt /home/${USER_NAME}/requirements.txt
RUN pip install -r /home/${USER_NAME}/requirements.txt && \
    rm /home/${USER_NAME}/requirements.txt

WORKDIR /home/${USER_NAME}/workspace