FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    vim \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (includes mamba - much faster solver than conda)
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh

# Add conda/mamba to PATH early so it can be used in subsequent RUN commands
ENV PATH="/opt/conda/bin:${PATH}"

# Create a non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set up conda environment using mamba (much faster than conda)
# CIL 25.0.0 supports Python 3.10-3.12 and requires Intel, conda-forge, and ccpi channels
RUN mamba install -c https://software.repos.intel.com/python/conda -c conda-forge -c ccpi -y \
    python=3.10 \
    cil=25.0.0 \
    && conda clean -afy

# Install Python dependencies
COPY requirements.txt requirements-optional.txt requirements-dev.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir -r /tmp/requirements-optional.txt && \
    pip install --no-cache-dir -r /tmp/requirements-dev.txt

# Set up working directory
WORKDIR /workspace

# Switch to non-root user
USER $USERNAME

# Set environment variables
ENV PYTHONUNBUFFERED=1

CMD ["/bin/bash"]
