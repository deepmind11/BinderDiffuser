# BinderDiffuser GPU image
#
# Bundles RFdiffusion, ProteinMPNN, and ColabFold so the pipeline runs in
# one container. CUDA 12.1 + PyTorch 2.2 to match RFdiffusion's tested combo.
#
# Build:
#   docker build -t binderdiffuser:latest -f docker/Dockerfile .
#
# Run (with NVIDIA runtime):
#   docker run --gpus all -v $(pwd):/workspace binderdiffuser:latest \
#     binderdiffuser run /workspace/examples/pdl1_binder/config.yaml

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-dev \
        git wget curl ca-certificates build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# PyTorch first (cuda-aware wheel), then BinderDiffuser deps.
RUN pip install --upgrade pip && \
    pip install torch==2.2.2 torchvision==0.17.2 \
        --index-url https://download.pytorch.org/whl/cu121

# RFdiffusion (run_inference.py lives at /opt/RFdiffusion/scripts).
RUN git clone https://github.com/RosettaCommons/RFdiffusion.git /opt/RFdiffusion && \
    pip install -e /opt/RFdiffusion && \
    pip install dgl==2.0.0 -f https://data.dgl.ai/wheels/cu121/repo.html
ENV PATH="/opt/RFdiffusion/scripts:${PATH}"

# ProteinMPNN
RUN git clone https://github.com/dauparas/ProteinMPNN.git /opt/ProteinMPNN
ENV PATH="/opt/ProteinMPNN:${PATH}"

# ColabFold (pip wheel; bundles AF2 weights download script).
RUN pip install --upgrade "colabfold[alphafold]"

# BinderDiffuser itself.
WORKDIR /workspace
COPY pyproject.toml README.md /workspace/
COPY src /workspace/src
RUN pip install -e ".[notebooks]"

ENTRYPOINT ["binderdiffuser"]
CMD ["--help"]
