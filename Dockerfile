FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV MPLBACKEND=Agg
ENV HF_HOME=/workspace/.cache/huggingface
ENV NUMBA_CACHE_DIR=/workspace/.cache/numba

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/dire-evoc

COPY requirements/cuda.txt /tmp/requirements-cuda.txt
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -r /tmp/requirements-cuda.txt

COPY . /workspace/dire-evoc

RUN python3 -m pip install -e '.[cluster,docs]' \
    && python3 -m pip install datasets sentence-transformers pytest hdbscan torch-geometric ogb

CMD ["bash"]
