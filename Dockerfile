ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
ARG PYTHON_VERSION=3.13

FROM ${BASE_IMAGE} AS dev-base

ENV TZ=AU \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1\
    VIRTUAL_ENV=/.venv \
    PATH="/.venv/bin:$PATH"

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y python3.13 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv ${VIRTUAL_ENV}

RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

RUN nvcc --version
RUN IS_CUDA=$(python3 -c 'import torch ; print(torch.cuda._is_compiled())'); \
    echo "Is torch compiled with cuda: ${IS_CUDA} version:${CUDA_VERSION}"; \
    if test "${IS_CUDA}" != "True" -a ! -z "${CUDA_VERSION}"; then \
        exit 1; \
    fi

COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install -r requirements.txt

COPY /unidepth/ops/knn/ /knn
RUN chmod u+x /knn/setup.py
RUN python3 /knn/setup.py build install

CMD ["python3", "/app/scripts/demo.py"]
  