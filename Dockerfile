FROM nvcr.io/nvidia/pytorch:22.06-py3

ENV PYTHONIOENCODING=utf-8

WORKDIR "/root"

RUN source /opt/conda/bin/activate && \
    which python && \
    apt -y update && \
    apt -y full-upgrade && \
    conda create -n py10 python==3.10 -y && \
    conda activate py10 && \
    pip install --upgrade pip && \
    pip install fire && \
    pip install composer==0.16.3 && \
    pip install llm-foundry==0.4.0 && \
    pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install torch-utils && \
    FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn==1.0.4 --no-build-isolation && \
    pip install mosaicml-streaming==0.7.1 && \
    git clone https://github.com/EleutherAI/lm-evaluation-harness

WORKDIR "/root/lm-evaluation-harness"

RUN source /opt/conda/bin/activate && \
    conda activate py10 && \
    pip install bitsandbytes && \
    pip install git+https://github.com/huggingface/peft.git && \
    pip install accelerate==0.30.0 && \
    pip install transformers==4.40.1 && \
    pip install transformers[deepspeed] && \
    pip install datasets==2.19.1 && \
    pip install evaluate && \
    pip install appdirs && \
    pip install loralib && \
    pip install black && \
    pip install black[jupyter] && \
    pip install sentencepiece && \
    pip install gradio && \
    pip install scipy && \
    pip install Flask && \
    pip install -e . && \
    pip install --upgrade pytest && \
    pip install protobuf==3.20.3 && \
    apt-get install -y parallel
