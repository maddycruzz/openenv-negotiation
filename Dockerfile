FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y git curl wget && rm -rf /var/lib/apt/lists/*

# Unsloth for A10G Large (cu118 + torch 2.1.0 — must come before other deps)
RUN pip install --no-cache-dir \
    "unsloth[cu118-torch210] @ git+https://github.com/unslothai/unsloth.git"

# Remaining deps (torch already in base image, unsloth installed above)
RUN pip install --no-cache-dir \
    trl \
    transformers \
    datasets \
    accelerate \
    peft \
    huggingface_hub \
    requests \
    matplotlib

COPY . /app
WORKDIR /app

EXPOSE 7860

CMD ["python", "training/train.py"]