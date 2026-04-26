FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Unsloth A10G install (cu118 + torch 2.1.0 — must come before other deps)
RUN pip install --no-cache-dir \
    "unsloth[cu118-torch210] @ git+https://github.com/unslothai/unsloth.git"

# Remaining deps (torch and unsloth already installed above)
RUN pip install --no-cache-dir \
    trl \
    transformers \
    datasets \
    accelerate \
    peft \
    huggingface_hub \
    requests \
    matplotlib

WORKDIR /app
COPY . /app

EXPOSE 7860

CMD ["python", "training/train.py"]
