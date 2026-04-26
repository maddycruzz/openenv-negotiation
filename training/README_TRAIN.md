# GRPO Training — Social Agent Negotiation

Fine-tunes **Qwen2.5-14B-Instruct** on the Social Agent Negotiation environment using GRPO (Group Relative Policy Optimization).

## Model

| | |
|---|---|
| Base model | `unsloth/Qwen2.5-14B-Instruct-bnb-4bit` |
| Fine-tuning method | GRPO with LoRA (r=16, alpha=16) |
| Output repo | [`Bharath-1608/negotiation-agent-grpo`](https://huggingface.co/Bharath-1608/negotiation-agent-grpo) |

## Hardware

| | |
|---|---|
| GPU | HuggingFace Spaces A10G (24 GB VRAM) |
| 4-bit quantisation | Yes (required — 14B will not fit otherwise) |
| Unsloth | Required (use `training/Dockerfile`) |

## Cost & Time

| | |
|---|---|
| A10G rate | $1.05 / hr |
| Hard time limit | 2.5 hours |
| Cost per run | ~$2.63 |
| Budget ceiling | $30 (~11 runs) |

The script enforces the 2.5-hour wall before every training round. If the limit is hit, it saves a checkpoint and exits cleanly.

## Training Config

| Param | Value |
|---|---|
| Training rounds | 2 |
| Episodes per task | 6 |
| Max turns per episode | 20 |
| Batch size | 1 |
| Gradient accumulation | 4 |
| Learning rate | 2e-5 |
| Max new tokens | 512 |
| num_generations | 2 |

## Tasks

All 5 environment tasks are used:

- `single-round-consensus`
- `multi-round-negotiation`
- `adversarial-information`
- `pediatric-meningitis`
- `opioid-overdose`

## Running

### On HuggingFace Spaces A10G (Docker)

Create a new HF Space with **A10G hardware** and **Docker SDK**, then push this repo with `training/Dockerfile` promoted to root, or set `Dockerfile` path in Space settings.

```bash
HF_TOKEN=<your_token> python training/train.py
```

### Building the Docker image locally

```bash
docker build -f training/Dockerfile -t negotiation-train .
docker run --gpus all -e HF_TOKEN=<your_token> negotiation-train
```

## Output

- **Checkpoints**: saved to `./checkpoints/round_N/` after each round
- **Model**: pushed to `Bharath-1608/negotiation-agent-grpo` on HuggingFace Hub
- **Reward curve**: saved as `reward_curve.png`
