#!/usr/bin/env bash
# LoRA fine-tune of Qwen3-4B-Instruct-2507 on Apple Silicon with MLX.
# Prereq: `legalrag build-data` (writes training/train.jsonl and valid.jsonl) and the base model in
# models/qwen3-4b-instruct-2507 (see README). Loss is computed on the assistant answer only.
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python -m mlx_lm lora \
  --model models/qwen3-4b-instruct-2507 \
  --train --data training \
  --fine-tune-type lora --num-layers 16 \
  --batch-size 1 --iters "${ITERS:-300}" --learning-rate 1e-4 \
  --max-seq-length 3072 --grad-checkpoint --mask-prompt \
  --steps-per-eval 50 --save-every 100 \
  --adapter-path adapters/legalrag
