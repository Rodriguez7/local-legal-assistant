#!/usr/bin/env bash
# Fuse the LoRA adapter into the base weights and import the result into Ollama (int4; this Ollama build has no q4_K_M for safetensors).
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf models/fused
.venv/bin/python -m mlx_lm fuse \
  --model models/qwen3-4b-instruct-2507 \
  --adapter-path adapters/legalrag \
  --save-path models/fused
printf 'FROM ./models/fused\nPARAMETER temperature 0\n' > models/Modelfile
ollama create legalrag-ft -f models/Modelfile --quantize int4
