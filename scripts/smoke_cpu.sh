#!/usr/bin/env bash
# CPU proof of the whole Block 2 pipeline with a tiny random stand-in model.
# Produces no useful model; proves every code path. About one minute.
set -euo pipefail
OUT="${1:-runs/smoke-cpu}"
[ -f models/tiny-qwen2-smoke/config.json ] || \
  python scripts/make_tiny_model.py --corpus data/pilot/corpus.jsonl --out models/tiny-qwen2-smoke
rm -rf "$OUT"
bash scripts/run_pilot.sh configs/smoke-cpu.yaml "$OUT"
