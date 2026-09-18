#!/usr/bin/env bash
# One-command pilot run: train -> infer on val -> evaluate per slice -> determinism test.
# Usage: bash scripts/run_pilot.sh [config] [output_dir]
# Re-running with the same output_dir resumes from the latest checkpoint.
set -euo pipefail
CFG="${1:-configs/pilot.yaml}"
OUT="${2:-runs/pilot-qwen2.5-3b}"
VAL="${VAL:-data/pilot/val.jsonl}"
CORPUS="${CORPUS:-data/pilot/corpus.jsonl}"
DET_N="${DET_N:-32}"

mkdir -p "$OUT"
echo "== config: $CFG"; echo "== output: $OUT"
cp "$CFG" "$OUT/config.yaml"

echo "== [1/4] train"
python -m lora.train --config "$CFG" --output-dir "$OUT" --resume-if-exists

echo "== [2/4] infer (greedy, production config) on $VAL"
python -m lora.infer --config "$CFG" --adapter "$OUT/adapter" --input "$VAL" --out "$OUT/predictions.jsonl"

echo "== [3/4] evaluate"
python -m lora.evaluate --predictions "$OUT/predictions.jsonl" --corpus "$CORPUS" --out "$OUT/eval"

echo "== [4/4] determinism (two runs, same session)"
python -m lora.determinism --config "$CFG" --adapter "$OUT/adapter" --input "$VAL" --n "$DET_N" --out "$OUT/determinism"

python -m pip freeze > "$OUT/pip_freeze.txt"
echo "== done. artifacts in $OUT:"; ls -1 "$OUT"
echo; sed -n '1,40p' "$OUT/eval_report.md"
