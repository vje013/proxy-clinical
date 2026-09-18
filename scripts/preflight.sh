#!/usr/bin/env bash
# Environment preflight: fails fast on a broken stack BEFORE any model download.
#   1. import probes for the packages peft/transformers touch lazily (torchvision,
#      torchaudio, torchao, bitsandbytes must be absent or compatible)
#   2. one training step + a two-sample inference on the tiny CPU stand-in model,
#      which exercises TRL SFTTrainer, PEFT LoRA injection (all dispatchers),
#      generation, and the strict evaluator on the installed versions.
# About 30-60 s on a Colab CPU. Usage: bash scripts/preflight.sh
set -euo pipefail
echo "== preflight: import probes"
python - <<'EOF'
import importlib.util, sys
import torch, transformers, peft, trl, datasets, accelerate
print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} | transformers {transformers.__version__} | "
      f"peft {peft.__version__} | trl {trl.__version__} | datasets {datasets.__version__} | accelerate {accelerate.__version__}")
problems = []
for name in ("torchvision", "torchaudio", "torchao", "bitsandbytes"):
    if importlib.util.find_spec(name) is None:
        print(f"  {name}: absent (good)")
        continue
    try:
        mod = importlib.import_module(name)
        print(f"  {name}: present, version {getattr(mod, '__version__', '?')}")
    except Exception as exc:  # built against a different torch, or otherwise broken
        problems.append(f"{name} is installed but fails to import: {type(exc).__name__}: {str(exc)[:120]}")
# peft's own compatibility probes, called the way its LoRA dispatchers call them.
from peft.import_utils import is_torchao_available, is_bnb_available
for probe in (is_torchao_available, is_bnb_available):
    try:
        probe()
    except Exception as exc:
        problems.append(f"peft probe {probe.__name__} raised: {str(exc)[:160]}")
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401  (lazy-import resolution must work)
if problems:
    print("PREFLIGHT FAILED:", file=sys.stderr)
    for p in problems:
        print("  -", p, file=sys.stderr)
    print("Fix: pip uninstall -y torchvision torchaudio torchao bitsandbytes   (none are used here)", file=sys.stderr)
    sys.exit(1)
print("  import probes OK")
EOF

echo "== preflight: one-step CPU training + inference on the tiny stand-in"
[ -f models/tiny-qwen2-smoke/config.json ] || \
  python scripts/make_tiny_model.py --corpus data/pilot/corpus.jsonl --out models/tiny-qwen2-smoke >/dev/null
rm -rf runs/preflight
python -m lora.train --config configs/smoke-cpu.yaml --output-dir runs/preflight --max-steps 1 2>&1 | grep -E "^\[train\]" || true
test -f runs/preflight/adapter/adapter_model.safetensors || { echo "PREFLIGHT FAILED: training did not produce an adapter" >&2; exit 1; }
python -m lora.infer --config configs/smoke-cpu.yaml --adapter runs/preflight/adapter --out runs/preflight/predictions.jsonl --limit 2 2>&1 | grep -E "^\[infer\] wrote" || { echo "PREFLIGHT FAILED: inference" >&2; exit 1; }
python -m lora.evaluate --predictions runs/preflight/predictions.jsonl --corpus data/pilot/corpus.jsonl --out runs/preflight/eval 2>&1 | grep -E "^\[evaluate\]" || { echo "PREFLIGHT FAILED: evaluate" >&2; exit 1; }
echo "== preflight OK"
