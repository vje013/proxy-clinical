"""Inference determinism test.

    python -m lora.determinism --config configs/pilot.yaml --adapter runs/pilot/adapter \
        --input data/pilot/val.jsonl --n 32 --out runs/pilot/determinism

Runs inference twice on the same inputs with the same checkpoint, in this
process and environment, and requires the raw output strings to be
byte-identical. The claim proven is exactly that: same checkpoint, same input,
same pinned environment. It says nothing about other GPUs or other library
versions, and the JSON it writes records the environment so nobody can read
it as more.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from .common import environment_info, seed_everything, set_determinism_env
from .config import load_config
from .data import read_training_view
from .infer import generate_all, load_model_for_inference


def _digest(preds: list[dict]) -> str:
    h = hashlib.sha256()
    for p in preds:
        h.update(p["sample_id"].encode("utf-8"))
        h.update(b"\x00")
        h.update(p["raw_output"].encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Two-run byte-identity test for inference")
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--input", default=None)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--out", required=True, help="output prefix: writes <out>.json")
    ap.add_argument("--reload", action="store_true", help="reload the model between runs (slower, stricter)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    set_determinism_env()
    rows = read_training_view(args.input or cfg.data.val, cfg.data.instruction_version, args.n)

    runs: list[list[dict]] = []
    tokenizer = model = device = None
    for run_no in range(2):
        seed_everything(cfg.train.seed, deterministic_algorithms=cfg.infer.deterministic_algorithms)
        if run_no == 0 or args.reload or model is None:
            tokenizer, model, device = load_model_for_inference(cfg, args.adapter)
        runs.append(generate_all(cfg, tokenizer, model, device, rows, progress=False))
        print(f"[determinism] run {run_no + 1} done, digest {_digest(runs[-1])[:16]}", file=sys.stderr)

    diffs = [a["sample_id"] for a, b in zip(runs[0], runs[1]) if a["raw_output"] != b["raw_output"]]
    result = {
        "pass": not diffs,
        "n": len(rows),
        "differing_samples": diffs,
        "digest_run1": _digest(runs[0]),
        "digest_run2": _digest(runs[1]),
        "reloaded_between_runs": bool(args.reload),
        "scope": "same checkpoint, same inputs, same process environment; not a cross-hardware claim",
        "adapter": args.adapter,
        "decoding": {"do_sample": False, "num_beams": 1, "max_new_tokens": cfg.infer.max_new_tokens,
                     "batch_size": cfg.infer.batch_size},
        "environment": environment_info(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(out) + ".json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"[determinism] {'PASS' if result['pass'] else 'FAIL'} on {len(rows)} samples "
          f"({len(diffs)} differ)", file=sys.stderr)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
