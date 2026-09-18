"""Deterministic inference with the production decoding config.

    python -m lora.infer --config configs/pilot.yaml --adapter runs/pilot/adapter \
        --input data/pilot/val.jsonl --out runs/pilot/predictions.jsonl

Decoding is greedy (do_sample=False, num_beams=1), which is what "temperature
0" means exactly; max_new_tokens is fixed by the config; the tokenizer is the
copy saved next to the adapter, so it is pinned to the training run. Samples
are batched by token length in a fixed order, so the same inputs always form
the same batches. The determinism claim this supports is: same checkpoint,
same inputs, same pinned environment, byte-identical raw output.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .common import environment_info, load_base_model, load_tokenizer, seed_everything, set_determinism_env
from .config import RunConfig, file_sha256, load_config
from .data import read_training_view, to_prompt_completion


def load_model_for_inference(cfg: RunConfig, adapter_dir: str | Path, merge: bool = True):
    import torch
    from peft import PeftModel
    adapter_dir = Path(adapter_dir)
    if not (adapter_dir / "adapter_config.json").exists():
        raise SystemExit(f"{adapter_dir} does not contain adapter_config.json")
    # Tokenizer: the copy saved with the adapter, never re-fetched from the Hub.
    tokenizer = load_tokenizer(str(adapter_dir), None, cfg.model.trust_remote_code)
    tokenizer.padding_side = "left"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_name = cfg.infer.dtype if device == "cuda" else "float32"
    base = load_base_model(cfg.model, dtype_name)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    if merge:
        model = model.merge_and_unload()
    model.to(device)
    model.eval()
    model.config.use_cache = True
    return tokenizer, model, device


def build_prompts(rows: list[dict], system_prompt: str | None) -> list[list[dict]]:
    return [to_prompt_completion(r, system_prompt)["prompt"] for r in rows]


def generate_all(cfg: RunConfig, tokenizer, model, device: str, rows: list[dict],
                 batch_size: int | None = None, progress: bool = True) -> list[dict]:
    import torch
    bs = batch_size or cfg.infer.batch_size
    prompts = build_prompts(rows, cfg.data.system_prompt)
    encoded = []
    for r, p in zip(rows, prompts):
        ids = tokenizer.apply_chat_template(p, tokenize=True, add_generation_prompt=True)
        if hasattr(ids, "keys") and "input_ids" in ids:
            ids = ids["input_ids"]
        encoded.append((r["sample_id"], list(ids)))
    # Fixed batching: sort by length (ties by sample_id) so batch composition is a
    # pure function of the input set.
    order = sorted(range(len(encoded)), key=lambda i: (len(encoded[i][1]), encoded[i][0]))
    results: dict[str, dict] = {}
    eos_ids = [tokenizer.eos_token_id]
    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id != tokenizer.eos_token_id:
        eos_ids.append(tokenizer.pad_token_id)
    t0 = time.time()
    for b in range(0, len(order), bs):
        idxs = order[b:b + bs]
        seqs = [encoded[i][1] for i in idxs]
        max_len = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), max_len), tokenizer.pad_token_id, dtype=torch.long)
        attention = torch.zeros((len(seqs), max_len), dtype=torch.long)
        for j, s in enumerate(seqs):
            input_ids[j, max_len - len(s):] = torch.tensor(s, dtype=torch.long)
            attention[j, max_len - len(s):] = 1
        input_ids = input_ids.to(device)
        attention = attention.to(device)
        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention,
                max_new_tokens=cfg.infer.max_new_tokens,
                do_sample=False,
                num_beams=1,
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_ids,
            )
        gen = out[:, max_len:]
        for j, i in enumerate(idxs):
            toks = gen[j].tolist()
            # Trim at the first eos/pad; record whether the model stopped on its own.
            finished = False
            cut = len(toks)
            for k, t in enumerate(toks):
                if t in eos_ids:
                    cut = k
                    finished = True
                    break
            text = tokenizer.decode(toks[:cut], skip_special_tokens=True)
            results[encoded[i][0]] = {
                "sample_id": encoded[i][0],
                "raw_output": text,
                "prompt_tokens": len(encoded[i][1]),
                "output_tokens": cut,
                "finished": finished,
            }
        if progress:
            done = min(b + bs, len(order))
            print(f"[infer] {done}/{len(order)} ({time.time() - t0:.0f}s)", file=sys.stderr)
    return [results[r["sample_id"]] for r in rows]


def run_inference(cfg: RunConfig, adapter_dir: str, input_path: str, out_path: str,
                  limit: int | None = None, batch_size: int | None = None, merge: bool = True) -> dict:
    set_determinism_env()
    seed_everything(cfg.train.seed, deterministic_algorithms=cfg.infer.deterministic_algorithms)
    rows = read_training_view(input_path, cfg.data.instruction_version, limit)
    tokenizer, model, device = load_model_for_inference(cfg, adapter_dir, merge=merge)
    preds = generate_all(cfg, tokenizer, model, device, rows, batch_size)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        for p in preds:
            fh.write(json.dumps(p, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        "adapter_dir": str(adapter_dir),
        "adapter_sha256": file_sha256(Path(adapter_dir) / "adapter_model.safetensors"),
        "input": input_path,
        "input_sha256": file_sha256(input_path),
        "n": len(preds),
        "decoding": {"do_sample": False, "num_beams": 1, "max_new_tokens": cfg.infer.max_new_tokens,
                     "batch_size": batch_size or cfg.infer.batch_size, "merged_adapter": merge,
                     "dtype": cfg.infer.dtype if device == "cuda" else "float32",
                     "deterministic_algorithms": cfg.infer.deterministic_algorithms},
        "finished_rate": round(sum(p["finished"] for p in preds) / max(len(preds), 1), 4),
        "environment": environment_info(),
        "predictions_sha256": file_sha256(out),
    }
    Path(str(out) + ".manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Deterministic greedy inference")
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", required=True, help="adapter directory written by lora.train")
    ap.add_argument("--input", default=None, help="training-view JSONL (default: config data.val)")
    ap.add_argument("--out", required=True, help="predictions JSONL path")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--no-merge", action="store_true", help="keep the adapter unmerged (slower)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    m = run_inference(cfg, args.adapter, args.input or cfg.data.val, args.out, args.limit, args.batch_size,
                      merge=not args.no_merge)
    print(f"[infer] wrote {m['n']} predictions to {args.out}; finished_rate={m['finished_rate']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
