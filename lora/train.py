"""LoRA SFT on the Block 1 training view.

    python -m lora.train --config configs/pilot.yaml
    python -m lora.train --config configs/pilot.yaml --resume            # latest checkpoint, error if none
    python -m lora.train --config configs/pilot.yaml --resume-if-exists  # resume when possible, else fresh

Writes <output_dir>/adapter (LoRA weights + tokenizer), checkpoints every
``save_steps`` for Colab disconnects, and run_manifest.json with everything
needed to reproduce: config and its hash, pinned revisions, data file hashes,
token-length stats, library versions, GPU, training log, and pip freeze.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from . import __version__
from .common import (
    environment_info, load_base_model, load_tokenizer, pip_freeze, resolved_revision, seed_everything,
    set_determinism_env,
)
from .config import RunConfig, file_sha256, load_config
from .data import check_lengths, load_datasets

_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def latest_checkpoint(output_dir: Path) -> Path | None:
    best: tuple[int, Path] | None = None
    if not output_dir.exists():
        return None
    for p in output_dir.iterdir():
        m = _CKPT_RE.match(p.name)
        if p.is_dir() and m:
            step = int(m.group(1))
            if best is None or step > best[0]:
                best = (step, p)
    return best[1] if best else None


def build_sft_config(cfg: RunConfig, output_dir: Path):
    from trl import SFTConfig
    t = cfg.train
    return SFTConfig(
        output_dir=str(output_dir),
        seed=t.seed,
        data_seed=t.seed,
        learning_rate=t.learning_rate,
        num_train_epochs=t.num_train_epochs,
        max_steps=t.max_steps,
        per_device_train_batch_size=t.per_device_train_batch_size,
        per_device_eval_batch_size=t.per_device_eval_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        max_length=t.max_length,
        lr_scheduler_type=t.lr_scheduler_type,
        warmup_steps=t.warmup_steps,
        weight_decay=t.weight_decay,
        max_grad_norm=t.max_grad_norm,
        logging_steps=t.logging_steps,
        eval_strategy=t.eval_strategy,
        eval_steps=t.save_steps if t.eval_strategy == "steps" else None,
        save_strategy=t.save_strategy,
        save_steps=t.save_steps,
        save_total_limit=t.save_total_limit,
        gradient_checkpointing=t.gradient_checkpointing,
        bf16=t.bf16,
        fp16=t.fp16,
        use_cpu=t.use_cpu,
        full_determinism=t.full_determinism,
        completion_only_loss=True,      # loss on the JSON target only, never on the prompt
        packing=False,                  # keep one sample per sequence; offsets must not see neighbours
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=True,
        disable_tqdm=False,
    )


def build_lora_config(cfg: RunConfig):
    from peft import LoraConfig
    return LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )


def train(cfg: RunConfig, resume: str | None, resume_if_exists: bool, output_dir_override: str | None = None,
          max_steps_override: int | None = None) -> dict:
    set_determinism_env()
    seed_everything(cfg.train.seed)
    from trl import SFTTrainer

    output_dir = Path(output_dir_override or cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if max_steps_override is not None:
        cfg.train.max_steps = max_steps_override

    resume_from: str | None = None
    if resume is not None:
        if resume == "auto":
            ckpt = latest_checkpoint(output_dir)
            if ckpt is None:
                if resume_if_exists:
                    print(f"[train] no checkpoint under {output_dir}; starting fresh", file=sys.stderr)
                else:
                    raise SystemExit(f"--resume requested but no checkpoint-* directory exists under {output_dir}")
            else:
                resume_from = str(ckpt)
        else:
            if not Path(resume).is_dir():
                raise SystemExit(f"--resume path {resume} is not a directory")
            resume_from = resume
    if resume_from:
        print(f"[train] resuming from {resume_from}", file=sys.stderr)

    t0 = time.time()
    tokenizer = load_tokenizer(cfg.model.tokenizer_path, cfg.model.tokenizer_rev, cfg.model.trust_remote_code)
    train_ds, val_ds = load_datasets(cfg.data)
    length_stats = {
        "train": check_lengths(train_ds, tokenizer, cfg.train.max_length, "train"),
        "val": check_lengths(val_ds, tokenizer, cfg.train.max_length, "val"),
    }
    print(f"[train] token lengths: {json.dumps(length_stats)}", file=sys.stderr)

    model = load_base_model(cfg.model, cfg.model.torch_dtype)
    model.config.use_cache = False
    base_revision = resolved_revision(model)

    trainer = SFTTrainer(
        model=model,
        args=build_sft_config(cfg, output_dir),
        train_dataset=train_ds,
        eval_dataset=val_ds if cfg.train.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=build_lora_config(cfg),
    )
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"[train] trainable params {trainable:,} of {total:,} ({100 * trainable / total:.3f}%)", file=sys.stderr)

    result = trainer.train(resume_from_checkpoint=resume_from)

    adapter_dir = output_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)

    manifest = {
        "lora_package_version": __version__,
        "run_name": cfg.name,
        "config": cfg.to_dict(),
        "config_sha256": cfg.config_hash(),
        "model": {
            "name_or_path": cfg.model.name_or_path,
            "pinned_revision": cfg.model.revision,
            "resolved_commit_hash": base_revision,
            "tokenizer": cfg.model.tokenizer_path,
            "tokenizer_revision": cfg.model.tokenizer_rev,
        },
        "data": {
            "train": cfg.data.train, "train_sha256": file_sha256(cfg.data.train),
            "val": cfg.data.val, "val_sha256": file_sha256(cfg.data.val),
            "corpus": cfg.data.corpus, "corpus_sha256": file_sha256(cfg.data.corpus) if Path(cfg.data.corpus).exists() else None,
            "instruction_version": cfg.data.instruction_version,
            "n_train": len(train_ds), "n_val": len(val_ds),
            "token_lengths": length_stats,
        },
        "lora": {"trainable_params": trainable, "total_params": total},
        "training": {
            "resumed_from": resume_from,
            "global_steps": trainer.state.global_step,
            "train_runtime_s": round(result.metrics.get("train_runtime", 0.0), 1),
            "train_loss": result.metrics.get("train_loss"),
            "log_history": trainer.state.log_history,
            "wall_time_s": round(time.time() - t0, 1),
        },
        "environment": environment_info(),
        "adapter_dir": str(adapter_dir),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    (output_dir / "pip_freeze.txt").write_text(pip_freeze(), encoding="utf-8")
    print(f"[train] adapter saved to {adapter_dir}; manifest at {output_dir / 'run_manifest.json'}", file=sys.stderr)
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LoRA SFT for the Proxy Clinical tagger")
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", nargs="?", const="auto", default=None,
                    help="resume from the latest checkpoint in output_dir (or an explicit checkpoint path)")
    ap.add_argument("--resume-if-exists", action="store_true",
                    help="with --resume: start fresh instead of failing when no checkpoint exists")
    ap.add_argument("--output-dir", default=None, help="override train.output_dir (e.g. a Drive path)")
    ap.add_argument("--max-steps", type=int, default=None, help="override train.max_steps (smoke runs)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    resume = args.resume if args.resume is not None else ("auto" if args.resume_if_exists else None)
    train(cfg, resume, args.resume_if_exists, args.output_dir, args.max_steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
