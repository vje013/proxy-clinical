"""Load the Block 1 training view (train.jsonl / val.jsonl) into TRL's
conversational prompt-completion format.

Every record must carry ``instruction_version == cfg.data.instruction_version``.
A mismatch on any line is a hard error: a later prompt change must force a
corpus regeneration, never a silent mix of formats.
"""
from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset

from .config import DataCfg

REQUIRED_KEYS = ("sample_id", "instruction_version", "instruction", "input", "output")


class DataError(ValueError):
    pass


def read_training_view(path: str | Path, expected_version: str, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    seen_instruction: str | None = None
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [k for k in REQUIRED_KEYS if k not in row]
            if missing:
                raise DataError(f"{path}:{lineno}: missing keys {missing}")
            if row["instruction_version"] != expected_version:
                raise DataError(
                    f"{path}:{lineno}: instruction_version {row['instruction_version']!r} != expected "
                    f"{expected_version!r} (sample {row['sample_id']}). Regenerate the corpus; do not mix formats."
                )
            if seen_instruction is None:
                seen_instruction = row["instruction"]
            elif row["instruction"] != seen_instruction:
                raise DataError(f"{path}:{lineno}: instruction text differs from earlier rows under the same version")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise DataError(f"{path}: no records")
    return rows


def build_user_message(instruction: str, text: str) -> str:
    """Exactly what the model sees at train and inference time. One place, so
    the two can never drift."""
    return f"{instruction}\n\nTEXT:\n{text}"


def to_prompt_completion(row: dict, system_prompt: str | None) -> dict:
    prompt: list[dict] = []
    if system_prompt:
        prompt.append({"role": "system", "content": system_prompt})
    prompt.append({"role": "user", "content": build_user_message(row["instruction"], row["input"])})
    return {
        "sample_id": row["sample_id"],
        "prompt": prompt,
        "completion": [{"role": "assistant", "content": row["output"]}],
    }


def load_datasets(cfg: DataCfg) -> tuple[Dataset, Dataset]:
    train_rows = read_training_view(cfg.train, cfg.instruction_version, cfg.limit_train)
    val_rows = read_training_view(cfg.val, cfg.instruction_version, cfg.limit_val)
    overlap = {r["sample_id"] for r in train_rows} & {r["sample_id"] for r in val_rows}
    if overlap:
        raise DataError(f"{len(overlap)} sample_ids appear in both train and val (e.g. {sorted(overlap)[:3]})")
    train = Dataset.from_list([to_prompt_completion(r, cfg.system_prompt) for r in train_rows])
    val = Dataset.from_list([to_prompt_completion(r, cfg.system_prompt) for r in val_rows])
    return train, val


def check_lengths(dataset: Dataset, tokenizer, max_length: int, label: str) -> dict:
    """Fail if any sample would be truncated. Truncating a completion would
    teach the model to stop early, so over-budget samples are a hard error."""
    lengths: list[int] = []
    too_long: list[tuple[str, int]] = []
    for ex in dataset:
        msgs = ex["prompt"] + ex["completion"]
        ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
        if hasattr(ids, "keys") and "input_ids" in ids:   # BatchEncoding (transformers 5.x)
            ids = ids["input_ids"]
        n = len(ids)
        lengths.append(n)
        if n > max_length:
            too_long.append((ex["sample_id"], n))
    if too_long:
        worst = sorted(too_long, key=lambda t: -t[1])[:5]
        raise DataError(
            f"{label}: {len(too_long)} samples exceed train.max_length={max_length} tokens "
            f"(worst: {worst}). Raise max_length or regenerate the corpus with fewer listing rows."
        )
    lengths.sort()
    return {
        "n": len(lengths),
        "min_tokens": lengths[0],
        "median_tokens": lengths[len(lengths) // 2],
        "p95_tokens": lengths[int(len(lengths) * 0.95) - 1] if len(lengths) >= 20 else lengths[-1],
        "max_tokens": lengths[-1],
    }
