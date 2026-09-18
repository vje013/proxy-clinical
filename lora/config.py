"""Run configuration: one YAML file, validated up front.

A Hub model id must carry a 40-hex commit hash in ``revision``; a local
directory may not. Anything else is a hard error before any download starts.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class ConfigError(ValueError):
    pass


@dataclass
class ModelCfg:
    name_or_path: str
    revision: str | None = None
    tokenizer_name_or_path: str | None = None
    tokenizer_revision: str | None = None
    torch_dtype: str = "bfloat16"          # bfloat16 | float16 | float32
    attn_implementation: str | None = None  # sdpa | eager | flash_attention_2 | None (library default)
    trust_remote_code: bool = False

    @property
    def is_local(self) -> bool:
        return os.path.isdir(self.name_or_path)

    @property
    def tokenizer_path(self) -> str:
        return self.tokenizer_name_or_path or self.name_or_path

    @property
    def tokenizer_rev(self) -> str | None:
        if self.tokenizer_name_or_path:
            return self.tokenizer_revision
        return self.tokenizer_revision or self.revision


@dataclass
class LoraCfg:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ])


@dataclass
class DataCfg:
    train: str
    val: str
    corpus: str                         # corpus.jsonl, for slices and gold entities at eval time
    instruction_version: str = "v1"
    system_prompt: str | None = None    # optional system turn; None = user turn only
    limit_train: int | None = None      # smoke runs only
    limit_val: int | None = None


@dataclass
class TrainCfg:
    output_dir: str
    seed: int = 1234
    learning_rate: float = 2.0e-4
    num_train_epochs: float = 3.0
    max_steps: int = -1
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_length: int = 6144
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 20
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    eval_strategy: str = "steps"        # eval on val every save_steps
    save_strategy: str = "steps"        # checkpoint every save_steps (Colab disconnect insurance)
    save_steps: int = 100
    save_total_limit: int | None = 3
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    use_cpu: bool = False
    full_determinism: bool = False      # cuDNN deterministic mode; slower, opt in


@dataclass
class InferCfg:
    max_new_tokens: int = 4096
    batch_size: int = 8
    dtype: str = "bfloat16"
    # Production decoding is greedy: do_sample=False, num_beams=1. Temperature is
    # not a parameter of greedy decoding; "temp 0" is realised by turning
    # sampling off, which is the only way to get it exactly.
    deterministic_algorithms: bool = True


@dataclass
class RunConfig:
    model: ModelCfg
    lora: LoraCfg
    data: DataCfg
    train: TrainCfg
    infer: InferCfg
    name: str = "run"

    def to_dict(self) -> dict:
        return {"name": self.name, "model": asdict(self.model), "lora": asdict(self.lora),
                "data": asdict(self.data), "train": asdict(self.train), "infer": asdict(self.infer)}

    def config_hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


def _build(cls, raw: dict | None, section: str):
    raw = raw or {}
    allowed = {f for f in cls.__dataclass_fields__}
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigError(f"{section}: unknown keys {sorted(unknown)}")
    try:
        return cls(**raw)
    except TypeError as exc:
        raise ConfigError(f"{section}: {exc}") from exc


def load_config(path: str | Path) -> RunConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = RunConfig(
        name=str(raw.get("name", Path(path).stem)),
        model=_build(ModelCfg, raw.get("model"), "model"),
        lora=_build(LoraCfg, raw.get("lora"), "lora"),
        data=_build(DataCfg, raw.get("data"), "data"),
        train=_build(TrainCfg, raw.get("train"), "train"),
        infer=_build(InferCfg, raw.get("infer"), "infer"),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: RunConfig) -> None:
    m = cfg.model
    if not m.is_local:
        if not m.revision or not _SHA_RE.match(m.revision):
            raise ConfigError(
                f"model.revision must be a 40-hex commit hash for Hub model {m.name_or_path!r} "
                f"(got {m.revision!r}); pin it with HfApi().model_info(id).sha"
            )
        if m.tokenizer_name_or_path and not os.path.isdir(m.tokenizer_name_or_path):
            if not m.tokenizer_revision or not _SHA_RE.match(m.tokenizer_revision):
                raise ConfigError("model.tokenizer_revision must be a 40-hex commit hash for a Hub tokenizer")
    if m.torch_dtype not in ("bfloat16", "float16", "float32"):
        raise ConfigError(f"model.torch_dtype {m.torch_dtype!r} not in bfloat16|float16|float32")
    if cfg.infer.dtype not in ("bfloat16", "float16", "float32"):
        raise ConfigError(f"infer.dtype {cfg.infer.dtype!r} not in bfloat16|float16|float32")
    if cfg.lora.r <= 0 or cfg.lora.alpha <= 0:
        raise ConfigError("lora.r and lora.alpha must be positive")
    if not cfg.lora.target_modules:
        raise ConfigError("lora.target_modules must not be empty")
    if cfg.train.max_length <= 0 or cfg.infer.max_new_tokens <= 0:
        raise ConfigError("train.max_length and infer.max_new_tokens must be positive")
    if cfg.train.bf16 and cfg.train.fp16:
        raise ConfigError("train.bf16 and train.fp16 are mutually exclusive")
    if cfg.data.instruction_version != "v1":
        raise ConfigError(f"data.instruction_version {cfg.data.instruction_version!r} is not supported by this trainer (v1)")


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
