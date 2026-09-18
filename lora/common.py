"""Shared helpers: environment capture, seeding, model/tokenizer loading."""
from __future__ import annotations

import datetime as dt
import os
import platform
import subprocess
import sys

from .config import ModelCfg

_DTYPES = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}


def torch_dtype(name: str):
    import torch
    return getattr(torch, _DTYPES[name])


def set_determinism_env() -> None:
    """Must run before CUDA is initialised. Harmless on CPU."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def seed_everything(seed: int, deterministic_algorithms: bool = False) -> None:
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)


def environment_info() -> dict:
    import torch
    info: dict = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    for mod in ("transformers", "peft", "trl", "datasets", "accelerate", "tokenizers"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # pragma: no cover
            info[mod] = None
    try:
        info["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or None
    except Exception:
        info["nvidia_smi"] = None
    return info


def pip_freeze() -> str:
    try:
        return subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True,
                              timeout=120).stdout
    except Exception as exc:  # pragma: no cover
        return f"# pip freeze failed: {exc}\n"


def load_tokenizer(path: str, revision: str | None, trust_remote_code: bool = False):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, revision=revision, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        # Llama 3.x ships a dedicated pad token; fall back to eos otherwise.
        if "<|finetune_right_pad_id|>" in tok.get_vocab():
            tok.pad_token = "<|finetune_right_pad_id|>"
        else:
            tok.pad_token = tok.eos_token
    return tok


def load_base_model(mcfg: ModelCfg, dtype_name: str, device_map=None):
    import torch
    from transformers import AutoModelForCausalLM
    kwargs = dict(revision=mcfg.revision, dtype=torch_dtype(dtype_name),
                  trust_remote_code=mcfg.trust_remote_code)
    if mcfg.attn_implementation:
        kwargs["attn_implementation"] = mcfg.attn_implementation
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(mcfg.name_or_path, **kwargs)
    if not torch.cuda.is_available() and dtype_name != "float32":
        model = model.float()  # CPU smoke runs: bf16 matmuls are slow or unsupported
    return model


def resolved_revision(model) -> str | None:
    """Commit hash transformers recorded when it fetched the model, if any."""
    return getattr(model.config, "_commit_hash", None)
