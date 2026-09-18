import json
from pathlib import Path

import pytest
import yaml

from lora.config import ConfigError, load_config
from lora.data import DataError, build_user_message, load_datasets, read_training_view, to_prompt_completion
from lora.config import DataCfg

PILOT = Path("configs/pilot.yaml")
SMOKE = Path("configs/smoke-cpu.yaml")


def _write(tmp_path: Path, raw: dict) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


def test_pilot_config_loads_with_pinned_revision():
    cfg = load_config(PILOT)
    assert cfg.model.name_or_path == "Qwen/Qwen2.5-3B-Instruct"
    assert len(cfg.model.revision) == 40
    assert cfg.lora.r == 16 and cfg.lora.alpha == 32
    assert set(cfg.lora.target_modules) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    assert cfg.data.instruction_version == "v1"
    assert cfg.train.save_steps == 100
    assert len(cfg.config_hash()) == 64


def test_llama_fallback_config_loads():
    cfg = load_config("configs/pilot-llama.yaml")
    assert cfg.model.name_or_path.startswith("meta-llama/") and len(cfg.model.revision) == 40


def test_hub_model_without_revision_is_rejected(tmp_path):
    raw = yaml.safe_load(PILOT.read_text())
    raw["model"].pop("revision")
    with pytest.raises(ConfigError, match="revision"):
        load_config(_write(tmp_path, raw))
    raw["model"]["revision"] = "main"
    with pytest.raises(ConfigError, match="40-hex"):
        load_config(_write(tmp_path, raw))


def test_local_model_needs_no_revision():
    cfg = load_config(SMOKE)
    assert cfg.model.is_local and cfg.model.revision is None


def test_unknown_key_and_bad_values_rejected(tmp_path):
    raw = yaml.safe_load(SMOKE.read_text())
    raw["lora"]["rank"] = 8
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(_write(tmp_path, raw))
    raw = yaml.safe_load(SMOKE.read_text())
    raw["data"]["instruction_version"] = "v2"
    with pytest.raises(ConfigError, match="instruction_version"):
        load_config(_write(tmp_path, raw))


def test_instruction_version_mismatch_is_hard_error(tmp_path):
    rows = [json.loads(l) for l in open("data/pilot/val.jsonl")][:5]
    rows[3]["instruction_version"] = "v2"
    p = tmp_path / "val.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(DataError, match="instruction_version 'v2' != expected 'v1'"):
        read_training_view(p, "v1")


def test_instruction_text_drift_is_hard_error(tmp_path):
    rows = [json.loads(l) for l in open("data/pilot/val.jsonl")][:5]
    rows[2]["instruction"] = rows[2]["instruction"] + " Also be nice."
    p = tmp_path / "val.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(DataError, match="instruction text differs"):
        read_training_view(p, "v1")


def test_train_val_overlap_is_hard_error(tmp_path):
    cfg = DataCfg(train="data/pilot/val.jsonl", val="data/pilot/val.jsonl", corpus="data/pilot/corpus.jsonl")
    with pytest.raises(DataError, match="both train and val"):
        load_datasets(cfg)


def test_prompt_completion_format_matches_trl_conversational():
    rows = read_training_view("data/pilot/val.jsonl", "v1", limit=2)
    ex = to_prompt_completion(rows[0], None)
    assert set(ex) == {"sample_id", "prompt", "completion"}
    assert ex["prompt"][-1]["role"] == "user" and ex["completion"][0]["role"] == "assistant"
    assert ex["prompt"][-1]["content"] == build_user_message(rows[0]["instruction"], rows[0]["input"])
    assert json.loads(ex["completion"][0]["content"])["mentions"]
    ex_sys = to_prompt_completion(rows[0], "You are a tagger.")
    assert ex_sys["prompt"][0] == {"role": "system", "content": "You are a tagger."}


def test_datasets_load_and_limit():
    cfg = load_config(SMOKE).data
    train, val = load_datasets(cfg)
    assert len(train) == cfg.limit_train and len(val) == cfg.limit_val
