"""End-to-end CPU smoke test: tiny stand-in model, a few steps of LoRA SFT,
resume from checkpoint, deterministic inference, evaluation, determinism test.
Slow (about a minute); skip with `-m "not slow"`."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.slow

TINY = Path("models/tiny-qwen2-smoke")


def _run(args: list[str]) -> subprocess.CompletedProcess:
    res = subprocess.run([sys.executable, "-m", *args], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-4000:]
    return res


@pytest.fixture(scope="module")
def tiny_model():
    if not (TINY / "config.json").exists():
        subprocess.run([sys.executable, "scripts/make_tiny_model.py", "--corpus", "data/pilot/corpus.jsonl",
                        "--out", str(TINY)], check=True, capture_output=True)
    return TINY


def test_pipeline_end_to_end(tmp_path: Path, tiny_model):
    raw = yaml.safe_load(Path("configs/smoke-cpu.yaml").read_text())
    raw["train"]["output_dir"] = str(tmp_path / "run")
    raw["train"]["max_steps"] = 2
    raw["train"]["save_steps"] = 1
    raw["data"]["limit_train"] = 8
    raw["data"]["limit_val"] = 4
    raw["infer"]["max_new_tokens"] = 12
    cfg_path = tmp_path / "smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(raw))

    # train
    _run(["lora.train", "--config", str(cfg_path)])
    run = tmp_path / "run"
    assert (run / "adapter" / "adapter_model.safetensors").exists()
    assert (run / "adapter" / "tokenizer.json").exists()
    manifest = json.loads((run / "run_manifest.json").read_text())
    assert manifest["training"]["global_steps"] == 2
    assert manifest["data"]["instruction_version"] == "v2"
    assert len(manifest["data"]["train_sha256"]) == 64
    assert (run / "pip_freeze.txt").stat().st_size > 0
    assert (run / "checkpoint-2").is_dir()

    # resume for one more step from the latest checkpoint
    _run(["lora.train", "--config", str(cfg_path), "--resume", "--max-steps", "3"])
    manifest2 = json.loads((run / "run_manifest.json").read_text())
    assert manifest2["training"]["resumed_from"].endswith("checkpoint-2")
    assert manifest2["training"]["global_steps"] == 3

    # A checkpoint may not be resumed under a different contract/data/config.
    lock = json.loads((run / "run_lock.json").read_text())
    assert lock["instruction_version"] == "v2"
    tampered = dict(lock); tampered["train_sha256"] = "0" * 64
    (run / "run_lock.json").write_text(json.dumps(tampered))
    res = subprocess.run([sys.executable, "-m", "lora.train", "--config", str(cfg_path), "--resume", "--max-steps", "4"],
                         capture_output=True, text=True)
    assert res.returncode != 0 and "refusing to resume" in res.stderr
    (run / "run_lock.json").write_text(json.dumps(lock))

    # --resume with no checkpoint must fail; --resume-if-exists must not
    res = subprocess.run([sys.executable, "-m", "lora.train", "--config", str(cfg_path), "--resume",
                          "--output-dir", str(tmp_path / "empty")], capture_output=True, text=True)
    assert res.returncode != 0 and "no checkpoint" in res.stderr

    # infer
    preds = run / "predictions.jsonl"
    _run(["lora.infer", "--config", str(cfg_path), "--adapter", str(run / "adapter"), "--out", str(preds), "--limit", "3"])
    lines = [json.loads(l) for l in preds.read_text().splitlines()]
    assert len(lines) == 3 and all(set(l) >= {"sample_id", "raw_output", "finished"} for l in lines)
    pm = json.loads(Path(str(preds) + ".manifest.json").read_text())
    assert pm["decoding"]["do_sample"] is False and pm["decoding"]["num_beams"] == 1

    # evaluate (random model: everything malformed, nothing crashes)
    _run(["lora.evaluate", "--predictions", str(preds), "--corpus", "data/pilot/corpus.jsonl", "--out", str(run / "eval")])
    ev = json.loads((run / "eval.json").read_text())
    assert ev["slices"]["all"]["samples"] == 3
    assert (run / "eval_report.md").read_text().startswith("# Proxy Clinical tagger evaluation")

    # determinism
    res = _run(["lora.determinism", "--config", str(cfg_path), "--adapter", str(run / "adapter"), "--n", "3",
                "--out", str(run / "determinism"), "--reload"])
    det = json.loads((run / "determinism.json").read_text())
    assert det["pass"] is True and det["digest_run1"] == det["digest_run2"]
