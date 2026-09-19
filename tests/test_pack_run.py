"""scripts/pack_run.py: zips a run folder without checkpoints, byte-stable, refuses incomplete runs unless --partial."""
import hashlib
import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path("scripts/pack_run.py")
spec = importlib.util.spec_from_file_location("pack_run", SCRIPT)
pack_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pack_run)


def _fake_run(root: Path, complete: bool = True) -> Path:
    run = root / "runs" / "pilot-x"
    (run / "adapter").mkdir(parents=True)
    (run / "adapter" / "adapter_model.safetensors").write_bytes(b"\x00" * 1024)
    (run / "adapter" / "tokenizer.json").write_text("{}")
    for name in ("run_manifest.json", "eval.json", "determinism.json", "pip_freeze.txt", "config.yaml", "run_lock.json"):
        (run / name).write_text(f"{name}\n")
    (run / "eval_report.md").write_text("# report\n")
    if complete:
        (run / "predictions.jsonl").write_text('{"sample_id":"a"}\n')
    for step in (100, 200):
        ck = run / f"checkpoint-{step}"
        (ck / "optimizer").mkdir(parents=True)
        (ck / "optimizer" / "optimizer.pt").write_bytes(b"\x01" * 4096)
        (ck / "trainer_state.json").write_text("{}")
    return run


def test_pack_excludes_checkpoints_and_is_byte_stable(tmp_path: Path):
    run = _fake_run(tmp_path)
    out, names = pack_run.pack(run)
    assert out == run.with_suffix(".zip") and out.exists()
    assert not any("checkpoint-" in n for n in names)
    assert "pilot-x/adapter/adapter_model.safetensors" in names and "pilot-x/predictions.jsonl" in names
    with zipfile.ZipFile(out) as z:
        assert sorted(z.namelist()) == sorted(names)
        assert z.read("pilot-x/predictions.jsonl") == b'{"sample_id":"a"}\n'
    first = hashlib.sha256(out.read_bytes()).hexdigest()
    assert Path(str(out) + ".sha256").read_text().split()[0] == first
    out2, _ = pack_run.pack(run)
    assert hashlib.sha256(out2.read_bytes()).hexdigest() == first


def test_incomplete_run_refused_unless_partial(tmp_path: Path):
    run = _fake_run(tmp_path, complete=False)
    with pytest.raises(SystemExit, match="predictions.jsonl"):
        pack_run.pack(run)
    out, names = pack_run.pack(run, require_complete=False)
    assert out.exists() and "pilot-x/eval_report.md" in names


def test_cli(tmp_path: Path):
    run = _fake_run(tmp_path)
    res = subprocess.run([sys.executable, str(SCRIPT), str(run), "--out", str(tmp_path / "x.zip"), "--quiet"],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert (tmp_path / "x.zip").exists() and (tmp_path / "x.zip.sha256").exists()
    assert "files, sha256" in res.stdout
    res = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path / "nope")], capture_output=True, text=True)
    assert res.returncode != 0 and "does not exist" in res.stderr
