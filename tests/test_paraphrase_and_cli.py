import json
import subprocess
import sys
from pathlib import Path

from synthgen.cast import parse_master_seed
from synthgen.generate import Composition, Generator, build_plan
from synthgen.paraphrase import apply_paraphrase, relocate

MASTER = parse_master_seed("feedface" * 4)


def _narratives(n=12):
    plan = build_plan(n, Composition(), MASTER, prefix="p")
    return [r for r in Generator(MASTER).generate(plan) if r["doc_type"] == "narrative"]


def test_relocate_accepts_reordered_prose_with_spans_intact():
    r = _narratives()[0]
    # A "paraphrase" that rewords outside the spans and adds a harmless sentence.
    new_text = r["text"].replace(" was ", " had been ").replace("The event", "This event") \
        + "\n\nNo further information was available."
    new_rec, reason = relocate(r, new_text)
    assert new_rec is not None, reason
    for m in new_rec["mentions"]:
        assert new_text[m["start"]:m["end"]] == m["text"]
    assert new_rec["paraphrased"] is True


def test_relocate_discards_when_a_span_is_altered():
    r = _narratives()[0]
    victim = next(m for m in r["mentions"] if m["type"] == "PATIENT")
    new_text = r["text"].replace(victim["text"], victim["text"].upper(), 1)
    new_rec, reason = relocate(r, new_text)
    assert new_rec is None and "count changed" in reason


def test_relocate_discards_when_paraphrase_adds_unlabeled_copy():
    r = _narratives()[0]
    victim = next(m for m in r["mentions"] if m["type"] == "PATIENT" and m["text"][0].isupper())
    new_text = r["text"] + f"\n\nSee also {victim['text']}."
    new_rec, reason = relocate(r, new_text)
    assert new_rec is None  # count changed: an extra unlabeled occurrence


def test_apply_paraphrase_keeps_template_on_failure_and_logs_rate():
    recs = _narratives()
    calls = {"n": 0}

    def flaky(text, protected):
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            return text.replace(protected[0], "REDACTED", 1)  # breaks a span -> discard
        return text.replace(" was ", " had been ", 1)         # harmless -> keep

    out, stats = apply_paraphrase(recs, flaky)
    assert stats.attempted == len(recs)
    assert stats.kept + stats.discarded == stats.attempted
    assert stats.discarded >= 1 and stats.kept >= 1
    assert len(out) == len(recs)


def test_cli_generate_and_validate(tmp_path: Path):
    out = tmp_path / "corpus"
    cmd = [sys.executable, "-m", "synthgen.cli", "generate", "--n", "40", "--seed", "abcd" * 8,
           "--out", str(out), "--prefix", "t"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    for name in ("corpus.jsonl", "train.jsonl", "val.jsonl", "meta.json", "pilot_report.md"):
        assert (out / name).exists(), name
    meta = json.loads((out / "meta.json").read_text())
    assert meta["instruction_version"] == "v2"
    assert meta["determinism"] == "PASS"
    lines = (out / "corpus.jsonl").read_text().splitlines()
    assert len(lines) == 40
    res2 = subprocess.run([sys.executable, "-m", "synthgen.cli", "validate", "--corpus", str(out / "corpus.jsonl")],
                          capture_output=True, text=True)
    assert res2.returncode == 0, res2.stdout + res2.stderr
    res3 = subprocess.run([sys.executable, "-m", "synthgen.cli", "check-determinism", "--n", "20", "--seed", "abcd" * 8],
                          capture_output=True, text=True)
    assert res3.returncode == 0 and "PASS" in res3.stdout
