"""Receipts (sign, verify, tamper) and the zero-tolerance utility check, plus the
CLI end to end on a pilot subset."""
import copy
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import pytest

from deid.policy import PolicyError, load_policy
from deid.receipt import (
    ReceiptError, Signer, canonical_bytes, docs_hash, load_secret, public_key_from_pem, receipt_hash, run_payload,
    save_secret, shard_payload, verify_run, verify_signature,
)
from deid.surrogate import SurrogateEngine, doc_from_record, group_scopes, parse_absolute_date, render_date
from deid.utility import check_utility
from synthgen.emit import read_jsonl, training_pair

CORPUS = "data/pilot/corpus.jsonl"
SECRET = bytes.fromhex("42" * 32)


@pytest.fixture(scope="module")
def records():
    return read_jsonl(CORPUS)[:60]


@pytest.fixture(scope="module")
def policy():
    return load_policy()


@pytest.fixture(scope="module")
def surrogated(records, policy):
    eng = SurrogateEngine(policy, SECRET)
    results = [eng.process_scope(s) for s in group_scopes([doc_from_record(r) for r in records])]
    return results


def _tagger():
    return {"kind": "gold", "corpus_sha256": "0" * 64}


def _receipts(results, policy, signer):
    receipts = []
    for res in results:
        docs_in = [{"sample_id": d["sample_id"], "text": d["text"], "mentions": d["mentions"]} for d in res.docs]  # stand-in
        p = shard_payload(res.scope_id, docs_in, res.docs, policy, _tagger(), "aa" * 8, signer.key_id,
                          {"ok": True}, {"deid_version": "t"}, timestamp="2026-01-01T00:00:00+00:00")
        receipts.append(signer.sign(p))
    return receipts


# ----------------------------------------------------------------- policy loader

def test_policy_loader_rejects_drift(tmp_path, policy):
    import yaml
    raw = copy.deepcopy(policy.raw)
    raw["entity_types"] = raw["entity_types"][:-1]
    p = tmp_path / "p.yaml"; p.write_text(yaml.safe_dump(raw))
    with pytest.raises(PolicyError, match="closed set"):
        load_policy(p)
    raw = copy.deepcopy(policy.raw); raw["types"]["AGE"]["threshold"] = 85
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(PolicyError, match="threshold: 90"):
        load_policy(p)
    raw = copy.deepcopy(policy.raw); raw["types"]["DATE"]["shift"]["unit"] = "days"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(PolicyError, match="weeks"):
        load_policy(p)
    raw = copy.deepcopy(policy.raw); raw["types"]["PATIENT"]["method"] = "redact"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(PolicyError, match="method"):
        load_policy(p)
    # Same content, different bytes -> different identity: the receipt binds the file, not the semantics.
    p.write_text(Path(policy.path).read_text() + "\n# trailing comment\n")
    p2 = load_policy(p)
    assert p2.policy_id == policy.policy_id and p2.sha256 != policy.sha256


# ----------------------------------------------------------------- receipts

def test_sign_and_verify_roundtrip_and_key_files(tmp_path):
    s = Signer.generate()
    s.save(tmp_path / "k.pem", tmp_path / "k.pub")
    assert oct((tmp_path / "k.pem").stat().st_mode & 0o777) == "0o600"
    s2 = Signer.load(tmp_path / "k.pem")
    assert s2.key_id == s.key_id
    r = s2.sign({"a": 1, "b": [1, 2]})
    verify_signature(r)
    verify_signature(r, public_key_from_pem(tmp_path / "k.pub"))
    other = Signer.generate()
    with pytest.raises(ReceiptError, match="different key"):
        verify_signature(r, other.private_key.public_key())
    r["payload"]["a"] = 2
    with pytest.raises(ReceiptError, match="does not verify"):
        verify_signature(r)


def test_canonical_bytes_are_order_independent():
    assert canonical_bytes({"b": 1, "a": {"y": 2, "x": 1}}) == canonical_bytes({"a": {"x": 1, "y": 2}, "b": 1})
    assert canonical_bytes({"a": 1}) == b'{"a":1}'


def test_shard_payload_carries_no_original_strings(surrogated, policy):
    s = Signer.generate()
    receipts = _receipts(surrogated, policy, s)
    blob = json.dumps(receipts)
    # No mention text (surrogate or original) appears in a receipt: names, dates,
    # ids, sites. Bare numbers (ages, site numbers) are too short to test by
    # substring against hex digests, so only alphabetic mentions are checked.
    for res in surrogated:
        for d in res.docs:
            for m in d["mentions"]:
                if any(c.isalpha() for c in m["text"]) and len(m["text"]) >= 4:
                    assert m["text"] not in blob, m["text"]
    assert "engine_report" in receipts[0]["payload"]
    assert all(isinstance(v, (int, bool, dict)) for v in receipts[0]["payload"]["engine_report"].values())


def test_verify_run_passes_and_detects_every_tamper(surrogated, policy, tmp_path):
    s = Signer.generate()
    receipts = _receipts(surrogated, policy, s)
    outputs = [d for r in surrogated for d in r.docs]
    util = {"pass": True, "intervals_checked": 1, "intervals_mismatched": 0}
    rp = run_payload(receipts, policy, _tagger(), "aa" * 8, s.key_id, util, {"hits": 0}, "00" * 32,
                     {"deid_version": "t"}, timestamp="2026-01-01T00:00:00+00:00")
    run_receipt = s.sign(rp)
    checks = verify_run(run_receipt, receipts, outputs, policy.path, s.private_key.public_key())
    assert all(checks.values()) and len(checks) >= 8

    # 1. a surrogate string edited after signing
    bad = copy.deepcopy(outputs)
    m = bad[0]["mentions"][0]
    bad[0]["text"] = bad[0]["text"][:m["start"]] + "X" * (m["end"] - m["start"]) + bad[0]["text"][m["end"]:]
    bad[0]["mentions"][0]["text"] = "X" * (m["end"] - m["start"])
    with pytest.raises(ReceiptError, match="output hash mismatch"):
        verify_run(run_receipt, receipts, bad, policy.path)
    # 2. a document dropped from the output
    with pytest.raises(ReceiptError, match="missing from output"):
        verify_run(run_receipt, receipts, outputs[1:], policy.path)
    # 3. a shard receipt removed / reordered
    with pytest.raises(ReceiptError, match="added, removed or reordered"):
        verify_run(run_receipt, receipts[1:], outputs, policy.path)
    with pytest.raises(ReceiptError, match="added, removed or reordered"):
        verify_run(run_receipt, receipts[::-1], outputs, policy.path)
    # 4. a shard receipt re-signed by another key
    rogue = Signer.generate()
    forged = copy.deepcopy(receipts)
    forged[0] = rogue.sign(forged[0]["payload"])
    with pytest.raises(ReceiptError):
        verify_run(run_receipt, forged, outputs, policy.path)
    # 5. run receipt payload edited
    rr = copy.deepcopy(run_receipt); rr["payload"]["utility"]["pass"] = True; rr["payload"]["shards"]["count"] += 1
    with pytest.raises(ReceiptError, match="does not verify"):
        verify_run(rr, receipts, outputs, policy.path)
    # 6. a different policy file
    p = tmp_path / "policy.yaml"; p.write_text(Path(policy.path).read_text() + "\n# edited\n")
    with pytest.raises(ReceiptError, match="policy file does not match"):
        verify_run(run_receipt, receipts, outputs, p)
    # 7. a run whose utility check failed cannot verify
    rp_bad = dict(rp); rp_bad["utility"] = {"pass": False}
    with pytest.raises(ReceiptError, match="failed utility"):
        verify_run(s.sign(rp_bad), receipts, outputs, policy.path)
    # 8. trusted key is not the signer
    with pytest.raises(ReceiptError, match="different key"):
        verify_run(run_receipt, receipts, outputs, policy.path, rogue.private_key.public_key())


def test_receipt_hash_covers_signature(surrogated, policy):
    s = Signer.generate()
    r = _receipts(surrogated[:1], policy, s)[0]
    h = receipt_hash(r)
    r2 = dict(r); r2["signature"] = "00" * 64
    assert receipt_hash(r2) != h


def test_secret_file_roundtrip(tmp_path):
    save_secret(SECRET, tmp_path / "s.hex")
    assert load_secret(tmp_path / "s.hex") == SECRET
    (tmp_path / "short.hex").write_text("abcd\n")
    with pytest.raises(ReceiptError, match="16 bytes"):
        load_secret(tmp_path / "short.hex")


# ----------------------------------------------------------------- utility check

def test_utility_passes_on_engine_output(records, surrogated, policy):
    outputs = [d for r in surrogated for d in r.docs]
    res = check_utility(records, outputs, policy)
    assert res.ok, res.failures[:3]
    s = res.summary()
    assert s["intervals_checked"] > 500 and s["intervals_mismatched"] == 0 and s["relative_failed"] == 0
    assert s["documents"] == len(records)


def _first_abs_date_mention(doc):
    for i, m in enumerate(doc["mentions"]):
        if m["type"] == "DATE" and parse_absolute_date(m["text"]):
            return i, m
    raise AssertionError


def _replace_mention(doc, idx, new_text):
    m = doc["mentions"][idx]
    delta = len(new_text) - (m["end"] - m["start"])
    doc["text"] = doc["text"][:m["start"]] + new_text + doc["text"][m["end"]:]
    m["text"] = new_text; m["end"] += delta
    for later in doc["mentions"][idx + 1:]:
        later["start"] += delta; later["end"] += delta


def test_utility_detects_one_day_drift_unshifted_date_and_age_change(records, surrogated, policy):
    outputs = [copy.deepcopy(d) for r in surrogated for d in r.docs]
    # a) one absolute date moved by one extra day: interval or entity check fails
    i, m = _first_abs_date_mention(outputs[0])
    d, shape = parse_absolute_date(m["text"])
    _replace_mention(outputs[0], i, render_date(d + dt.timedelta(days=1), shape))
    res = check_utility(records[:1], outputs[:1], policy)
    assert not res.ok
    kinds = {f["kind"] for f in res.failures}
    assert kinds & {"interval", "entity_dates_disagree", "inconsistent_shift_in_scope", "entity_date_not_shifted"}

    # b) an age changed by jitter
    outputs = [copy.deepcopy(d) for r in surrogated for d in r.docs]
    j = next(k for k, mm in enumerate(outputs[0]["mentions"]) if mm["type"] == "AGE")
    age = outputs[0]["mentions"][j]["text"]
    _replace_mention(outputs[0], j, age.replace(age.strip("aged -yearold"), str(int("".join(c for c in age if c.isdigit())) - 3), 1))
    res = check_utility(records[:1], outputs[:1], policy)
    assert any(f["kind"] == "age" for f in res.failures)

    # c) a mention removed
    outputs = [copy.deepcopy(d) for r in surrogated for d in r.docs]
    outputs[0]["mentions"].pop()
    res = check_utility(records[:1], outputs[:1], policy)
    assert any(f["kind"] == "mention_structure" for f in res.failures)


def test_utility_detects_stale_month_in_prose(policy):
    """Build a document whose prose anchor month is left unshifted and check
    it is caught (this is exactly the failure the engine's month re-rendering
    prevents)."""
    recs = read_jsonl(CORPUS)
    rec = next(r for r in recs if any(" after the " in m["text"] and "visit" in m["text"] or " after the " in m["text"] and "assessment" in m["text"]
                                      for m in r["mentions"] if m["type"] == "DATE"))
    eng = SurrogateEngine(policy, SECRET)
    res = eng.process_scope([doc_from_record(rec)])
    out = copy.deepcopy(res.docs[0])
    gold = sorted(rec["mentions"], key=lambda m: (m["start"], m["end"]))
    idx = next(i for i, (g, n) in enumerate(zip(gold, out["mentions"]))
               if g["type"] == "DATE" and " after the " in g["text"] and any(mo in g["text"] for mo in ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")))
    if out["mentions"][idx]["text"] == gold[idx]["text"]:
        pytest.skip("this scope's shift did not move the anchor month")
    _replace_mention(out, idx, gold[idx]["text"])           # put the original (stale) month back
    u = check_utility([rec], [out], policy)
    assert any(f["kind"] == "relative_date" for f in u.failures)


# ----------------------------------------------------------------- CLI end to end

def _run(args):
    res = subprocess.run([sys.executable, "-m", "deid.cli", *args], capture_output=True, text=True)
    return res


def test_cli_keygen_run_verify_and_tamper(tmp_path):
    keys = tmp_path / "keys"
    assert _run(["keygen", "--out", str(keys)]).returncode == 0
    assert (keys / "signing_key.pem").exists() and (keys / "surrogate_secret.hex").exists()
    assert _run(["keygen", "--out", str(keys)]).returncode == 2          # refuses to overwrite
    out = tmp_path / "run"
    r = _run(["run", "--corpus", CORPUS, "--keys", str(keys), "--out", str(out), "--limit", "50"])
    assert r.returncode == 0, r.stderr[-2000:]
    for name in ("output.jsonl", "receipts.jsonl", "run_receipt.json", "utility_report.md", "utility_report.json",
                 "engine_report.json", "run_manifest.json", "signing_key.pub"):
        assert (out / name).exists(), name
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["documents"] == 50 and manifest["utility"]["pass"] and manifest["residual_scan"]["hits"] == 0
    assert "secret" not in json.dumps(manifest).lower() or "surrogate_key_id" in manifest
    blob = (out / "receipts.jsonl").read_text() + (out / "run_receipt.json").read_text()
    assert load_secret(keys / "surrogate_secret.hex").hex() not in blob
    v = _run(["verify", "--run", str(out), "--public-key", str(keys / "signing_key.pub")])
    assert v.returncode == 0 and "PASS" in v.stdout, v.stdout + v.stderr
    # Determinism of the output under the same secret.
    out2 = tmp_path / "run2"
    assert _run(["run", "--corpus", CORPUS, "--keys", str(keys), "--out", str(out2), "--limit", "50"]).returncode == 0
    assert (out / "output.jsonl").read_bytes() == (out2 / "output.jsonl").read_bytes()
    # Tamper with one byte of the output: verification fails.
    p = out / "output.jsonl"
    data = p.read_text().splitlines()
    data[3] = data[3].replace('"text":"', '"text":"Z', 1)
    p.write_text("\n".join(data) + "\n")
    v = _run(["verify", "--run", str(out), "--public-key", str(keys / "signing_key.pub")])
    assert v.returncode == 1 and "FAIL" in v.stdout


def test_cli_run_from_predictions_and_refusal(tmp_path):
    """Mentions from a predictions file (oracle output in the v2 contract) give
    the same result as gold; a malformed prediction refuses the run."""
    keys = tmp_path / "keys"
    assert _run(["keygen", "--out", str(keys)]).returncode == 0
    recs = read_jsonl(CORPUS)[:20]
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps(r) + "\n" for r in recs))
    preds = tmp_path / "predictions.jsonl"
    preds.write_text("".join(json.dumps({"sample_id": r["sample_id"], "raw_output": training_pair(r, "v2")["output"],
                                         "finished": True}) + "\n" for r in recs))
    Path(str(preds) + ".manifest.json").write_text(json.dumps({"instruction_version": "v2", "adapter": {"sha256": "ab" * 32}}))
    out = tmp_path / "from-model"
    r = _run(["run", "--corpus", str(corpus), "--predictions", str(preds), "--keys", str(keys), "--out", str(out)])
    assert r.returncode == 0, r.stderr[-2000:]
    rr = json.loads((out / "run_receipt.json").read_text())["payload"]
    assert rr["tagger"]["kind"] == "model" and rr["tagger"]["adapter_sha256"] == "ab" * 32 and rr["tagger"]["contract"] == "v2"
    # Same secret, same mentions: identical output to the gold path (ids differ per sample but the engine keys on ids consistently).
    out_gold = tmp_path / "from-gold"
    assert _run(["run", "--corpus", str(corpus), "--keys", str(keys), "--out", str(out_gold)]).returncode == 0
    a = [json.loads(l)["text"] for l in (out / "output.jsonl").read_text().splitlines()]
    b = [json.loads(l)["text"] for l in (out_gold / "output.jsonl").read_text().splitlines()]
    # Training-view ids are renumbered per sample; corpus ids are bundle-wide. For
    # single documents they coincide, so texts match exactly there.
    singles = [i for i, rec in enumerate(recs) if not rec["bundle_id"]]
    assert singles and all(a[i] == b[i] for i in singles)
    # One malformed prediction -> refused run, nothing signed.
    lines = preds.read_text().splitlines()
    bad = json.loads(lines[2]); bad["raw_output"] = bad["raw_output"][:-3]
    lines[2] = json.dumps(bad)
    preds.write_text("\n".join(lines) + "\n")
    out3 = tmp_path / "refused"
    r = _run(["run", "--corpus", str(corpus), "--predictions", str(preds), "--keys", str(keys), "--out", str(out3)])
    assert r.returncode == 3 and (out3 / "refused_documents.json").exists() and not (out3 / "run_receipt.json").exists()
