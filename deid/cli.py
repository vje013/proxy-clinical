"""Block 3 command line.

    python -m deid.cli keygen --out keys/
    python -m deid.cli run --corpus data/pilot/corpus.jsonl --keys keys/ --out runs/deid-pilot/
    python -m deid.cli run --corpus ... --predictions runs/<run>/predictions.jsonl --keys keys/ --out ...
    python -m deid.cli verify --run runs/deid-pilot/ [--public-key keys/signing_key.pub]

``run`` writes, under --out:
    output.jsonl            surrogated documents: sample_id, text, mentions (offset-exact)
    receipts.jsonl          one Ed25519-signed shard receipt per consistency scope
    run_receipt.json        signed receipt over all shard receipts + utility/residual verdicts
    utility_report.md/json  AE-timeline reconstruction diffed against the gold date graph
    engine_report.json      per-scope engine details. CONTAINS ORIGINAL STRINGS
                            (reported unresolvable phrases); controller-side only.
    run_manifest.json       what ran: policy identity, code, key ids, counts. No secrets.

The surrogate secret and the private signing key are read from --keys and are
written nowhere else. Receipts carry only their truncated hashes as ids.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .policy import DEFAULT_POLICY, load_policy
from .receipt import (
    ReceiptError, Signer, code_info, file_sha256, load_secret, new_surrogate_secret, public_key_from_pem,
    run_payload, save_secret, shard_payload, verify_run,
)
from .surrogate import DocIn, SurrogateEngine, doc_from_record, group_scopes, secret_key_id
from .utility import check_utility, prose_examples, render_utility_report

PRIVATE_KEY = "signing_key.pem"
PUBLIC_KEY = "signing_key.pub"
SECRET = "surrogate_secret.hex"


def _read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")


def cmd_keygen(args) -> int:
    out = Path(args.out)
    if (out / PRIVATE_KEY).exists() or (out / SECRET).exists():
        if not args.force:
            print(f"{out} already holds keys; pass --force to overwrite (this orphans every receipt signed so far)",
                  file=sys.stderr)
            return 2
    signer = Signer.generate()
    signer.save(out / PRIVATE_KEY, out / PUBLIC_KEY)
    save_secret(new_surrogate_secret(), out / SECRET)
    print(f"signing key id {signer.key_id}; surrogate secret id {secret_key_id(load_secret(out / SECRET))}; written to {out}/")
    print("Keep signing_key.pem and surrogate_secret.hex private. Distribute signing_key.pub with the receipts.")
    return 0


def _docs_from_predictions(records: list[dict], predictions_path: Path, contract: str | None) -> tuple[list[DocIn], dict, list[dict]]:
    """Mentions from model output, located exactly as the evaluator does. A
    malformed or partially unlocatable output cannot be de-identified and is a
    hard failure for that document (listed, run refused)."""
    from lora.metrics import MalformedPrediction, parse_and_locate
    preds = {p["sample_id"]: p for p in _read_jsonl(predictions_path)}
    manifest_path = Path(str(predictions_path) + ".manifest.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    contract = contract or manifest.get("instruction_version") or "v2"
    docs: list[DocIn] = []
    problems: list[dict] = []
    for rec in records:
        p = preds.get(rec["sample_id"])
        if p is None:
            problems.append({"sample_id": rec["sample_id"], "problem": "no prediction"})
            continue
        try:
            located, unloc = parse_and_locate(p["raw_output"], rec["text"], contract)
        except MalformedPrediction as exc:
            problems.append({"sample_id": rec["sample_id"], "problem": f"malformed: {exc}"})
            continue
        if unloc:
            problems.append({"sample_id": rec["sample_id"], "problem": f"{len(unloc)} unlocatable mentions"})
            continue
        docs.append(DocIn(rec["sample_id"], rec["text"],
                          [{"start": m["start"], "end": m["end"], "type": m["type"], "entity_id": m["id"]} for m in located],
                          rec.get("bundle_id")))
    tagger = {"kind": "model", "contract": contract, "predictions_sha256": file_sha256(predictions_path),
              "adapter_sha256": (manifest.get("adapter") or {}).get("sha256"), "model": manifest.get("model")}
    return docs, tagger, problems


def cmd_run(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    policy = load_policy(args.policy)
    keys = Path(args.keys)
    secret = load_secret(keys / SECRET)
    signer = Signer.load(keys / PRIVATE_KEY)
    records = _read_jsonl(args.corpus)
    if args.limit:
        records = records[:args.limit]

    if args.predictions:
        docs, tagger, problems = _docs_from_predictions(records, Path(args.predictions), args.contract)
        if problems:
            (out / "refused_documents.json").write_text(json.dumps(problems, indent=2) + "\n")
            print(f"[deid] {len(problems)} documents cannot be de-identified from these predictions "
                  f"(see {out / 'refused_documents.json'}); refusing the run", file=sys.stderr)
            return 3
    else:
        docs = [doc_from_record(r) for r in records]
        tagger = {"kind": "gold", "corpus_sha256": file_sha256(args.corpus)}

    engine = SurrogateEngine(policy, secret)
    code = code_info()
    skid = secret_key_id(secret)
    outputs: list[dict] = []
    receipts: list[dict] = []
    engine_reports: list[dict] = []
    shifts: dict[str, int] = {}
    docs_by_id = {d.sample_id: d for d in docs}
    n_residual = 0
    for scope in group_scopes(docs):
        res = engine.process_scope(scope)
        outputs.extend(res.docs)
        for d in res.docs:
            shifts[d["sample_id"]] = res.shift_days
        rep = res.report.as_dict()
        engine_reports.append(rep)
        n_residual += len(res.report.residual_hits)
        summary = {k: (len(v) if isinstance(v, list) else v) for k, v in rep.items() if k not in ("scope_id", "documents")}
        docs_in = [{"sample_id": d.sample_id, "text": d.text, "mentions": d.mentions} for d in scope]
        payload = shard_payload(res.scope_id, docs_in, res.docs, policy, tagger, skid, signer.key_id, summary, code)
        receipts.append(signer.sign(payload))

    _write_jsonl(out / "output.jsonl", outputs)
    _write_jsonl(out / "receipts.jsonl", receipts)
    (out / "engine_report.json").write_text(json.dumps(engine_reports, indent=1) + "\n", encoding="utf-8")

    # Utility check against the gold date graph (the corpus is ground truth here).
    util = check_utility(records, outputs, policy)
    examples = prose_examples(records, outputs, shifts)
    (out / "utility_report.md").write_text(render_utility_report(util, examples), encoding="utf-8")
    (out / "utility_report.json").write_text(json.dumps({"summary": util.summary(), "failures": util.failures,
                                                          "timelines": util.timelines}, indent=1) + "\n", encoding="utf-8")

    residual = {"hits": n_residual, "scopes": len(receipts)}
    rpayload = run_payload(receipts, policy, tagger, skid, signer.key_id, util.summary(), residual,
                           file_sha256(out / "output.jsonl"), code)
    run_receipt = signer.sign(rpayload)
    (out / "run_receipt.json").write_text(json.dumps(run_receipt, indent=1) + "\n", encoding="utf-8")
    (out / "signing_key.pub").write_bytes(signer.public_pem())
    manifest = {
        "deid_version": __version__, "policy": policy.identity(), "policy_path": str(policy.path), "code": code,
        "corpus": str(args.corpus), "corpus_sha256": file_sha256(args.corpus), "tagger": tagger,
        "documents": len(outputs), "shards": len(receipts), "surrogate_key_id": skid, "signing_key_id": signer.key_id,
        "utility": util.summary(), "residual_scan": residual,
        "outputs": sorted(p.name for p in out.iterdir()),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")

    ok = util.ok and n_residual == 0
    print(f"[deid] {len(outputs)} documents in {len(receipts)} shards; policy {policy.policy_id} ({policy.sha256[:16]}); "
          f"utility {'PASS' if util.ok else 'FAIL'} ({util.intervals_checked} intervals, {util.intervals_mismatched} mismatched, "
          f"{util.relative_failed} relative failed); residual hits {n_residual}; receipts signed by {signer.key_id}")
    print(f"[deid] artifacts in {out}")
    return 0 if ok else 1


def cmd_verify(args) -> int:
    run_dir = Path(args.run)
    run_receipt = json.loads((run_dir / "run_receipt.json").read_text())
    shard_receipts = _read_jsonl(run_dir / "receipts.jsonl")
    outputs = _read_jsonl(run_dir / "output.jsonl")
    trusted = public_key_from_pem(args.public_key) if args.public_key else None
    policy_path = args.policy or DEFAULT_POLICY
    try:
        checks = verify_run(run_receipt, shard_receipts, outputs, policy_path, trusted, run_dir / "output.jsonl")
    except ReceiptError as exc:
        print(f"[verify] FAIL: {exc}")
        return 1
    for k, v in checks.items():
        print(f"[verify] {k}: {'ok' if v else 'FAIL'}")
    p = run_receipt["payload"]
    print(f"[verify] PASS: {p['shards']['documents']} documents in {p['shards']['count']} shards under policy "
          f"{p['policy']['id']} ({p['policy']['sha256'][:16]}), signed by {run_receipt['signing_key_id']}"
          f"{' (trusted key)' if trusted else ' (embedded key; origin not proven)'}, at {p['timestamp_utc']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="deid", description="Proxy Clinical de-identification runtime (Block 3)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    k = sub.add_parser("keygen", help="generate the Ed25519 signing key pair and the surrogate secret")
    k.add_argument("--out", default="keys")
    k.add_argument("--force", action="store_true")
    k.set_defaults(fn=cmd_keygen)

    r = sub.add_parser("run", help="surrogate, sign, check utility")
    r.add_argument("--corpus", required=True, help="corpus.jsonl (text, mentions; entities/date_graph used only by the utility check)")
    r.add_argument("--predictions", default=None, help="predictions.jsonl from lora.infer; mentions come from the model instead of gold")
    r.add_argument("--contract", default=None, help="v1|v2; default from the predictions manifest")
    r.add_argument("--policy", default=str(DEFAULT_POLICY))
    r.add_argument("--keys", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--limit", type=int, default=None)
    r.set_defaults(fn=cmd_run)

    v = sub.add_parser("verify", help="offline verification of a run folder")
    v.add_argument("--run", required=True)
    v.add_argument("--public-key", default=None, help="trusted signing_key.pub; without it the embedded key is used")
    v.add_argument("--policy", default=None)
    v.set_defaults(fn=cmd_verify)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
