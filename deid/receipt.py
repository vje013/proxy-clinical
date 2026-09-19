"""Receipts: Ed25519-signed attestations, one per shard and one per run.

A shard is one surrogate-consistency scope (a document bundle, or a single
document). Its receipt binds, under one signature:

  policy      id, version and sha256 of the policy file bytes
  code        deid version, synthgen version, git commit when available
  shard       scope id, scope rule, ordered document ids
  input       sha256 of the canonical (text, mentions) of the documents in
  output      sha256 of the canonical (text, mentions) of the documents out
  mentions    count and per-type counts (input == output by construction)
  tagger      where the mentions came from: gold labels, or a model adapter
              (sha256) and its predictions file (sha256) under a contract
  keys        surrogate secret id (sha256 of the secret, truncated) and the
              signing key id (sha256 of the raw public key, truncated)
  timestamp   UTC

The run receipt signs the ordered list of shard receipt hashes plus the
utility and residual-scan verdicts, so a set of shards cannot be added to,
removed from or reordered after the fact.

Verification is offline: the public key, the receipts, the output file and
the policy file. Nothing is recomputed from the secret. Signatures use the
``cryptography`` package (pinned); nothing cryptographic is implemented here.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.exceptions import InvalidSignature

from . import __version__ as DEID_VERSION
from .policy import Policy

RECEIPT_VERSION = "1"
SCOPE_RULE = {"document_bundle": "bundle_id when present, else sample_id", "document": "sample_id"}


class ReceiptError(ValueError):
    pass


# --------------------------------------------------------------------------- canonical hashing


def canonical_bytes(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_doc(doc: dict) -> dict:
    """The part of a document that a receipt commits to."""
    return {
        "sample_id": doc["sample_id"],
        "text": doc["text"],
        "mentions": [{"start": m["start"], "end": m["end"], "type": m["type"], "entity_id": m["entity_id"]}
                     for m in sorted(doc["mentions"], key=lambda m: (m["start"], m["end"]))],
    }


def docs_hash(docs: list[dict]) -> str:
    return sha256_hex(canonical_bytes([canonical_doc(d) for d in sorted(docs, key=lambda d: d["sample_id"])]))


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_info() -> dict:
    from synthgen import __version__ as SYN
    commit = None
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                             cwd=Path(__file__).resolve().parent.parent)
        if out.returncode == 0:
            commit = out.stdout.strip()
            dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5,
                                   cwd=Path(__file__).resolve().parent.parent)
            if dirty.returncode == 0 and dirty.stdout.strip():
                commit += "-dirty"
    except Exception:  # pragma: no cover
        commit = None
    return {"deid_version": DEID_VERSION, "synthgen_version": SYN, "git_commit": commit}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- keys


@dataclass
class Signer:
    private_key: Ed25519PrivateKey

    @classmethod
    def generate(cls) -> "Signer":
        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def load(cls, private_pem_path: str | Path) -> "Signer":
        key = serialization.load_pem_private_key(Path(private_pem_path).read_bytes(), password=None)
        if not isinstance(key, Ed25519PrivateKey):
            raise ReceiptError(f"{private_pem_path} is not an Ed25519 private key")
        return cls(key)

    def save(self, private_pem_path: str | Path, public_pem_path: str | Path) -> None:
        priv = self.private_key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
        p = Path(private_pem_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(priv)
        os.chmod(p, 0o600)
        Path(public_pem_path).write_bytes(self.public_pem())

    def public_raw(self) -> bytes:
        return self.private_key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)

    def public_pem(self) -> bytes:
        return self.private_key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)

    @property
    def key_id(self) -> str:
        return sha256_hex(self.public_raw())[:16]

    def sign(self, payload: dict) -> dict:
        data = canonical_bytes(payload)
        sig = self.private_key.sign(data)
        return {"payload": payload, "signature": sig.hex(), "public_key": self.public_raw().hex(),
                "signing_key_id": self.key_id, "signature_scheme": "Ed25519", "canonicalization": "json-sorted-compact-utf8"}


def public_key_from_hex(h: str) -> Ed25519PublicKey:
    return Ed25519PublicKey.from_public_bytes(bytes.fromhex(h))


def public_key_from_pem(path: str | Path) -> Ed25519PublicKey:
    key = serialization.load_pem_public_key(Path(path).read_bytes())
    if not isinstance(key, Ed25519PublicKey):
        raise ReceiptError(f"{path} is not an Ed25519 public key")
    return key


def verify_signature(receipt: dict, trusted_public_key: Ed25519PublicKey | None = None) -> None:
    """Raises ReceiptError when the signature does not verify. With a trusted
    key, the receipt's embedded key must be that key; without one, the
    embedded key is used (which proves integrity, not origin)."""
    try:
        embedded = public_key_from_hex(receipt["public_key"])
    except Exception as exc:
        raise ReceiptError(f"bad embedded public key: {exc}") from exc
    if trusted_public_key is not None:
        t = trusted_public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        if t.hex() != receipt["public_key"]:
            raise ReceiptError("receipt was signed by a different key than the trusted one")
    if sha256_hex(bytes.fromhex(receipt["public_key"]))[:16] != receipt.get("signing_key_id"):
        raise ReceiptError("signing_key_id does not match the embedded public key")
    try:
        embedded.verify(bytes.fromhex(receipt["signature"]), canonical_bytes(receipt["payload"]))
    except InvalidSignature as exc:
        raise ReceiptError("signature does not verify") from exc


def receipt_hash(receipt: dict) -> str:
    """Hash of a whole receipt (payload + signature), used by the run receipt."""
    return sha256_hex(canonical_bytes({"payload": receipt["payload"], "signature": receipt["signature"],
                                       "public_key": receipt["public_key"]}))


# --------------------------------------------------------------------------- payloads


def shard_payload(scope_id: str, docs_in: list[dict], docs_out: list[dict], policy: Policy, tagger: dict,
                  surrogate_key_id: str, signing_key_id: str, report: dict, code: dict | None = None,
                  timestamp: str | None = None) -> dict:
    ids_in = sorted(d["sample_id"] for d in docs_in)
    ids_out = sorted(d["sample_id"] for d in docs_out)
    if ids_in != ids_out:
        raise ReceiptError("input and output document sets differ")
    by_type: dict[str, int] = {}
    for d in docs_out:
        for m in d["mentions"]:
            by_type[m["type"]] = by_type.get(m["type"], 0) + 1
    return {
        "receipt_version": RECEIPT_VERSION,
        "kind": "shard",
        "policy": policy.identity(),
        "code": code or code_info(),
        "shard": {"scope_id": scope_id, "scope_rule": SCOPE_RULE[policy.scope], "documents": ids_in},
        "input": {"sha256": docs_hash(docs_in), "documents": len(docs_in)},
        "output": {"sha256": docs_hash(docs_out), "documents": len(docs_out)},
        "mentions": {"count": sum(by_type.values()), "by_type": dict(sorted(by_type.items()))},
        "tagger": tagger,
        "surrogate_key_id": surrogate_key_id,
        "signing_key_id": signing_key_id,
        "engine_report": report,
        "timestamp_utc": timestamp or utc_now(),
    }


def run_payload(shard_receipts: list[dict], policy: Policy, tagger: dict, surrogate_key_id: str,
                signing_key_id: str, utility: dict, residual: dict, output_sha256: str, code: dict | None = None,
                timestamp: str | None = None) -> dict:
    hashes = [receipt_hash(r) for r in shard_receipts]
    n_docs = sum(r["payload"]["output"]["documents"] for r in shard_receipts)
    return {
        "receipt_version": RECEIPT_VERSION,
        "kind": "run",
        "policy": policy.identity(),
        "code": code or code_info(),
        "shards": {"count": len(shard_receipts), "documents": n_docs, "receipt_sha256s": hashes,
                   "root": sha256_hex("".join(hashes).encode("ascii"))},
        "output_file_sha256": output_sha256,
        "tagger": tagger,
        "surrogate_key_id": surrogate_key_id,
        "signing_key_id": signing_key_id,
        "utility": utility,
        "residual_scan": residual,
        "timestamp_utc": timestamp or utc_now(),
    }


# --------------------------------------------------------------------------- offline verification


def verify_run(run_receipt: dict, shard_receipts: list[dict], output_docs: list[dict], policy_path: str | Path,
               trusted_public_key: Ed25519PublicKey | None = None, output_file: str | Path | None = None) -> dict:
    """Every check a third party can make with the public key, the receipts,
    the output and the policy file. Returns a dict of check -> True; raises
    ReceiptError at the first failure with the check named."""
    checks: dict[str, bool] = {}

    verify_signature(run_receipt, trusted_public_key)
    checks["run_signature"] = True
    rp = run_receipt["payload"]
    if rp.get("kind") != "run" or rp.get("receipt_version") != RECEIPT_VERSION:
        raise ReceiptError("run receipt has the wrong kind or version")

    from .policy import load_policy
    pol = load_policy(policy_path)
    if pol.identity() != rp["policy"]:
        raise ReceiptError(f"policy file does not match the run receipt: {pol.identity()} != {rp['policy']}")
    checks["policy_matches_file"] = True

    hashes = [receipt_hash(r) for r in shard_receipts]
    if hashes != rp["shards"]["receipt_sha256s"]:
        raise ReceiptError("shard receipts do not match the run receipt's list (added, removed or reordered)")
    if sha256_hex("".join(hashes).encode("ascii")) != rp["shards"]["root"]:
        raise ReceiptError("shard root hash mismatch")
    checks["shard_list_matches"] = True

    by_id = {d["sample_id"]: d for d in output_docs}
    if len(by_id) != len(output_docs):
        raise ReceiptError("duplicate sample_id in output")
    seen: set[str] = set()
    for r in shard_receipts:
        verify_signature(r, trusted_public_key)
        p = r["payload"]
        if p.get("kind") != "shard":
            raise ReceiptError("shard receipt has the wrong kind")
        if p["policy"] != rp["policy"] or p["signing_key_id"] != rp["signing_key_id"] \
                or p["surrogate_key_id"] != rp["surrogate_key_id"]:
            raise ReceiptError(f"shard {p['shard']['scope_id']} disagrees with the run receipt on policy or keys")
        docs = []
        for sid in p["shard"]["documents"]:
            if sid not in by_id:
                raise ReceiptError(f"shard {p['shard']['scope_id']}: document {sid} missing from output")
            if sid in seen:
                raise ReceiptError(f"document {sid} appears in two shards")
            seen.add(sid)
            docs.append(by_id[sid])
        if docs_hash(docs) != p["output"]["sha256"]:
            raise ReceiptError(f"shard {p['shard']['scope_id']}: output hash mismatch (output altered after signing)")
        n = sum(len(d["mentions"]) for d in docs)
        if n != p["mentions"]["count"]:
            raise ReceiptError(f"shard {p['shard']['scope_id']}: mention count mismatch")
        for d in docs:
            for m in d["mentions"]:
                if d["text"][m["start"]:m["end"]] != m["text"]:
                    raise ReceiptError(f"{d['sample_id']}: mention offsets do not match text")
    if seen != set(by_id):
        raise ReceiptError(f"{len(set(by_id) - seen)} output documents are covered by no shard receipt")
    checks["shard_signatures"] = True
    checks["output_hashes"] = True
    checks["every_document_covered"] = True

    if output_file is not None:
        if file_sha256(output_file) != rp["output_file_sha256"]:
            raise ReceiptError("output file hash does not match the run receipt")
        checks["output_file_hash"] = True
    if not rp["utility"].get("pass", False):
        raise ReceiptError("run receipt records a failed utility check")
    if rp["residual_scan"].get("hits", 1) != 0:
        raise ReceiptError("run receipt records residual-scan hits")
    checks["utility_pass_recorded"] = True
    checks["residual_scan_clean_recorded"] = True
    return checks


# --------------------------------------------------------------------------- secret handling


def new_surrogate_secret() -> bytes:
    return os.urandom(32)


def save_secret(secret: bytes, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(secret.hex() + "\n", encoding="utf-8")
    os.chmod(p, 0o600)


def load_secret(path: str | Path) -> bytes:
    raw = Path(path).read_text(encoding="utf-8").strip()
    try:
        b = bytes.fromhex(raw)
    except ValueError as exc:
        raise ReceiptError(f"{path}: secret must be hex") from exc
    if len(b) < 16:
        raise ReceiptError(f"{path}: secret must be at least 16 bytes")
    return b
