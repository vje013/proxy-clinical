"""Corpus records, training view, and the deterministic train/val split."""
from __future__ import annotations

import hashlib
import json
from typing import Iterable

from .cast import Cast
from .render import Rendered

INSTRUCTION_VERSION = "v1"
INSTRUCTION = (
    "Extract every person, site, location, date, identifier, age and contact mention "
    "from the clinical text. Return JSON {\"mentions\":[{\"span\":[start,end],\"type\":T,\"id\":E}]} "
    "with character offsets, sorted by start. Mentions of the same real-world entity share one id."
)

_FORMAT_ORDER = ("PATIENT", "INVESTIGATOR", "SITE", "LOCATION", "DATE", "ID", "AGE", "CONTACT")


# --------------------------------------------------------------------------- entity ids


def assign_entity_ids(cast: Cast, rendered_docs: list[Rendered],
                      existing: dict[str, str] | None = None) -> dict[str, str]:
    """Map cast keys to ``E1, E2, ...`` ordered by first mention across the
    given documents (in order), then unmentioned cast entities in cast order.
    ``existing`` lets a bundle's listing extend the narrative's numbering."""
    key_to_id: dict[str, str] = dict(existing or {})
    next_n = len(key_to_id) + 1
    for doc in rendered_docs:
        for m in sorted(doc.mentions, key=lambda m: m.start):
            if m.entity_key not in key_to_id:
                key_to_id[m.entity_key] = f"E{next_n}"
                next_n += 1
    for rec in cast.entity_records():
        if rec["key"] not in key_to_id:
            key_to_id[rec["key"]] = f"E{next_n}"
            next_n += 1
    return key_to_id


# --------------------------------------------------------------------------- records


def build_record(sample_id: str, cast: Cast, rendered: Rendered, key_to_id: dict[str, str],
                 hard_case_kinds: list[str], bundle_id: str | None, slice_name: str) -> dict:
    mentions = sorted(rendered.mentions, key=lambda m: (m.start, m.end))
    counts: dict[str, int] = {}
    for m in mentions:
        counts[m.entity_key] = counts.get(m.entity_key, 0) + 1

    entities: list[dict] = []
    for rec in cast.entity_records():
        if rec["type"] in ("INVESTIGATOR", "LOCATION") and counts.get(rec["key"], 0) == 0:
            # A sub-investigator or a location the layout never used is not
            # in this document (listing titles do not always carry them).
            continue
        e: dict = {
            "entity_id": key_to_id[rec["key"]],
            "type": rec["type"],
            "canonical": rec["canonical"],
            "forms": list(rec["forms"]),
            "mention_count": counts.get(rec["key"], 0),
        }
        if rec["type"] == "DATE":
            e["role"] = rec["role"]
            e["patient"] = key_to_id[rec["patient"]]
        entities.append(e)
    entities.sort(key=lambda e: int(e["entity_id"][1:]))

    # A bare age number (listing "Age" column) is a legitimate AGE mention for
    # gate 2 but must not be scanned by gate 3, so it is kept apart from forms.
    for e in entities:
        if e["type"] == "PATIENT":
            p = next(p for p in cast.patients if key_to_id[p.key] == e["entity_id"])
            e["no_scan_forms"] = [str(p.age)]
        elif e["type"] == "SITE":
            e["no_scan_forms"] = [cast.site.number]

    date_graph: list[dict] = []
    for node in cast.date_nodes():
        date_graph.append({
            "entity_id": key_to_id[node.key],
            "role": node.role,
            "patient": key_to_id[node.patient_key],
            "anchor": key_to_id[node.anchor_key] if node.anchor_key else None,
            "interval_days": node.interval_days,
        })

    return {
        "sample_id": sample_id,
        "doc_type": rendered.doc_type,
        "slice": slice_name,
        "hard_case": bool(hard_case_kinds),
        "hard_case_kinds": list(hard_case_kinds),
        "bundle_id": bundle_id,
        "seed": cast.seed_hex,
        "template_id": rendered.template_id,
        "locale": cast.locale,
        "text": rendered.text,
        "entities": entities,
        "mentions": [m.to_json(key_to_id) for m in mentions],
        "date_graph": date_graph,
        "stats": {
            "words": rendered.words,
            "paragraphs": rendered.paragraph_count,
            "patients": len(cast.patients),
            "cast_resamples": cast.resamples,
        },
    }


# --------------------------------------------------------------------------- training view


def renumber_for_training(mentions: list[dict]) -> list[dict]:
    """Per-sample ids by first appearance, so a listing from a bundle still
    starts at E1 when seen on its own."""
    remap: dict[str, str] = {}
    out: list[dict] = []
    for m in sorted(mentions, key=lambda m: (m["start"], m["end"])):
        eid = m["entity_id"]
        if eid not in remap:
            remap[eid] = f"E{len(remap) + 1}"
        out.append({"span": [m["start"], m["end"]], "type": m["type"], "id": remap[eid]})
    return out


def training_pair(record: dict) -> dict:
    target = {"mentions": renumber_for_training(record["mentions"])}
    return {
        "sample_id": record["sample_id"],
        "instruction_version": INSTRUCTION_VERSION,
        "instruction": INSTRUCTION,
        "input": record["text"],
        "output": json.dumps(target, separators=(",", ":"), ensure_ascii=False),
    }


def split_key(record: dict) -> str:
    return record["bundle_id"] or record["sample_id"]


def is_val(record: dict, val_share: float = 0.10) -> bool:
    h = hashlib.sha256(split_key(record).encode("utf-8")).digest()
    return (int.from_bytes(h[:8], "big") % 10_000) < int(val_share * 10_000)


# --------------------------------------------------------------------------- io


def dumps_record(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"))


def write_jsonl(path, records: Iterable[dict]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(dumps_record(r))
            fh.write("\n")
            n += 1
    return n


def read_jsonl(path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
