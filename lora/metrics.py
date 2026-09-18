"""Pure metric functions. No model code here so they are trivially testable.

Parsing is strict on purpose. A prediction is either a well-formed
``{"mentions": [{"span": [s, e], "type": T, "id": E}, ...]}`` or it is
malformed, in which case every gold mention of that sample counts as missed
and every predicted span counts as zero. No repair heuristics: a repaired
number is a number that lies about production behaviour.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field

Span = tuple[int, int, str]   # (start, end, type)


class MalformedPrediction(ValueError):
    pass


def parse_prediction(raw: str, text_len: int | None = None) -> list[dict]:
    """Strict parse. Returns a list of {"start","end","type","id"} or raises."""
    if not isinstance(raw, str):
        raise MalformedPrediction("prediction is not a string")
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MalformedPrediction(f"invalid JSON: {exc.msg} at {exc.pos}") from exc
    if not isinstance(obj, dict) or set(obj.keys()) != {"mentions"}:
        raise MalformedPrediction("top level must be an object with exactly one key 'mentions'")
    ms = obj["mentions"]
    if not isinstance(ms, list):
        raise MalformedPrediction("'mentions' must be a list")
    out: list[dict] = []
    for i, m in enumerate(ms):
        if not isinstance(m, dict) or set(m.keys()) != {"span", "type", "id"}:
            raise MalformedPrediction(f"mention {i}: must have exactly keys span,type,id")
        span = m["span"]
        if (not isinstance(span, list) or len(span) != 2
                or not all(isinstance(x, int) and not isinstance(x, bool) for x in span)):
            raise MalformedPrediction(f"mention {i}: span must be [int, int]")
        s, e = span
        if s < 0 or e <= s or (text_len is not None and e > text_len):
            raise MalformedPrediction(f"mention {i}: span [{s},{e}] out of range")
        if not isinstance(m["type"], str) or not m["type"]:
            raise MalformedPrediction(f"mention {i}: type must be a non-empty string")
        if not isinstance(m["id"], str) or not m["id"]:
            raise MalformedPrediction(f"mention {i}: id must be a non-empty string")
        out.append({"start": s, "end": e, "type": m["type"], "id": m["id"]})
    return out


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: "Counts") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn,
                "precision": round(self.precision, 4), "recall": round(self.recall, 4), "f1": round(self.f1, 4)}


def span_counts(gold: list[dict], pred: list[dict] | None) -> tuple[Counts, dict[str, Counts]]:
    """Exact-match (start, end, type). ``pred=None`` means malformed: all gold
    spans are misses and nothing is credited."""
    per_type: dict[str, Counts] = defaultdict(Counts)
    total = Counts()
    gold_set = {(g["start"], g["end"], g["type"]) for g in gold}
    if pred is None:
        for s, e, t in gold_set:
            per_type[t].fn += 1
        total.fn = len(gold_set)
        return total, dict(per_type)
    pred_set = {(p["start"], p["end"], p["type"]) for p in pred}
    for sp in gold_set & pred_set:
        per_type[sp[2]].tp += 1
    for sp in gold_set - pred_set:
        per_type[sp[2]].fn += 1
    for sp in pred_set - gold_set:
        per_type[sp[2]].fp += 1
    for c in per_type.values():
        total.add(c)
    return total, dict(per_type)


@dataclass
class ConsistencyResult:
    entities: int = 0            # gold entities with >= 1 mention
    consistent: int = 0          # all mentions found, one predicted id, id not shared with another gold entity
    split: int = 0               # mentions found but spread over > 1 predicted id
    merged: int = 0              # predicted id also used for another gold entity's mentions
    incomplete: int = 0          # at least one gold mention span not predicted at all
    detail: dict[str, str] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.consistent / self.entities if self.entities else 0.0

    def add(self, other: "ConsistencyResult") -> None:
        self.entities += other.entities
        self.consistent += other.consistent
        self.split += other.split
        self.merged += other.merged
        self.incomplete += other.incomplete

    def as_dict(self) -> dict:
        return {"entities": self.entities, "consistent": self.consistent, "split": self.split,
                "merged": self.merged, "incomplete": self.incomplete, "accuracy": round(self.accuracy, 4)}


def entity_consistency(gold: list[dict], pred: list[dict] | None) -> ConsistencyResult:
    """Per gold entity: every gold mention span must be predicted (any type is
    fine here; span F1 already scores types), all of them under a single
    predicted id, and that id must not appear on any mention of a different
    gold entity. Matching is by exact span."""
    res = ConsistencyResult()
    gold_by_entity: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for g in gold:
        gold_by_entity[g["entity_id"]].append((g["start"], g["end"]))
    res.entities = len(gold_by_entity)
    if pred is None:
        res.incomplete = res.entities
        for eid in gold_by_entity:
            res.detail[eid] = "incomplete"
        return res
    pred_id_by_span: dict[tuple[int, int], str] = {}
    for p in pred:
        pred_id_by_span.setdefault((p["start"], p["end"]), p["id"])
    span_to_gold: dict[tuple[int, int], str] = {}
    for eid, spans in gold_by_entity.items():
        for sp in spans:
            span_to_gold[sp] = eid
    # Which gold entities does each predicted id touch (via gold spans only)?
    pred_id_to_gold: dict[str, set[str]] = defaultdict(set)
    for sp, pid in pred_id_by_span.items():
        if sp in span_to_gold:
            pred_id_to_gold[pid].add(span_to_gold[sp])
    for eid, spans in gold_by_entity.items():
        pids = [pred_id_by_span.get(sp) for sp in spans]
        if any(p is None for p in pids):
            res.incomplete += 1
            res.detail[eid] = "incomplete"
            continue
        distinct = set(pids)
        if len(distinct) > 1:
            res.split += 1
            res.detail[eid] = "split"
            continue
        pid = next(iter(distinct))
        if len(pred_id_to_gold[pid]) > 1:
            res.merged += 1
            res.detail[eid] = "merged"
            continue
        res.consistent += 1
        res.detail[eid] = "consistent"
    return res
