"""Evaluation: span F1 (exact offsets and type) and entity-consistency
accuracy, reported overall and per slice.

    python -m lora.evaluate --predictions runs/pilot/predictions.jsonl \
        --corpus data/pilot/corpus.jsonl --out runs/pilot/eval

Slices: single, multi, listing, bundle, hard_case (overlapping), all. Gold
comes from corpus.jsonl (mentions with entity ids); the training view is only
what the model saw. A malformed prediction counts every gold mention of that
sample as missed. There is no repair path.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .metrics import (
    CONTRACTS, ConsistencyResult, Counts, MalformedPrediction, entity_consistency, parse_and_locate, span_counts,
)

SLICE_ORDER = ("all", "single", "multi", "listing", "bundle", "hard_case")


def slices_of(record: dict) -> list[str]:
    out = ["all"]
    s = record.get("slice", "")
    if s == "single":
        out.append("single")
    elif s == "multi":
        out.append("multi")
    elif s == "listing":
        out.append("listing")
    elif s.startswith("bundle"):
        out.append("bundle")
    if record.get("hard_case"):
        out.append("hard_case")
    return out


class SliceAgg:
    def __init__(self) -> None:
        self.samples = 0
        self.malformed = 0
        self.unfinished = 0
        self.unlocatable = 0          # v2: predicted mentions whose text/occurrence is not in the document
        self.predicted = 0
        self.spans = Counts()
        self.per_type: dict[str, Counts] = defaultdict(Counts)
        self.consistency = ConsistencyResult()
        self.malformed_reasons: Counter = Counter()

    def as_dict(self) -> dict:
        return {
            "samples": self.samples,
            "malformed": self.malformed,
            "malformed_rate": round(self.malformed / self.samples, 4) if self.samples else 0.0,
            "unfinished": self.unfinished,
            "predicted_mentions": self.predicted,
            "unlocatable": self.unlocatable,
            "unlocatable_rate": round(self.unlocatable / self.predicted, 4) if self.predicted else 0.0,
            "span": self.spans.as_dict(),
            "per_type": {t: c.as_dict() for t, c in sorted(self.per_type.items())},
            "entity_consistency": self.consistency.as_dict(),
            "malformed_reasons": dict(self.malformed_reasons.most_common(5)),
        }


_MENTION_PREFIX = re.compile(r"^mention \d+: ")


def malformed_reason(exc: Exception) -> str:
    """Bucket label for a MalformedPrediction: the message without the mention
    index and without detail after the first ':' or ' ('. 'mention 12: unknown
    type 'DRUG' (closed set: ...)' -> "unknown type 'DRUG'"."""
    msg = _MENTION_PREFIX.sub("", str(exc))
    cut = len(msg)
    for sep in (":", " ("):
        i = msg.find(sep)
        if i != -1:
            cut = min(cut, i)
    return msg[:cut].strip() or "malformed"


def evaluate(predictions: list[dict], corpus: dict[str, dict], contract: str = "v2") -> dict:
    if contract not in CONTRACTS:
        raise SystemExit(f"unknown contract {contract!r}; expected one of {CONTRACTS}")
    aggs: dict[str, SliceAgg] = {k: SliceAgg() for k in SLICE_ORDER}
    hard_kind_cons: dict[str, ConsistencyResult] = defaultdict(ConsistencyResult)
    per_sample: list[dict] = []
    missing = [p["sample_id"] for p in predictions if p["sample_id"] not in corpus]
    if missing:
        raise SystemExit(f"{len(missing)} predicted sample_ids are not in the corpus (e.g. {missing[:3]})")

    for p in predictions:
        rec = corpus[p["sample_id"]]
        gold = rec["mentions"]
        try:
            pred, unloc_list = parse_and_locate(p["raw_output"], rec["text"], contract)
            reason = None
        except MalformedPrediction as exc:
            pred = None
            unloc_list = []
            reason = malformed_reason(exc)
        unloc = len(unloc_list)
        total, per_type = span_counts(gold, pred)
        for u in unloc_list:                      # hallucinated / miscounted anchors are false positives
            total.fp += 1
            per_type.setdefault(u["type"], Counts()).fp += 1
        cons = entity_consistency(gold, pred)
        for s in slices_of(rec):
            a = aggs[s]
            a.samples += 1
            if pred is None:
                a.malformed += 1
                a.malformed_reasons[reason or "malformed"] += 1
            if not p.get("finished", True):
                a.unfinished += 1
            a.unlocatable += unloc
            a.predicted += (len(pred) if pred is not None else 0) + unloc
            a.spans.add(total)
            for t, c in per_type.items():
                a.per_type[t].add(c)
            a.consistency.add(cons)
        for kind in rec.get("hard_case_kinds", []):
            hard_kind_cons[kind].add(cons)
        per_sample.append({
            "sample_id": p["sample_id"], "slice": rec.get("slice"), "hard_case_kinds": rec.get("hard_case_kinds", []),
            "malformed": pred is None, "finished": p.get("finished", True),
            "span_f1": round(total.f1, 4), "consistency": cons.as_dict(),
        })

    return {
        "contract": contract,
        "slices": {k: aggs[k].as_dict() for k in SLICE_ORDER if aggs[k].samples},
        "hard_case_kinds": {k: v.as_dict() for k, v in sorted(hard_kind_cons.items())},
        "per_sample": per_sample,
    }


def render_report(result: dict, meta: dict) -> str:
    L: list[str] = []
    L.append("# Proxy Clinical tagger evaluation")
    L.append("")
    L.append(f"Adapter `{meta.get('adapter_dir')}` (sha256 `{str(meta.get('adapter_sha256'))[:16]}...`), "
             f"predictions `{meta.get('predictions')}`, corpus `{meta.get('corpus')}`.")
    env = meta.get("environment") or {}
    L.append(f"Output contract: {result.get('contract')}. ")
    L.append(f"Decoding: greedy, max_new_tokens {meta.get('max_new_tokens')}, batch {meta.get('batch_size')}; "
             f"GPU {env.get('gpu')}; torch {env.get('torch')}, transformers {env.get('transformers')}, "
             f"peft {env.get('peft')}, trl {env.get('trl')}.")
    L.append("")
    if result.get("contract") == "v2":
        L.append("v2 predictions are text anchors (exact string + occurrence index) relocated to offsets "
                 "deterministically before scoring; a mention whose string/occurrence is not in the document is "
                 "'unlocatable' and scores as a miss. No repair is applied.")
        L.append("")
    L.append("Span F1 is exact-match on (start, end, type). Entity consistency counts a gold entity as consistent "
             "only when every one of its mention spans was predicted, all under a single predicted id, and that id "
             "is not used for any other gold entity's mentions. A malformed output counts every gold mention of "
             "the sample as missed and every gold entity as incomplete.")
    L.append("")
    L.append("## Per slice")
    L.append("")
    L.append("| slice | samples | malformed | unfinished | unlocatable | span P | span R | span F1 | entity acc | consistent | split | merged | incomplete |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, s in result["slices"].items():
        sp, ec = s["span"], s["entity_consistency"]
        L.append(f"| {k} | {s['samples']} | {s['malformed']} ({100 * s['malformed_rate']:.0f}%) | {s['unfinished']} | "
                 f"{s['unlocatable']}/{s['predicted_mentions']} | "
                 f"{sp['precision']:.3f} | {sp['recall']:.3f} | {sp['f1']:.3f} | {ec['accuracy']:.3f} | "
                 f"{ec['consistent']}/{ec['entities']} | {ec['split']} | {ec['merged']} | {ec['incomplete']} |")
    L.append("")
    L.append("## Per type (all samples)")
    L.append("")
    L.append("| type | tp | fp | fn | P | R | F1 |")
    L.append("|---|---|---|---|---|---|---|")
    for t, c in result["slices"].get("all", {}).get("per_type", {}).items():
        L.append(f"| {t} | {c['tp']} | {c['fp']} | {c['fn']} | {c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f} |")
    if result["hard_case_kinds"]:
        L.append("")
        L.append("## Entity consistency by hard-case kind")
        L.append("")
        L.append("| kind | entities | consistent | split | merged | incomplete | acc |")
        L.append("|---|---|---|---|---|---|---|")
        for k, c in result["hard_case_kinds"].items():
            L.append(f"| {k} | {c['entities']} | {c['consistent']} | {c['split']} | {c['merged']} | {c['incomplete']} | {c['accuracy']:.3f} |")
    reasons = result["slices"].get("all", {}).get("malformed_reasons", {})
    if reasons:
        L.append("")
        L.append("## Malformed output reasons (top 5)")
        L.append("")
        for r, n in reasons.items():
            L.append(f"- {r}: {n}")
    L.append("")
    return "\n".join(L) + "\n"


def load_predictions(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_corpus(path: str | Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[r["sample_id"]] = r
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Span F1 and entity-consistency evaluation")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True, help="output prefix: writes <out>_report.md and <out>.json")
    ap.add_argument("--contract", default=None, choices=list(CONTRACTS),
                    help="output contract to parse (default: the instruction_version recorded in the predictions manifest)")
    args = ap.parse_args(argv)
    preds = load_predictions(args.predictions)
    corpus = load_corpus(args.corpus)
    manifest_path = Path(args.predictions + ".manifest.json")
    pm = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    contract = args.contract or pm.get("instruction_version")
    if contract is None:
        raise SystemExit("cannot determine the output contract: pass --contract or evaluate predictions that have a manifest")
    result = evaluate(preds, corpus, contract)
    meta = {
        "adapter_dir": pm.get("adapter_dir"), "adapter_sha256": pm.get("adapter_sha256"),
        "predictions": args.predictions, "corpus": args.corpus,
        "max_new_tokens": (pm.get("decoding") or {}).get("max_new_tokens"),
        "batch_size": (pm.get("decoding") or {}).get("batch_size"),
        "environment": pm.get("environment"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(out) + "_report.md").write_text(render_report(result, meta), encoding="utf-8")
    Path(str(out) + ".json").write_text(json.dumps({"meta": meta, **result}, indent=2, default=str) + "\n",
                                         encoding="utf-8")
    s = result["slices"]["all"]
    print(f"[evaluate] contract {contract}; all: samples {s['samples']}, malformed {s['malformed']}, "
          f"unlocatable {s['unlocatable']}/{s['predicted_mentions']}, span F1 {s['span']['f1']:.3f}, "
          f"entity acc {s['entity_consistency']['accuracy']:.3f}; report at {out}_report.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
