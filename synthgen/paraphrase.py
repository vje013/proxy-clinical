"""Optional LLM paraphrase pass (``--paraphrase``). OFF by default.

The paraphraser must keep every labelled span verbatim and in order. On
return, every mention is re-located by exact string match in order of prior
occurrence; if any span fails to relocate, if the count of any span text
changed, or if the relocated record fails the coverage / distractor gates, the
paraphrase is discarded and the template rendering is kept. Keep/discard
counts are returned for the report.

Only narratives are paraphrased. Listings are tables and stay as rendered.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable

from .validate import gate_coverage, gate_distractors, gate_offsets

Paraphraser = Callable[[str, list[str]], str]


class ParaphraseStats:
    def __init__(self) -> None:
        self.attempted = 0
        self.kept = 0
        self.discarded = 0
        self.reasons: Counter = Counter()

    def as_dict(self) -> dict:
        return {"attempted": self.attempted, "kept": self.kept, "discarded": self.discarded,
                "discard_reasons": dict(self.reasons)}


def relocate(record: dict, new_text: str) -> tuple[dict | None, str]:
    """Return (new_record, "") or (None, reason)."""
    old_text = record["text"]
    mentions = sorted(record["mentions"], key=lambda m: (m["start"], m["end"]))
    # Substring counts must be preserved (a first name also occurs inside the
    # full name, so compare against the old text, not the mention count).
    for text in {m["text"] for m in mentions}:
        if new_text.count(text) != old_text.count(text):
            return None, f"count changed for {text!r}"
    cursor = 0
    new_mentions: list[dict] = []
    for m in mentions:
        idx = new_text.find(m["text"], cursor)
        if idx < 0:
            return None, f"could not relocate {m['text']!r}"
        nm = dict(m)
        nm["start"] = idx
        nm["end"] = idx + len(m["text"])
        new_mentions.append(nm)
        cursor = nm["end"]
    new_rec = dict(record)
    new_rec["text"] = new_text
    new_rec["mentions"] = new_mentions
    new_rec["stats"] = dict(record["stats"])
    new_rec["stats"]["words"] = len(new_text.split())
    new_rec["stats"]["paragraphs"] = new_text.count("\n\n") + 1
    new_rec["paraphrased"] = True
    for gate in (gate_offsets, gate_coverage, gate_distractors):
        res = gate([new_rec])
        if not res.ok:
            return None, f"{res.gate}: {res.failures[0]}"
    return new_rec, ""


def apply_paraphrase(records: list[dict], paraphraser: Paraphraser) -> tuple[list[dict], ParaphraseStats]:
    stats = ParaphraseStats()
    out: list[dict] = []
    for r in records:
        if r["doc_type"] != "narrative":
            out.append(r)
            continue
        stats.attempted += 1
        protected = [m["text"] for m in sorted(r["mentions"], key=lambda m: m["start"])]
        try:
            new_text = paraphraser(r["text"], protected)
        except Exception as exc:  # network / API failure keeps the template version
            stats.discarded += 1
            stats.reasons[f"error: {type(exc).__name__}"] += 1
            out.append(r)
            continue
        new_rec, reason = relocate(r, new_text)
        if new_rec is None:
            stats.discarded += 1
            stats.reasons[reason.split(":")[0]] += 1
            out.append(r)
        else:
            stats.kept += 1
            out.append(new_rec)
    return out, stats


_PROMPT = (
    "Rewrite the clinical study report narrative below with varied phrasing and sentence "
    "structure while keeping its meaning, its clinical register, and its paragraph breaks. "
    "Every string in PROTECTED must appear verbatim, exactly as many times as in the "
    "original, and in the same order. Do not add, remove, reorder or alter any of them. "
    "Return only the rewritten text.\n\nPROTECTED:\n{protected}\n\nTEXT:\n{text}"
)


def anthropic_paraphraser(model: str = "claude-sonnet-4-5") -> Paraphraser:
    """Builds a paraphraser backed by the Anthropic API. Imported lazily so the
    required path never depends on the SDK or the network."""
    import anthropic  # type: ignore[import-not-found]

    client = anthropic.Anthropic()

    def run(text: str, protected: list[str]) -> str:
        prompt = _PROMPT.format(protected="\n".join(protected), text=text)
        resp = client.messages.create(
            model=model, max_tokens=2048, temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(getattr(block, "text", "") for block in resp.content).strip()

    return run
