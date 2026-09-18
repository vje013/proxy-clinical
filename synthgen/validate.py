"""Validation gates. Run in CI order; every gate returns a list of failures
(empty means green). Gate 5 also produces the distribution report and gate 6
is a re-generation byte comparison driven from the CLI.

Gates
  1 offset integrity      text[start:end] == mention.text
  2 entity consistency    mention -> entity exists, type compatible, text is a
                          known surface form of that identity (relative dates
                          verified arithmetically); no duplicate identities
  3 coverage              every cast entity that the plan says is present has a
                          mention; no unlabeled occurrence of any surface form
  4 distractor purity     no mention overlaps a drug / AE / lab / conmed phrase
  5 distribution report   per-type counts, form histograms, hard-case counts,
                          length histogram, duplicate-text check, resamples
  6 determinism           two runs, same seed, byte-identical (see cli.py)
  7 cross-reference       narratives: >=4 mentions, >=3 forms, >=2-paragraph gap
"""
from __future__ import annotations

import datetime as dt
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .cast import NUMBER_WORDS
from .vocab import WEEKDAYS, load_vocab

TYPE_COMPAT: dict[str, set[str]] = {
    "PATIENT": {"PATIENT", "ID", "AGE", "CONTACT"},
    "INVESTIGATOR": {"INVESTIGATOR", "CONTACT"},
    "SITE": {"SITE"},
    "LOCATION": {"LOCATION"},
    "DATE": {"DATE"},
}

_PROSE_RE = re.compile(
    r"^(?P<n>[a-z]+|\d+) (?P<unit>day|days|week|weeks) (?P<dir>after|before) the "
)
_FOLLOWING_RE = re.compile(r"^the following (?P<wd>[A-Z][a-z]+)$")
_DAY_RE = re.compile(r"^Day (?P<n>-?\d+)$")


@dataclass
class GateResult:
    gate: str
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def _boundary_pattern(form: str) -> re.Pattern:
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(form) + r"(?![A-Za-z0-9])", re.IGNORECASE)


def _covered(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    for s, e in spans:
        if s <= start and end <= e:
            return True
    return False


# --------------------------------------------------------------------------- gate 1


def gate_offsets(records: list[dict]) -> GateResult:
    res = GateResult("1 offset integrity")
    for r in records:
        text = r["text"]
        prev_end = -1
        for m in sorted(r["mentions"], key=lambda m: (m["start"], m["end"])):
            if text[m["start"]:m["end"]] != m["text"]:
                res.failures.append(f"{r['sample_id']}: [{m['start']}:{m['end']}] "
                                    f"{text[m['start']:m['end']]!r} != {m['text']!r}")
            if m["start"] < prev_end:
                res.failures.append(f"{r['sample_id']}: overlapping mention at {m['start']}")
            prev_end = max(prev_end, m["end"])
    return res


# --------------------------------------------------------------------------- gate 2


def _numword(s: str) -> int | None:
    if s.isdigit():
        return int(s)
    for k, v in NUMBER_WORDS.items():
        if v == s:
            return k
    return None


def _check_relative_date(m: dict, ent: dict, ents: dict[str, dict], graph: dict[str, dict]) -> str | None:
    attrs = m.get("attrs") or {}
    anchor_id = attrs.get("anchor")
    if not attrs.get("relative") or not anchor_id or anchor_id not in ents:
        return "relative date without a valid anchor"
    target = dt.date.fromisoformat(ent["canonical"])
    anchor = dt.date.fromisoformat(ents[anchor_id]["canonical"])
    delta = (target - anchor).days
    text = m["text"]
    dm = _DAY_RE.match(text)
    if dm:
        n = int(dm.group("n"))
        expect = delta + 1 if delta >= 0 else delta
        return None if n == expect else f"Day arithmetic {n} != {expect}"
    fm = _FOLLOWING_RE.match(text)
    if fm:
        if not (1 <= delta <= 7):
            return f"'following' with delta {delta}"
        return None if WEEKDAYS[target.weekday()] == fm.group("wd") else "weekday mismatch"
    pm = _PROSE_RE.match(text[0].lower() + text[1:])
    if pm:
        n = _numword(pm.group("n"))
        if n is None:
            return f"unparseable number in {text!r}"
        days = n * 7 if pm.group("unit").startswith("week") else n
        if pm.group("dir") == "before":
            days = -days
        return None if days == delta else f"prose interval {days} != {delta}"
    return f"unrecognised relative form {text!r}"


def gate_entities(records: list[dict]) -> GateResult:
    res = GateResult("2 entity consistency")
    for r in records:
        ents = {e["entity_id"]: e for e in r["entities"]}
        graph = {g["entity_id"]: g for g in r["date_graph"]}
        # Identity uniqueness (dates may legitimately coincide).
        seen: dict[tuple[str, str], str] = {}
        for e in r["entities"]:
            if e["type"] == "DATE":
                continue
            k = (e["type"], e["canonical"].lower())
            if k in seen:
                res.failures.append(f"{r['sample_id']}: duplicate identity {k} ({seen[k]}, {e['entity_id']})")
            seen[k] = e["entity_id"]
        for m in r["mentions"]:
            eid = m["entity_id"]
            if eid not in ents:
                res.failures.append(f"{r['sample_id']}: mention {m['text']!r} -> unknown {eid}")
                continue
            ent = ents[eid]
            if m["type"] not in TYPE_COMPAT[ent["type"]]:
                res.failures.append(f"{r['sample_id']}: {m['type']} mention on {ent['type']} entity {eid}")
            if ent["type"] == "DATE" and (m.get("attrs") or {}).get("relative"):
                err = _check_relative_date(m, ent, ents, graph)
                if err:
                    res.failures.append(f"{r['sample_id']}: {eid} {m['text']!r}: {err}")
                continue
            known = [f.lower() for f in ent["forms"]] + [f.lower() for f in ent.get("no_scan_forms", [])]
            if m["text"].lower() not in known:
                res.failures.append(f"{r['sample_id']}: {m['text']!r} is not a surface form of {eid} "
                                    f"({ent['canonical']})")
        # Date graph arithmetic.
        for g in r["date_graph"]:
            if g["anchor"] is None:
                continue
            if g["entity_id"] not in ents or g["anchor"] not in ents:
                res.failures.append(f"{r['sample_id']}: date graph references unknown entity")
                continue
            d = dt.date.fromisoformat(ents[g["entity_id"]]["canonical"])
            a = dt.date.fromisoformat(ents[g["anchor"]]["canonical"])
            if (d - a).days != g["interval_days"]:
                res.failures.append(f"{r['sample_id']}: interval mismatch for {g['entity_id']}")
    return res


# --------------------------------------------------------------------------- gate 3


def gate_coverage(records: list[dict]) -> GateResult:
    res = GateResult("3 coverage")
    for r in records:
        text = r["text"]
        spans = [(m["start"], m["end"]) for m in r["mentions"]]
        # Every planted identity (patient / investigator / site / location) is
        # mentioned. Emit drops an unmentioned sub-investigator from the cast
        # record, so anything left here with zero mentions is a real miss.
        for e in r["entities"]:
            if e["type"] in ("PATIENT", "INVESTIGATOR", "SITE", "LOCATION") and e["mention_count"] == 0:
                res.failures.append(f"{r['sample_id']}: planted {e['type']} {e['entity_id']} never mentioned")
        # No unlabeled occurrence of any surface form.
        for e in r["entities"]:
            for form in e["forms"]:
                for hit in _boundary_pattern(form).finditer(text):
                    if not _covered(hit.start(), hit.end(), spans):
                        res.failures.append(f"{r['sample_id']}: unlabeled {form!r} at {hit.start()} "
                                            f"({e['type']} {e['entity_id']})")
    return res


# --------------------------------------------------------------------------- gate 4


def gate_distractors(records: list[dict]) -> GateResult:
    res = GateResult("4 distractor purity")
    v = load_vocab()
    patterns = [(p, _boundary_pattern(p)) for p in v.distractor_phrases]
    for r in records:
        text = r["text"]
        ms = r["mentions"]
        for m in ms:
            low = m["text"].lower()
            if low in {p.lower() for p in v.distractor_phrases}:
                res.failures.append(f"{r['sample_id']}: mention {m['text']!r} equals a distractor")
        for phrase, pat in patterns:
            for hit in pat.finditer(text):
                for m in ms:
                    if m["start"] < hit.end() and hit.start() < m["end"]:
                        res.failures.append(f"{r['sample_id']}: mention {m['text']!r} overlaps distractor "
                                            f"{phrase!r} at {hit.start()}")
    return res


# --------------------------------------------------------------------------- gate 7


def gate_cross_reference(records: list[dict]) -> GateResult:
    res = GateResult("7 cross-reference pattern")
    for r in records:
        if r["doc_type"] != "narrative":
            continue
        paras = _paragraph_index(r["text"])
        by_entity: dict[str, list[dict]] = defaultdict(list)
        for m in r["mentions"]:
            if m["type"] in ("PATIENT", "ID"):
                by_entity[m["entity_id"]].append(m)
        patients = [e for e in r["entities"] if e["type"] == "PATIENT"]
        for e in patients:
            ms = by_entity.get(e["entity_id"], [])
            if len(ms) < 4:
                res.failures.append(f"{r['sample_id']}: {e['entity_id']} has {len(ms)} name-like mentions")
                continue
            forms = {m["text"].lower() for m in ms}
            if len(forms) < 3:
                res.failures.append(f"{r['sample_id']}: {e['entity_id']} has {len(forms)} distinct forms")
            ps = sorted({paras[m["start"]] for m in ms})
            if ps[-1] - ps[0] < 2:
                res.failures.append(f"{r['sample_id']}: {e['entity_id']} mentions span paragraphs {ps}")
    return res


def _paragraph_index(text: str) -> dict[int, int]:
    """Character offset -> paragraph number (paragraphs split on blank lines)."""
    idx: dict[int, int] = {}
    para = 0
    pos = 0
    for chunk in text.split("\n\n"):
        for i in range(len(chunk)):
            idx[pos + i] = para
        pos += len(chunk) + 2
        para += 1
    return idx


# --------------------------------------------------------------------------- gate 5 (report)


def distribution_report(records: list[dict], meta: dict, gate_results: list[GateResult],
                        determinism: str | None = None) -> str:
    n = len(records)
    by_type = Counter(m["type"] for r in records for m in r["mentions"])
    doc_types = Counter(r["doc_type"] for r in records)
    slices = Counter(r.get("slice", "?") for r in records)
    hard = Counter(k for r in records for k in r["hard_case_kinds"])
    hard_docs = sum(1 for r in records if r["hard_case"])
    words = [r["stats"]["words"] for r in records]
    narr_words = [r["stats"]["words"] for r in records if r["doc_type"] == "narrative"]
    total_words = sum(words)
    dup = n - len({r["text"] for r in records})
    templates = Counter(r["template_id"] for r in records)
    locales = Counter(r["locale"] for r in records)
    resamples = sum(r["stats"]["cast_resamples"] for r in records)
    multi = sum(1 for r in records if r["doc_type"] == "narrative" and r["stats"]["patients"] >= 2)
    narratives = sum(1 for r in records if r["doc_type"] == "narrative")
    bundles = len({r["bundle_id"] for r in records if r["bundle_id"]})

    # Surface forms per patient (narratives only).
    fp_hist: Counter = Counter()
    mp_hist: Counter = Counter()
    for r in records:
        if r["doc_type"] != "narrative":
            continue
        per: dict[str, set[str]] = defaultdict(set)
        cnt: Counter = Counter()
        for m in r["mentions"]:
            if m["type"] in ("PATIENT", "ID"):
                per[m["entity_id"]].add(m["text"].lower())
                cnt[m["entity_id"]] += 1
        for eid, forms in per.items():
            fp_hist[len(forms)] += 1
            mp_hist[min(cnt[eid], 12)] += 1

    def hist_lines(counter: Counter, label: str) -> list[str]:
        out = [f"| {label} | count |", "|---|---|"]
        for k in sorted(counter):
            out.append(f"| {k} | {counter[k]} |")
        return out

    length_bins: Counter = Counter()
    for w in narr_words:
        length_bins[f"{(w // 50) * 50}-{(w // 50) * 50 + 49}"] += 1

    lines: list[str] = []
    lines.append("# Proxy Clinical synthetic pilot report")
    lines.append("")
    lines.append(f"Generator `synthgen {meta.get('generator_version')}`, master seed `{meta.get('master_seed')}`, "
                 f"instruction version `{meta.get('instruction_version')}`.")
    lines.append("")
    lines.append("## Gates")
    lines.append("")
    lines.append("| Gate | Result | Failures |")
    lines.append("|---|---|---|")
    for g in gate_results:
        lines.append(f"| {g.gate} | {'PASS' if g.ok else 'FAIL'} | {len(g.failures)} |")
    if determinism is not None:
        lines.append(f"| 6 determinism | {determinism} | |")
    lines.append("")
    lines.append("## Composition")
    lines.append("")
    lines.append(f"- Samples: {n} ({', '.join(f'{k}: {v}' for k, v in sorted(doc_types.items()))})")
    lines.append(f"- Slices: {', '.join(f'{k}: {v}' for k, v in sorted(slices.items()))}")
    lines.append(f"- Bundles: {bundles} (narrative+listing pairs sharing entity ids)")
    lines.append(f"- Multi-patient narratives: {multi} of {narratives} "
                 f"({100.0 * multi / max(narratives, 1):.1f}%)")
    lines.append(f"- Hard-case flagged samples: {hard_docs}")
    lines.append(f"- Locales: {', '.join(f'{k}: {v}' for k, v in sorted(locales.items()))}")
    lines.append(f"- Total words: {total_words:,}; narrative words: min {min(narr_words) if narr_words else 0}, "
                 f"median {sorted(narr_words)[len(narr_words) // 2] if narr_words else 0}, "
                 f"max {max(narr_words) if narr_words else 0}")
    lines.append(f"- Duplicate texts: {dup}")
    lines.append(f"- Cast resamples (Faker name collided with vocab, template words or another cast member): "
                 f"{resamples} across {n} casts")
    lines.append("")
    lines.append("## Hard-case kinds")
    lines.append("")
    lines.extend(hist_lines(hard, "kind"))
    lines.append("")
    lines.append("## Mentions per type")
    lines.append("")
    lines.extend(hist_lines(by_type, "type"))
    lines.append("")
    lines.append("## Distinct surface forms per patient (narratives)")
    lines.append("")
    lines.extend(hist_lines(fp_hist, "distinct forms"))
    lines.append("")
    lines.append("## Name-like mentions per patient (narratives, capped at 12)")
    lines.append("")
    lines.extend(hist_lines(mp_hist, "mentions"))
    lines.append("")
    lines.append("## Narrative length histogram (words)")
    lines.append("")
    lines.extend(hist_lines(length_bins, "words"))
    lines.append("")
    lines.append("## Templates")
    lines.append("")
    lines.extend(hist_lines(templates, "template"))
    lines.append("")
    failing = [g for g in gate_results if not g.ok]
    if failing:
        lines.append("## Failure detail (first 50 per gate)")
        lines.append("")
        for g in failing:
            lines.append(f"### {g.gate}")
            lines.append("")
            for f in g.failures[:50]:
                lines.append(f"- {f}")
            lines.append("")
    return "\n".join(lines) + "\n"


def run_all_gates(records: list[dict]) -> list[GateResult]:
    return [
        gate_offsets(records),
        gate_entities(records),
        gate_coverage(records),
        gate_distractors(records),
        gate_cross_reference(records),
    ]
