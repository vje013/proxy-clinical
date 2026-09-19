"""Utility check: rebuild the adverse-event timeline from the *surrogated*
text and diff it against the corpus date graph. Zero tolerance.

What is reconstructed, from the output text alone (offsets from the output
mentions, values by parsing the output strings):

  absolute dates   every absolute DATE mention parses, and within a scope all
                   of them moved by one offset D, a whole number of weeks in
                   the policy's window
  entity dates     every DATE entity's absolute mentions agree on one value,
                   equal to gold + D
  intervals        for every edge (target, anchor, interval_days) of the gold
                   date graph, reconstructed target - anchor == interval_days
  relative prose   "Day N", "the following <weekday>", "<n> days after the
                   <Month> visit" still resolve to gold + D against the
                   reconstructed anchor; a month word is the shifted anchor's
                   month, or was dropped by policy and reported
  ages             unchanged under the threshold, capped at and above it
  structure        same mentions (count, order, type, entity id); text outside
                   mentions byte-identical

Any mismatch is a failure. There is no tolerance parameter.
"""
from __future__ import annotations

import datetime as dt
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from synthgen.vocab import MONTHS, WEEKDAYS

from .policy import Policy
from .surrogate import classify_relative, parse_absolute_date

_DAY_RE = re.compile(r"^Day (-?\d+)$")
_FOLLOWING_RE = re.compile(r"^the following ([A-Z][a-z]+)$")
_DIGITS = re.compile(r"\d+")


@dataclass
class UtilityResult:
    scopes: int = 0
    documents: int = 0
    absolute_checked: int = 0
    entities_checked: int = 0
    intervals_checked: int = 0
    intervals_mismatched: int = 0
    relative_checked: Counter = field(default_factory=Counter)
    relative_failed: int = 0
    months_rerendered: int = 0
    months_dropped: int = 0
    ages_checked: int = 0
    failures: list[dict] = field(default_factory=list)
    timelines: list[dict] = field(default_factory=list)      # per patient, reconstructed vs gold (shifted)

    @property
    def ok(self) -> bool:
        return not self.failures

    def summary(self) -> dict:
        """Counts only: safe to put in a receipt."""
        return {
            "pass": self.ok,
            "scopes": self.scopes, "documents": self.documents,
            "absolute_dates_checked": self.absolute_checked,
            "date_entities_checked": self.entities_checked,
            "intervals_checked": self.intervals_checked, "intervals_mismatched": self.intervals_mismatched,
            "relative_checked": dict(self.relative_checked), "relative_failed": self.relative_failed,
            "months_rerendered": self.months_rerendered, "months_dropped": self.months_dropped,
            "ages_checked": self.ages_checked,
            "failures": len(self.failures),
        }


def _fail(res: UtilityResult, sample_id: str, kind: str, **detail) -> None:
    res.failures.append({"sample_id": sample_id, "kind": kind, **detail})


def _pairs(rec: dict, out: dict):
    gold = sorted(rec["mentions"], key=lambda m: (m["start"], m["end"]))
    return gold, out["mentions"]


def check_scope(recs: list[dict], outs: dict[str, dict], policy: Policy, res: UtilityResult) -> None:
    res.scopes += 1
    # ---- structure and shift
    shifts: Counter = Counter()
    for rec in recs:
        out = outs.get(rec["sample_id"])
        if out is None:
            _fail(res, rec["sample_id"], "missing_output")
            continue
        res.documents += 1
        gold, new = _pairs(rec, out)
        if len(gold) != len(new) or any((g["type"], g["entity_id"]) != (n["type"], n["entity_id"]) for g, n in zip(gold, new)):
            _fail(res, rec["sample_id"], "mention_structure", gold=len(gold), out=len(new))
            continue
        for n in new:
            if out["text"][n["start"]:n["end"]] != n["text"]:
                _fail(res, rec["sample_id"], "offset_integrity", at=n["start"])
        gaps_in = [rec["text"][a["end"]:b["start"]] for a, b in zip(gold, gold[1:])] + [rec["text"][gold[-1]["end"]:] if gold else rec["text"]]
        gaps_out = [out["text"][a["end"]:b["start"]] for a, b in zip(new, new[1:])] + [out["text"][new[-1]["end"]:] if new else out["text"]]
        head_in = rec["text"][:gold[0]["start"]] if gold else ""
        head_out = out["text"][:new[0]["start"]] if new else ""
        if gaps_in != gaps_out or head_in != head_out:
            _fail(res, rec["sample_id"], "text_outside_mentions_changed")
        for g, n in zip(gold, new):
            if g["type"] == "DATE":
                pa = parse_absolute_date(g["text"])
                if pa is None:
                    continue
                pb = parse_absolute_date(n["text"])
                if pb is None or pb[1] != pa[1]:
                    _fail(res, rec["sample_id"], "date_format", original=g["text"], output=n["text"])
                    continue
                shifts[(pb[0] - pa[0]).days] += 1
                res.absolute_checked += 1
    if not shifts:
        return
    if len(shifts) != 1:
        _fail(res, recs[0]["sample_id"], "inconsistent_shift_in_scope", shifts=dict(shifts))
        return
    D = next(iter(shifts))
    lo, hi = 7 * policy.date_shift.min_weeks, 7 * policy.date_shift.max_weeks
    if D % 7 != 0 or not (lo <= abs(D) <= hi) or (policy.date_shift.direction == "backward" and D >= 0) \
            or (policy.date_shift.direction == "forward" and D <= 0):
        _fail(res, recs[0]["sample_id"], "shift_outside_policy", shift_days=D)

    # ---- entity dates reconstructed from the output text, per document
    for rec in recs:
        out = outs.get(rec["sample_id"])
        if out is None:
            continue
        gold, new = _pairs(rec, out)
        if len(gold) != len(new):
            continue
        ents = {e["entity_id"]: e for e in rec["entities"]}
        expected = {e["entity_id"]: dt.date.fromisoformat(e["canonical"]) + dt.timedelta(days=D)
                    for e in rec["entities"] if e["type"] == "DATE"}
        reconstructed: dict[str, set[dt.date]] = defaultdict(set)
        relative_out: list[tuple[dict, dict]] = []
        for g, n in zip(gold, new):
            if g["type"] == "DATE":
                pb = parse_absolute_date(n["text"])
                if pb:
                    reconstructed[g["entity_id"]].add(pb[0])
                else:
                    relative_out.append((g, n))
            elif g["type"] == "AGE":
                res.ages_checked += 1
                age = int(_DIGITS.search(g["text"]).group())
                if policy.age_threshold is not None and age >= policy.age_threshold:
                    want = g["text"].replace(str(age), policy.age_cap_label, 1)
                else:
                    want = g["text"]
                if n["text"] != want:
                    _fail(res, rec["sample_id"], "age", original=g["text"], output=n["text"], expected=want)
        values: dict[str, dt.date] = {}
        for eid, vals in reconstructed.items():
            res.entities_checked += 1
            if len(vals) != 1:
                _fail(res, rec["sample_id"], "entity_dates_disagree", entity=eid, values=sorted(v.isoformat() for v in vals))
                continue
            v = vals.pop()
            values[eid] = v
            if v != expected[eid]:
                _fail(res, rec["sample_id"], "entity_date_not_shifted", entity=eid, got=v.isoformat(), want=expected[eid].isoformat())
        # Entities without an absolute mention: reconstruct from their relative mentions.
        for g, n in relative_out:
            eid = g["entity_id"]
            if eid in values:
                continue
            anchor_id = (g.get("attrs") or {}).get("anchor")
            anchor_val = values.get(anchor_id) or expected.get(anchor_id)
            if anchor_val is None:
                continue
            kind, info = classify_relative(n["text"]) or (None, {})
            if kind == "study_day":
                k = int(_DAY_RE.match(n["text"]).group(1))
                values[eid] = anchor_val + dt.timedelta(days=k - 1 if k > 0 else k)
            elif kind == "prose":
                values[eid] = anchor_val + dt.timedelta(days=info["delta_days"])
        # ---- date graph intervals against reconstructed values
        for edge in rec["date_graph"]:
            if edge["anchor"] is None:
                continue
            t, a = edge["entity_id"], edge["anchor"]
            tv, av = values.get(t, expected.get(t)), values.get(a, expected.get(a))
            res.intervals_checked += 1
            if (tv - av).days != edge["interval_days"]:
                res.intervals_mismatched += 1
                _fail(res, rec["sample_id"], "interval", target=t, anchor=a, want=edge["interval_days"], got=(tv - av).days)
        # ---- relative mentions still true against the reconstructed anchor
        for g, n in relative_out:
            eid = g["entity_id"]
            anchor_id = (g.get("attrs") or {}).get("anchor")
            target = values.get(eid, expected.get(eid))
            anchor = values.get(anchor_id, expected.get(anchor_id))
            kind_g, info_g = classify_relative(g["text"]) or (None, {})
            kind_n, info_n = classify_relative(n["text"]) or (None, {})
            if kind_n is None or kind_n != kind_g:
                res.relative_failed += 1
                _fail(res, rec["sample_id"], "relative_form_changed", original=g["text"], output=n["text"])
                continue
            res.relative_checked[kind_n] += 1
            delta = (target - anchor).days
            if kind_n == "study_day":
                k = int(_DAY_RE.match(n["text"]).group(1))
                ok = k == (delta + 1 if delta >= 0 else delta) and n["text"] == g["text"]
            elif kind_n == "weekday":
                ok = _FOLLOWING_RE.match(n["text"]).group(1) == WEEKDAYS[target.weekday()] and 1 <= delta <= 7 and n["text"] == g["text"]
            else:
                ok = info_n["delta_days"] == delta
                month_want = MONTHS[anchor.month - 1]
                if info_g["month"] is None:
                    ok = ok and n["text"] == g["text"]
                elif info_n["month"] is None:
                    res.months_dropped += 1                      # policy: unresolvable anchor -> month removed
                    ok = ok and n["text"] == g["text"].replace(info_g["month"] + " ", "", 1)
                else:
                    ok = ok and info_n["month"] == month_want
                    res.months_rerendered += 1
            if not ok:
                res.relative_failed += 1
                _fail(res, rec["sample_id"], "relative_date", original=g["text"], output=n["text"],
                      target=target.isoformat(), anchor=anchor.isoformat())
        # ---- per-patient timeline (reconstructed vs gold shifted), for the report
        by_patient: dict[str, dict] = defaultdict(dict)
        for edge in rec["date_graph"]:
            eid = edge["entity_id"]
            by_patient[edge["patient"]][edge["role"]] = {
                "reconstructed": values.get(eid, expected.get(eid)).isoformat(),
                "expected": expected[eid].isoformat(),
                "interval_days": edge["interval_days"], "anchor_role": ents[edge["anchor"]]["role"] if edge["anchor"] else None,
            }
        for pid, roles in by_patient.items():
            res.timelines.append({"sample_id": rec["sample_id"], "patient": pid, "roles": roles})


def check_utility(records: list[dict], outputs: list[dict], policy: Policy) -> UtilityResult:
    res = UtilityResult()
    outs = {o["sample_id"]: o for o in outputs}
    scopes: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        scopes[r["bundle_id"] or r["sample_id"]].append(r)
    for sid in sorted(scopes):
        check_scope(scopes[sid], outs, policy, res)
    return res


def render_utility_report(res: UtilityResult, examples: list[dict]) -> str:
    s = res.summary()
    L = ["# Utility check: AE timeline reconstruction", ""]
    L.append(f"**Verdict: {'PASS' if s['pass'] else 'FAIL'}** ({s['failures']} failures, zero tolerance).")
    L.append("")
    L.append("Reconstructed from the surrogated text and its mentions only; ground truth is the corpus date graph "
             "shifted by the scope's offset (recovered from the gold/output date pairs, never from the engine).")
    L.append("")
    L.append("| check | count | failed |")
    L.append("|---|---|---|")
    L.append(f"| scopes / documents | {s['scopes']} / {s['documents']} | |")
    L.append(f"| absolute DATE mentions shifted, same format, one offset per scope | {s['absolute_dates_checked']} | |")
    L.append(f"| DATE entities: output mentions agree and equal gold + D | {s['date_entities_checked']} | |")
    L.append(f"| date-graph intervals preserved exactly | {s['intervals_checked']} | {s['intervals_mismatched']} |")
    for k, v in sorted(s["relative_checked"].items()):
        L.append(f"| relative form `{k}` still true | {v} | |")
    L.append(f"| relative forms failed | | {s['relative_failed']} |")
    L.append(f"| anchored-prose month re-rendered / dropped by policy | {s['months_rerendered']} / {s['months_dropped']} | |")
    L.append(f"| AGE mentions (pass-through under 90, 90+ above) | {s['ages_checked']} | |")
    L.append("")
    if res.failures:
        L.append("## Failures (first 50)")
        L.append("")
        for f in res.failures[:50]:
            L.append(f"- `{f['sample_id']}` {f['kind']}: " + ", ".join(f"{k}={v!r}" for k, v in f.items() if k not in ('sample_id', 'kind')))
        L.append("")
    if examples:
        L.append("## Split screen: relative-date prose before and after")
        L.append("")
        for ex in examples:
            L.append(f"`{ex['sample_id']}`, shift {ex['shift_days']} days")
            L.append("")
            L.append("> **Original:** " + ex["original"].replace("\n", " "))
            L.append(">")
            L.append("> **Surrogated:** " + ex["surrogated"].replace("\n", " "))
            L.append("")
    return "\n".join(L) + "\n"


def prose_examples(records: list[dict], outputs: list[dict], shifts: dict[str, int], limit: int = 3) -> list[dict]:
    """Sentence pairs around month-anchored prose mentions, for the report."""
    outs = {o["sample_id"]: o for o in outputs}
    ex: list[dict] = []
    for rec in records:
        out = outs.get(rec["sample_id"])
        if out is None:
            continue
        gold = sorted(rec["mentions"], key=lambda m: (m["start"], m["end"]))
        for g, n in zip(gold, out["mentions"]):
            if g["type"] == "DATE" and (g.get("attrs") or {}).get("relative") and " after the " in g["text"] or (g["type"] == "DATE" and " before the " in g["text"]):
                def sentence(text, start, end):
                    a = max(text.rfind(". ", 0, start), text.rfind("\n", 0, start))
                    b = text.find(". ", end)
                    return text[a + 1 if a >= 0 else 0:(b + 1) if b >= 0 else len(text)].strip()
                ex.append({"sample_id": rec["sample_id"], "shift_days": shifts.get(rec["sample_id"], 0),
                           "original": sentence(rec["text"], g["start"], g["end"]),
                           "surrogated": sentence(out["text"], n["start"], n["end"])})
                break
        if len(ex) >= limit:
            break
    return ex
