"""Surrogate engine against the 500 pilot gold labels. The corpus's entity
records and date graph are ground truth here; the engine never sees them."""
import datetime as dt
import json
import re
from collections import Counter, defaultdict

import pytest

from deid.policy import load_policy
from deid.surrogate import (
    DocIn, SurrogateEngine, classify_relative, doc_from_record, group_scopes, parse_absolute_date, render_date,
    secret_key_id, shift_days_for_scope, scope_key,
)
from synthgen.emit import read_jsonl
from synthgen.vocab import MONTHS, load_vocab

CORPUS = "data/pilot/corpus.jsonl"
SECRET = bytes.fromhex("9f" * 32)
OTHER_SECRET = bytes.fromhex("a1" * 32)


@pytest.fixture(scope="module")
def corpus():
    return {r["sample_id"]: r for r in read_jsonl(CORPUS)}


@pytest.fixture(scope="module")
def policy():
    return load_policy()


@pytest.fixture(scope="module")
def run(corpus, policy):
    eng = SurrogateEngine(policy, SECRET)
    results = [eng.process_scope(s) for s in group_scopes([doc_from_record(r) for r in corpus.values()])]
    out = {d["sample_id"]: d for r in results for d in r.docs}
    shift = {d["sample_id"]: r.shift_days for r in results for d in r.docs}
    return results, out, shift


def _pairs(rec: dict, out: dict):
    gold = sorted(rec["mentions"], key=lambda m: (m["start"], m["end"]))
    assert len(gold) == len(out["mentions"])
    return list(zip(gold, out["mentions"]))


# ----------------------------------------------------------------- structure

def test_every_scope_is_clean(run):
    results, _, _ = run
    for r in results:
        rep = r.report
        assert rep.ok, (r.scope_id, rep.unparsed_dates[:3], rep.residual_hits[:3])
        assert not rep.unclassified_name_tokens, (r.scope_id, rep.unclassified_name_tokens[:3])


def test_mention_structure_and_offsets_preserved(corpus, run):
    _, out, _ = run
    for sid, rec in corpus.items():
        o = out[sid]
        for g, n in _pairs(rec, o):
            assert (g["type"], g["entity_id"]) == (n["type"], n["entity_id"])
            assert o["text"][n["start"]:n["end"]] == n["text"]
        # text outside mentions is untouched
        gold = sorted(rec["mentions"], key=lambda m: m["start"])
        gaps_in = [rec["text"][a["end"]:b["start"]] for a, b in zip(gold, gold[1:])]
        gaps_out = [o["text"][a["end"]:b["start"]] for a, b in zip(o["mentions"], o["mentions"][1:])]
        assert gaps_in == gaps_out


def test_generic_references_and_ages_pass_through(corpus, run):
    _, out, _ = run
    generic = re.compile(r"^the (first |second |third |fourth |fifth )?(patient|subject|principal investigator|sub-investigator)$", re.I)
    n_generic = n_age = 0
    for sid, rec in corpus.items():
        for g, n in _pairs(rec, out[sid]):
            if g["type"] == "AGE":
                assert n["text"] == g["text"]            # all pilot ages < 90
                n_age += 1
            elif generic.match(g["text"]):
                assert n["text"] == g["text"]
                n_generic += 1
    assert n_generic > 1000 and n_age == 2060


# ----------------------------------------------------------------- names

def _name_parts(canonical: str):
    first, last = canonical.split(" ", 1)
    return first, last


def test_person_forms_are_rendered_from_one_surrogate_name(corpus, run):
    """For every PATIENT/INVESTIGATOR entity: the surrogate full name is one
    (first, last) pair, and every original form renders as the same form of it."""
    _, out, _ = run
    vocab = load_vocab()
    checked = Counter()
    for sid, rec in corpus.items():
        ents = {e["entity_id"]: e for e in rec["entities"]}
        per_entity = defaultdict(list)
        for g, n in _pairs(rec, out[sid]):
            if g["type"] in ("PATIENT", "INVESTIGATOR"):
                per_entity[g["entity_id"]].append((g["text"], n["text"]))
        for eid, pairs in per_entity.items():
            ent = ents[eid]
            first, last = _name_parts(ent["canonical"])
            # find the surrogate pair from a full-form rendering
            sur_first = sur_last = None
            for a, b in pairs:
                a_words = a.replace("Dr. ", "").split()
                if a == f"{first} {last}" or a == f"Dr. {first} {last}":
                    words = b.replace("Dr. ", "").split()
                    assert len(words) == 2, (a, b)
                    sur_first, sur_last = words
            if sur_first is None:
                continue                      # entity never appears in full form; other tests cover it
            assert sur_first != first and sur_last != last
            assert sur_first[0].lower() != first[0].lower() and sur_last[0].lower() != last[0].lower()
            nick = vocab.nicknames.get(sur_first, sur_first)
            for a, b in pairs:
                if a.lower().startswith("the "):
                    assert b == a
                    continue
                exp = None
                if a == first:
                    exp = sur_first
                elif a == f"Dr. {last}":
                    exp = f"Dr. {sur_last}"
                elif re.fullmatch(r"(Mr|Mrs|Ms)\. \w+", a):
                    exp = f"{a.split()[0]} {sur_last}"
                elif a == f"{first[0]}. {last}":
                    exp = f"{sur_first[0]}. {sur_last}"
                elif a == f"{first[0]}.{last[0]}.":
                    exp = f"{sur_first[0]}.{sur_last[0]}."
                elif a == f"{first} {last}" or a == f"Dr. {first} {last}":
                    exp = b
                else:
                    # nickname or misspelled forms: same word count, no original word survives
                    aw, bw = a.split(), b.split()
                    assert len(aw) == len(bw), (a, b)
                    assert not ({first.lower(), last.lower()} & set(w.lower() for w in bw)), (a, b)
                    if len(aw) == 1:                     # bare nickname ("Maddie")
                        assert bw[0] in (nick, sur_first), (a, b)
                    else:                                # "Nick Last" or "First Misspelled"
                        assert bw[-1] == sur_last, (a, b)
                        assert bw[0] in (nick, sur_first), (a, b)
                    checked["variant"] += 1
                    continue
                assert b == exp, (a, b, exp)
                checked["form"] += 1
    assert checked["form"] > 5000 and checked["variant"] >= 10


def test_sex_of_first_name_is_preserved_where_original_is_unambiguous(corpus, run):
    from faker.providers.person.en_US import Provider
    male, female = set(Provider.first_names_male), set(Provider.first_names_female)
    _, out, _ = run
    agree = total = 0
    for sid, rec in corpus.items():
        ents = {e["entity_id"]: e for e in rec["entities"]}
        for g, n in _pairs(rec, out[sid]):
            if g["type"] != "PATIENT":
                continue
            ent = ents[g["entity_id"]]
            first, last = _name_parts(ent["canonical"])
            if g["text"] != f"{first} {last}":
                continue
            sf = n["text"].split()[0]
            if first in male and first not in female:
                total += 1; agree += sf in male
            elif first in female and first not in male:
                total += 1; agree += sf in female
    assert total > 300 and agree == total


# ----------------------------------------------------------------- ids, sites, places

def test_ids_keep_shape_and_site_link(corpus, run):
    _, out, _ = run
    for sid, rec in corpus.items():
        ents = {e["entity_id"]: e for e in rec["entities"]}
        site_numbers = {re.search(r"\d+", f).group() for e in rec["entities"] if e["type"] == "SITE"
                        for f in e["forms"] if re.fullmatch(r"Site \d+", f)}
        site_sur = set()
        for g, n in _pairs(rec, out[sid]):
            if g["type"] == "SITE" and re.fullmatch(r"(Site )?\d+", g["text"]):
                site_sur.add(re.search(r"\d+", n["text"]).group())
        for g, n in _pairs(rec, out[sid]):
            if g["type"] != "ID":
                continue
            assert re.sub(r"\d", "9", g["text"]) == re.sub(r"\d", "9", n["text"]), (g["text"], n["text"])
            for ra, rb in zip(re.findall(r"\d+", g["text"]), re.findall(r"\d+", n["text"])):
                assert ra != rb
                if ra in site_numbers and site_sur:
                    assert rb in site_sur, "subject id must carry the surrogate site number"


def test_sites_and_locations_are_coordinated(corpus, run):
    _, out, _ = run
    vocab = load_vocab()
    us = set(vocab.places_us); ca = set(vocab.places_ca)     # (city, region) pairs; a city name can sit in two regions
    n_city_sites = 0
    for sid, rec in corpus.items():
        loc = next((e for e in rec["entities"] if e["type"] == "LOCATION"), None)
        pairs = _pairs(rec, out[sid])
        loc_sur = None
        for g, n in pairs:
            if g["type"] == "LOCATION" and loc and g["text"] == loc["canonical"]:
                parts_in = [p.strip() for p in g["text"].split(",")]
                parts_out = [p.strip() for p in n["text"].split(",")]
                assert len(parts_in) == len(parts_out)
                if parts_in[-1] == "Canada":
                    assert parts_out[-1] == "Canada" and (parts_out[0], parts_out[1]) in ca
                else:
                    assert (parts_out[0], parts_out[1]) in us
                assert parts_out[0] != parts_in[0] and parts_out[1] != parts_in[1]
                loc_sur = parts_out[0]
        for g, n in pairs:
            if g["type"] == "SITE" and loc and loc_sur and g["text"].startswith(loc["forms"][1] + " "):
                assert n["text"].startswith(loc_sur + " "), (g["text"], n["text"])
                n_city_sites += 1
            if g["type"] == "SITE" and re.fullmatch(r"the \w[\w ]* site", g["text"]) and loc_sur:
                assert n["text"] == f"the {loc_sur} site"
    assert n_city_sites > 100


# ----------------------------------------------------------------- dates

def test_dates_shift_by_one_whole_week_offset_per_scope_and_keep_format(corpus, run, policy):
    _, out, shift = run
    n_abs = 0
    for sid, rec in corpus.items():
        d_scope = shift[sid]
        assert d_scope % 7 == 0 and -7 * policy.date_shift.max_weeks <= d_scope <= -7 * policy.date_shift.min_weeks
        for g, n in _pairs(rec, out[sid]):
            if g["type"] != "DATE":
                continue
            pa = parse_absolute_date(g["text"])
            if pa is None:
                continue
            pb = parse_absolute_date(n["text"])
            assert pb is not None and pb[1] == pa[1], (g["text"], n["text"])
            assert (pb[0] - pa[0]).days == d_scope
            assert pb[0].weekday() == pa[0].weekday()
            n_abs += 1
    assert n_abs == 7019 - 459


def test_relative_dates_stay_true(corpus, run):
    """Day N and weekday forms pass through; month-anchored prose names the
    shifted anchor's month (anchor from the gold date graph); an unresolvable
    month is dropped, never left stale."""
    _, out, shift = run
    counts = Counter()
    for sid, rec in corpus.items():
        ents = {e["entity_id"]: e for e in rec["entities"]}
        for g, n in _pairs(rec, out[sid]):
            if g["type"] != "DATE" or parse_absolute_date(g["text"]):
                continue
            kind, info = classify_relative(g["text"])
            if kind in ("study_day", "weekday"):
                assert n["text"] == g["text"]
                counts[kind] += 1
                continue
            anchor_id = g["attrs"]["anchor"]
            anchor = dt.date.fromisoformat(ents[anchor_id]["canonical"])
            shifted_month = MONTHS[(anchor + dt.timedelta(days=shift[sid])).month - 1]
            if info["month"] is None:
                assert n["text"] == g["text"]
                counts["first_dose_anchor"] += 1
            elif " the " + shifted_month + " " in " " + n["text"] + " ":
                counts["rerendered"] += 1
            else:
                assert n["text"] == g["text"].replace(info["month"] + " ", ""), (g["text"], n["text"])
                assert info["month"] not in n["text"]
                counts["dropped"] += 1
    assert counts["study_day"] > 400 and counts["rerendered"] >= 8 and counts["dropped"] <= 3


# ----------------------------------------------------------------- keys, consistency, determinism

def test_bundle_consistency_and_cross_scope_unlinkability(corpus, run):
    results, out, _ = run
    bundles = defaultdict(list)
    for rec in corpus.values():
        if rec["bundle_id"]:
            bundles[rec["bundle_id"]].append(rec)
    assert len(bundles) == 20
    for bid, docs in bundles.items():
        rendering: dict[tuple[str, str], str] = {}
        for rec in docs:
            for g, n in _pairs(rec, out[rec["sample_id"]]):
                key = (g["entity_id"], g["text"])
                assert rendering.setdefault(key, n["text"]) == n["text"], (bid, key)
    # The same entity id + same original string in two different scopes gives different surrogates.
    seen: dict[str, set[str]] = defaultdict(set)
    for rec in corpus.values():
        for g, n in _pairs(rec, out[rec["sample_id"]]):
            if g["type"] == "PATIENT" and g["text"] == next(e["canonical"] for e in rec["entities"] if e["entity_id"] == g["entity_id"]):
                seen[g["entity_id"]].add(n["text"])
    assert len(seen["E1"]) > 200


def test_determinism_and_secret_dependence(corpus, policy):
    recs = list(corpus.values())[:40]
    scopes = group_scopes([doc_from_record(r) for r in recs])
    a = [SurrogateEngine(policy, SECRET).process_scope(s) for s in scopes]
    b = [SurrogateEngine(policy, SECRET).process_scope(s) for s in scopes]
    assert [json.dumps(r.docs, sort_keys=True) for r in a] == [json.dumps(r.docs, sort_keys=True) for r in b]
    c = [SurrogateEngine(policy, OTHER_SECRET).process_scope(s) for s in scopes]
    assert all(x.docs != y.docs for x, y in zip(a, c))
    assert secret_key_id(SECRET) != secret_key_id(OTHER_SECRET) and len(secret_key_id(SECRET)) == 16
    # Shift is a function of (secret, scope) only.
    sk = scope_key(SECRET, "pilot-000000")
    assert shift_days_for_scope(sk, policy) == a[0].shift_days


def test_single_document_scope_keys_on_sample_id(policy):
    eng = SurrogateEngine(policy, SECRET)
    doc = DocIn("x-1", "Julian Gray was seen on 13 July 2024.", [
        {"start": 0, "end": 11, "type": "PATIENT", "entity_id": "E1"},
        {"start": 24, "end": 36, "type": "DATE", "entity_id": "E2"}])
    r1 = eng.process_scope([doc])
    r2 = eng.process_scope([DocIn("x-2", doc.text, doc.mentions)])
    assert r1.docs[0]["text"] != r2.docs[0]["text"]
    assert r1.scope_id == "x-1"


# ----------------------------------------------------------------- unit: age cap, unknown types, formats

def test_age_cap_safe_harbor_90(policy):
    eng = SurrogateEngine(policy, SECRET)
    text = "Ann Lee, a 94-year-old woman aged 94, and Bo Kim, a 89-year-old man."
    ms = [
        {"start": 0, "end": 7, "type": "PATIENT", "entity_id": "E1"},
        {"start": 11, "end": 22, "type": "AGE", "entity_id": "E1"},
        {"start": 29, "end": 36, "type": "AGE", "entity_id": "E1"},
        {"start": 42, "end": 48, "type": "PATIENT", "entity_id": "E2"},
        {"start": 52, "end": 63, "type": "AGE", "entity_id": "E2"},
    ]
    r = eng.process_scope([DocIn("a", text, ms)])
    outs = [m["text"] for m in r.docs[0]["mentions"]]
    assert outs[1] == "90+-year-old" and outs[2] == "aged 90+" and outs[4] == "89-year-old"
    assert r.report.ages_capped == 2 and r.report.ok


def test_unknown_type_and_overlap_are_rejected(policy):
    eng = SurrogateEngine(policy, SECRET)
    with pytest.raises(ValueError, match="unknown type"):
        eng.process_scope([DocIn("a", "Zinprolin", [{"start": 0, "end": 9, "type": "DRUG", "entity_id": "E1"}])])
    with pytest.raises(ValueError, match="overlapping"):
        eng.process_scope([DocIn("a", "Julian Gray", [
            {"start": 0, "end": 11, "type": "PATIENT", "entity_id": "E1"},
            {"start": 7, "end": 11, "type": "PATIENT", "entity_id": "E1"}])])


@pytest.mark.parametrize("text,shape", [
    ("13 July 2024", "long"), ("6 August 2024", "long"), ("13-Jul-2024", "dmy"), ("07/13/2024", "us"),
    ("2024-07-13", "iso"), ("July 13, 2024", "mdy_long"),
])
def test_date_shapes_round_trip(text, shape):
    d, s = parse_absolute_date(text)
    assert s == shape and render_date(d, s) == text


def test_contact_forms(policy):
    eng = SurrogateEngine(policy, SECRET)
    text = "Dr. Michael Gamble (michael.gamble@site3974.example, +1 (541) 555-0191) at Site 3974."
    ms = [
        {"start": 0, "end": 18, "type": "INVESTIGATOR", "entity_id": "E1"},
        {"start": 20, "end": 51, "type": "CONTACT", "entity_id": "E1"},
        {"start": 53, "end": 70, "type": "CONTACT", "entity_id": "E1"},
        {"start": 75, "end": 84, "type": "SITE", "entity_id": "E2"},
    ]
    r = eng.process_scope([DocIn("c", text, ms)])
    name, email, phone, site = [m["text"] for m in r.docs[0]["mentions"]]
    first, last = name.replace("Dr. ", "").split()
    site_no = re.search(r"\d+", site).group()
    assert email == f"{first.lower()}.{last.lower()}@site{site_no}.example"
    assert re.fullmatch(r"\+1 \(\d{3}\) 555-\d{4}", phone) and phone != "+1 (541) 555-0191"
    assert r.report.ok
