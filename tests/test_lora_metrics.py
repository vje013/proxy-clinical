import json

import pytest

from lora.evaluate import evaluate
from lora.metrics import (
    MalformedPrediction, entity_consistency, parse_prediction, parse_prediction_v2, relocate_anchored, span_counts,
)
from synthgen.anchors import nth_occurrence, occurrence_index
from synthgen.emit import read_jsonl, training_pair

CORPUS = "data/pilot/corpus.jsonl"


def gold_to_output(rec: dict) -> str:
    """What a perfect model would emit for this record (training-view format,
    ids renumbered per sample by first appearance)."""
    remap: dict[str, str] = {}
    ms = []
    for m in sorted(rec["mentions"], key=lambda m: (m["start"], m["end"])):
        remap.setdefault(m["entity_id"], f"E{len(remap) + 1}")
        ms.append({"span": [m["start"], m["end"]], "type": m["type"], "id": remap[m["entity_id"]]})
    return json.dumps({"mentions": ms}, separators=(",", ":"))


@pytest.fixture(scope="module")
def corpus():
    recs = read_jsonl(CORPUS)
    return {r["sample_id"]: r for r in recs}


# ---------------------------------------------------------------- parser strictness

@pytest.mark.parametrize("raw", [
    "", "null", "[]", "{}", '{"mentions": {}}', '{"mentions": [], "extra": 1}',
    '{"mentions": [{"span": [0, 5], "type": "PATIENT"}]}',                       # missing id
    '{"mentions": [{"span": [0, 5], "type": "PATIENT", "id": "E1", "x": 1}]}',   # extra key
    '{"mentions": [{"span": [5, 5], "type": "PATIENT", "id": "E1"}]}',           # empty span
    '{"mentions": [{"span": [0, 5.0], "type": "PATIENT", "id": "E1"}]}',         # float offset
    '{"mentions": [{"span": [0, true], "type": "PATIENT", "id": "E1"}]}',        # bool offset
    '{"mentions": [{"span": [0, 5], "type": "", "id": "E1"}]}',
    '{"mentions": [{"span": [0, 5], "type": "PATIENT", "id": ""}]}',
    '{"mentions": [{"span": [0, 50], "type": "PATIENT", "id": "E1"}]}',          # beyond text_len=20
    '{"mentions":[{"span":[0,5],"type":"PATIENT","id":"E1"}]} trailing',
])
def test_parser_rejects_malformed(raw):
    with pytest.raises(MalformedPrediction):
        parse_prediction(raw, text_len=20)


def test_parser_accepts_well_formed_and_preserves_values():
    raw = '{"mentions":[{"span":[3,9],"type":"DATE","id":"E2"},{"span":[0,2],"type":"ID","id":"E1"}]}'
    out = parse_prediction(raw, text_len=20)
    assert out == [{"start": 3, "end": 9, "type": "DATE", "id": "E2"}, {"start": 0, "end": 2, "type": "ID", "id": "E1"}]


# ---------------------------------------------------------------- span counts

GOLD = [
    {"start": 0, "end": 5, "type": "PATIENT", "entity_id": "E1"},
    {"start": 10, "end": 14, "type": "DATE", "entity_id": "E2"},
    {"start": 20, "end": 25, "type": "PATIENT", "entity_id": "E1"},
]


def test_span_counts_exact_match_and_type_sensitivity():
    pred = [
        {"start": 0, "end": 5, "type": "PATIENT", "id": "E1"},     # tp
        {"start": 10, "end": 14, "type": "ID", "id": "E2"},        # wrong type -> fp + fn
        {"start": 20, "end": 26, "type": "PATIENT", "id": "E1"},   # off by one -> fp + fn
    ]
    total, per_type = span_counts(GOLD, pred)
    assert (total.tp, total.fp, total.fn) == (1, 2, 2)
    assert per_type["PATIENT"].tp == 1 and per_type["PATIENT"].fn == 1 and per_type["PATIENT"].fp == 1
    assert per_type["DATE"].fn == 1 and per_type["ID"].fp == 1
    assert round(total.f1, 4) == round(2 * (1 / 3) * (1 / 3) / (2 / 3), 4)


def test_span_counts_malformed_is_all_misses():
    total, per_type = span_counts(GOLD, None)
    assert (total.tp, total.fp, total.fn) == (0, 0, 3)
    assert total.f1 == 0.0 and per_type["PATIENT"].fn == 2


# ---------------------------------------------------------------- entity consistency

def test_consistency_perfect():
    pred = [{"start": 0, "end": 5, "type": "PATIENT", "id": "X"},
            {"start": 10, "end": 14, "type": "DATE", "id": "Y"},
            {"start": 20, "end": 25, "type": "PATIENT", "id": "X"}]
    r = entity_consistency(GOLD, pred)
    assert (r.entities, r.consistent, r.split, r.merged, r.incomplete) == (2, 2, 0, 0, 0)
    assert r.accuracy == 1.0


def test_consistency_split():
    pred = [{"start": 0, "end": 5, "type": "PATIENT", "id": "A"},
            {"start": 10, "end": 14, "type": "DATE", "id": "B"},
            {"start": 20, "end": 25, "type": "PATIENT", "id": "C"}]   # E1 split over A and C
    r = entity_consistency(GOLD, pred)
    assert r.split == 1 and r.consistent == 1 and r.detail["E1"] == "split"


def test_consistency_merged():
    pred = [{"start": 0, "end": 5, "type": "PATIENT", "id": "A"},
            {"start": 10, "end": 14, "type": "DATE", "id": "A"},      # the date shares E1's id
            {"start": 20, "end": 25, "type": "PATIENT", "id": "A"}]
    r = entity_consistency(GOLD, pred)
    assert r.merged == 2 and r.consistent == 0


def test_consistency_incomplete_and_type_agnostic():
    pred = [{"start": 0, "end": 5, "type": "ID", "id": "A"},          # wrong type but right span: still counts
            {"start": 20, "end": 25, "type": "PATIENT", "id": "A"}]   # E2's span missing
    r = entity_consistency(GOLD, pred)
    assert r.consistent == 1 and r.incomplete == 1 and r.detail["E2"] == "incomplete"


def test_consistency_malformed():
    r = entity_consistency(GOLD, None)
    assert r.incomplete == 2 and r.accuracy == 0.0


# ---------------------------------------------------------------- end to end on the pilot

def test_oracle_predictions_score_perfectly(corpus):
    ids = list(corpus)[:60]
    preds = [{"sample_id": s, "raw_output": gold_to_output(corpus[s]), "finished": True} for s in ids]
    res = evaluate(preds, corpus, contract="v1")
    a = res["slices"]["all"]
    assert a["malformed"] == 0
    assert a["span"]["f1"] == 1.0 and a["span"]["fp"] == 0 and a["span"]["fn"] == 0
    assert a["entity_consistency"]["accuracy"] == 1.0
    assert set(res["slices"]) >= {"all", "single"}


def test_perturbed_oracle_scores_as_expected(corpus):
    s = next(k for k, r in corpus.items() if r["slice"] == "multi" and not r["hard_case"])
    rec = corpus[s]
    good = json.loads(gold_to_output(rec))["mentions"]
    # Drop one PATIENT mention, merge the two patients' ids, leave everything else.
    patient_ids = sorted({m["id"] for m in good if m["type"] == "PATIENT"})
    a, b = patient_ids[0], patient_ids[1]
    perturbed = []
    dropped = False
    for m in good:
        if m["type"] == "DATE" and not dropped:
            dropped = True
            continue
        m = dict(m)
        if m["id"] == b:
            m["id"] = a
        perturbed.append(m)
    preds = [{"sample_id": s, "raw_output": json.dumps({"mentions": perturbed}), "finished": True},
             {"sample_id": s, "raw_output": "not json at all", "finished": False}]
    # Two "predictions" for the same sample: one perturbed, one malformed.
    res = evaluate(preds, corpus, contract="v1")
    all_ = res["slices"]["all"]
    assert all_["samples"] == 2 and all_["malformed"] == 1 and all_["unfinished"] == 1
    n_gold = len(rec["mentions"])
    # perturbed: one fn (dropped date); malformed: all fn.
    assert all_["span"]["fn"] == 1 + n_gold and all_["span"]["fp"] == 0
    ec = all_["entity_consistency"]
    n_entities = len({m["entity_id"] for m in rec["mentions"]})
    assert ec["entities"] == 2 * n_entities
    assert ec["merged"] == 2                  # both patients merged under one id
    assert ec["incomplete"] == n_entities + 1 # malformed sample + the entity whose date was dropped


# ---------------------------------------------------------------- v2 contract: text anchors

def test_occurrence_index_and_nth_occurrence_are_inverse_incl_overlaps():
    text = "aaa bab aaa"
    for start in (0, 1, 8, 9):
        n = occurrence_index(text, "aa", start)
        assert nth_occurrence(text, "aa", n) == start
    assert occurrence_index("the patient saw the patient", "the patient", 16) == 2
    assert nth_occurrence("x", "y", 1) is None and nth_occurrence("x", "x", 2) is None
    with pytest.raises(ValueError):
        occurrence_index("abc", "b", 0)


@pytest.mark.parametrize("raw", [
    "", "{}", '{"mentions": [{"text": "a", "type": "PATIENT", "id": "E1"}]}',              # missing n
    '{"mentions": [{"text": "a", "n": 0, "type": "PATIENT", "id": "E1"}]}',              # n < 1
    '{"mentions": [{"text": "a", "n": true, "type": "PATIENT", "id": "E1"}]}',           # bool n
    '{"mentions": [{"text": "a", "n": 1.0, "type": "PATIENT", "id": "E1"}]}',            # float n
    '{"mentions": [{"text": "", "n": 1, "type": "PATIENT", "id": "E1"}]}',               # empty text
    '{"mentions": [{"text": "a", "n": 1, "type": "PATIENT", "id": "E1", "span": [0, 1]}]}',  # extra key
])
def test_v2_parser_rejects_malformed(raw):
    with pytest.raises(MalformedPrediction):
        parse_prediction_v2(raw)


def test_v2_relocation_exact_and_unlocatable():
    text = "Robert Chen was seen. Robert Chen returned. Chen left."
    ms = parse_prediction_v2(json.dumps({"mentions": [
        {"text": "Robert Chen", "n": 2, "type": "PATIENT", "id": "E1"},
        {"text": "Chen", "n": 3, "type": "PATIENT", "id": "E1"},
        {"text": "Robert Chen", "n": 3, "type": "PATIENT", "id": "E1"},   # only two occurrences
        {"text": "Nobody", "n": 1, "type": "PATIENT", "id": "E2"},        # not in text
    ]}))
    located, unloc = relocate_anchored(ms, text)
    assert located == [{"start": 22, "end": 33, "type": "PATIENT", "id": "E1"},
                       {"start": 44, "end": 48, "type": "PATIENT", "id": "E1"}]
    assert [u["text"] for u in unloc] == ["Robert Chen", "Nobody"]


def test_v2_round_trip_is_exact_on_whole_pilot(corpus):
    """The emitted v2 target relocates back to exactly the gold offsets for
    every mention of every sample. This is the invariant the contract rests on."""
    for rec in corpus.values():
        out = json.loads(training_pair(rec, "v2")["output"])["mentions"]
        located, unloc = relocate_anchored(out, rec["text"])
        assert not unloc, rec["sample_id"]
        gold = sorted(((m["start"], m["end"], m["type"]) for m in rec["mentions"]))
        assert sorted((m["start"], m["end"], m["type"]) for m in located) == gold, rec["sample_id"]


def test_v2_oracle_scores_perfectly_and_unlocatable_is_fp(corpus):
    ids = list(corpus)[:40]
    preds = [{"sample_id": s, "raw_output": training_pair(corpus[s], "v2")["output"], "finished": True} for s in ids]
    res = evaluate(preds, corpus, contract="v2")
    a = res["slices"]["all"]
    assert a["malformed"] == 0 and a["unlocatable"] == 0
    assert a["span"]["f1"] == 1.0 and a["entity_consistency"]["accuracy"] == 1.0
    # Add one hallucinated anchor: precision drops, recall does not, unlocatable is counted.
    s0 = ids[0]
    out = json.loads(training_pair(corpus[s0], "v2")["output"])
    out["mentions"].append({"text": "Zzyzx Quux", "n": 1, "type": "PATIENT", "id": "E9"})
    res2 = evaluate([{"sample_id": s0, "raw_output": json.dumps(out), "finished": True}], corpus, contract="v2")
    a2 = res2["slices"]["all"]
    assert a2["unlocatable"] == 1 and a2["span"]["fp"] == 1 and a2["span"]["fn"] == 0
    assert a2["span"]["recall"] == 1.0 and a2["span"]["precision"] < 1.0
    assert a2["entity_consistency"]["accuracy"] == 1.0


# ---------------------------------------------------------------- closed type set

def test_closed_type_set_is_shared_by_generator_validator_and_parsers():
    from synthgen.cast import INVESTIGATOR_FORM_TYPES, LOCATION_FORM_TYPES, PATIENT_FORM_TYPES, SITE_FORM_TYPES
    from synthgen.emit import _FORMAT_ORDER
    from synthgen.types import ENTITY_TYPES, ENTITY_TYPE_SET
    from synthgen.validate import TYPE_COMPAT
    produced = set()
    for table in (PATIENT_FORM_TYPES, INVESTIGATOR_FORM_TYPES, SITE_FORM_TYPES, LOCATION_FORM_TYPES):
        produced |= set(table.values())
    produced |= {"DATE"}
    assert produced == ENTITY_TYPE_SET
    assert tuple(_FORMAT_ORDER) == ENTITY_TYPES
    assert set(TYPE_COMPAT) <= ENTITY_TYPE_SET and all(v <= ENTITY_TYPE_SET for v in TYPE_COMPAT.values())


@pytest.mark.parametrize("bad_type", ["DRUG", "patient", "PERSON", "Date", "AGE ", "CONTACTS"])
def test_unknown_type_makes_whole_output_malformed(bad_type, corpus):
    with pytest.raises(MalformedPrediction, match="unknown type"):
        parse_prediction(json.dumps({"mentions": [{"span": [0, 5], "type": bad_type, "id": "E1"}]}), text_len=20)
    with pytest.raises(MalformedPrediction, match="unknown type"):
        parse_prediction_v2(json.dumps({"mentions": [{"text": "a", "n": 1, "type": bad_type, "id": "E1"}]}))
    # End to end: a perfect output plus one DRUG mention scores as a malformed
    # sample (every gold mention missed), not as one false positive.
    s0 = list(corpus)[0]
    out = json.loads(training_pair(corpus[s0], "v2")["output"])
    out["mentions"].append({"text": "Zinprolin", "n": 1, "type": bad_type, "id": "E9"})
    res = evaluate([{"sample_id": s0, "raw_output": json.dumps(out), "finished": True}], corpus, contract="v2")
    a = res["slices"]["all"]
    assert a["malformed"] == 1
    assert a["span"]["tp"] == 0 and a["span"]["fn"] == len(corpus[s0]["mentions"])
    assert a["entity_consistency"]["accuracy"] == 0.0
    assert any("unknown type" in k for k in a["malformed_reasons"])
