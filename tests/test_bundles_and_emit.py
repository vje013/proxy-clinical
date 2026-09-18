import json

from synthgen.cast import parse_master_seed
from synthgen.emit import is_val, renumber_for_training, training_pair
from synthgen.generate import Composition, Generator, build_plan
from synthgen.validate import run_all_gates

MASTER = parse_master_seed("0badf00d" * 4)


def _small_corpus(n=60):
    plan = build_plan(n, Composition(), MASTER, prefix="t")
    return plan, Generator(MASTER).generate(plan)


def test_plan_counts_and_ids():
    comp = Composition()
    c = comp.counts(500)
    assert c == {"single": 260, "multi": 120, "listing": 80, "bundle_docs": 40, "hard": 50}
    plan = build_plan(500, comp, MASTER)
    assert len(plan) == 500
    assert len({s.sample_id for s in plan}) == 500
    assert sum(1 for s in plan if s.hard_kinds) == 50
    assert sum(1 for s in plan if s.slice == "bundle_narrative") == 20
    # bundles keep narrative before listing
    pos = {}
    for i, s in enumerate(plan):
        if s.bundle_no is not None:
            pos.setdefault(s.bundle_no, {})[s.slice] = i
    for b, p in pos.items():
        assert p["bundle_narrative"] < p["bundle_listing"]


def test_plan_scales_with_n_only():
    c = Composition().counts(3000)
    assert sum(v for k, v in c.items() if k != "hard") == 3000
    assert c["bundle_docs"] % 2 == 0
    assert c["hard"] == 300


def test_bundles_share_entity_ids_and_gates_pass():
    plan, recs = _small_corpus(80)
    by_bundle: dict[str, list[dict]] = {}
    for r in recs:
        if r["bundle_id"]:
            by_bundle.setdefault(r["bundle_id"], []).append(r)
    assert by_bundle
    for bid, docs in by_bundle.items():
        assert len(docs) == 2
        narr = next(d for d in docs if d["doc_type"] == "narrative")
        lst = next(d for d in docs if d["doc_type"] == "listing")
        n_ent = {e["entity_id"]: e for e in narr["entities"]}
        l_ent = {e["entity_id"]: e for e in lst["entities"]}
        # Every narrative patient appears in the listing under the same id with the same canonical.
        for eid, e in n_ent.items():
            if e["type"] == "PATIENT":
                assert eid in l_ent, (bid, eid)
                assert l_ent[eid]["canonical"] == e["canonical"]
                assert any(m["entity_id"] == eid for m in lst["mentions"])
        assert n_ent[next(k for k, v in n_ent.items() if v["type"] == "SITE")]["canonical"] == \
            l_ent[next(k for k, v in l_ent.items() if v["type"] == "SITE")]["canonical"]
        # Listing has extra patients beyond the narrative's.
        assert sum(1 for e in lst["entities"] if e["type"] == "PATIENT") > \
            sum(1 for e in narr["entities"] if e["type"] == "PATIENT")
    for g in run_all_gates(recs):
        assert g.ok, (g.gate, g.failures[:5])


def test_split_keeps_bundles_together():
    plan, recs = _small_corpus(80)
    side = {}
    for r in recs:
        if r["bundle_id"]:
            side.setdefault(r["bundle_id"], set()).add(is_val(r))
    for bid, sides in side.items():
        assert len(sides) == 1, bid


def test_training_view_format():
    plan, recs = _small_corpus(20)
    for r in recs:
        pair = training_pair(r)
        assert set(pair) == {"sample_id", "instruction_version", "instruction", "input", "output"}
        assert pair["instruction_version"] == "v1"
        out = json.loads(pair["output"])
        spans = out["mentions"]
        assert spans == sorted(spans, key=lambda s: (s["span"][0], s["span"][1]))
        for s in spans:
            assert list(s) == ["span", "type", "id"]
            assert r["text"][s["span"][0]:s["span"][1]]
        seen = []
        for s in spans:
            if s["id"] not in seen:
                seen.append(s["id"])
        assert seen == [f"E{i + 1}" for i in range(len(seen))]


def test_renumber_for_training_is_first_appearance():
    ms = [{"start": 10, "end": 12, "type": "PATIENT", "entity_id": "E7"},
          {"start": 0, "end": 3, "type": "DATE", "entity_id": "E2"},
          {"start": 20, "end": 25, "type": "ID", "entity_id": "E7"}]
    out = renumber_for_training(ms)
    assert [o["id"] for o in out] == ["E1", "E2", "E2"]


def test_corpus_is_byte_deterministic():
    from synthgen.emit import dumps_record
    _, a = _small_corpus(40)
    _, b = _small_corpus(40)
    assert [dumps_record(r) for r in a] == [dumps_record(r) for r in b]
