import pytest

from synthgen.cast import CastSpec, doc_seed, parse_master_seed, rng_from_seed
from synthgen.emit import assign_entity_ids, build_record
from synthgen.generate import Generator
from synthgen.hardcases import (
    ALL_KINDS, KIND_ID_ONLY, KIND_INITIALS_FAR, KIND_MISSPELLING, KIND_NAMELIKE, KIND_NICKNAME,
    KIND_PROSE_DATE, KIND_SAME_SURNAME, plan, verify,
)
from synthgen.validate import gate_coverage, gate_entities, gate_offsets

MASTER = parse_master_seed("deadbeef" * 4)


@pytest.fixture(scope="module")
def gen():
    return Generator(MASTER)


def _narrative(gen, doc_index, kinds, n_patients=1):
    seed = doc_seed(MASTER, doc_index)
    hard = plan(kinds, CastSpec(n_patients=n_patients), rng_from_seed(seed, "hard"))
    cast = gen.builder.build(doc_index, hard.spec)
    rendered = gen._render_narrative(cast, hard, seed)
    ids = assign_entity_ids(cast, [rendered])
    return cast, rendered, build_record(f"h-{doc_index}", cast, rendered, ids, kinds, None, "test")


@pytest.mark.parametrize("kind", [KIND_PROSE_DATE, KIND_MISSPELLING, KIND_NICKNAME, KIND_INITIALS_FAR, KIND_NAMELIKE])
def test_single_patient_injectors_produce_pattern_and_survive_gates(gen, kind):
    for i in range(12):
        cast, rendered, rec = _narrative(gen, 100 + i, [kind])
        assert verify(kind, cast, rendered), (kind, rendered.text)
        assert gate_offsets([rec]).ok
        assert gate_entities([rec]).ok, gate_entities([rec]).failures
        assert gate_coverage([rec]).ok, gate_coverage([rec]).failures


def test_prose_date_is_labeled_relative_with_anchor(gen):
    cast, rendered, rec = _narrative(gen, 300, [KIND_PROSE_DATE])
    prose_texts = {m.text for m in rendered.mentions if m.form == "prose"}
    assert prose_texts
    prose = [m for m in rec["mentions"] if m["text"] in prose_texts]
    for m in prose:
        assert m["attrs"]["relative"] is True
        assert m["attrs"]["anchor"].startswith("E")
        if not m["text"].lower().startswith("the following"):
            assert " on " + m["text"] not in rec["text"]  # no dangling preposition


def test_misspelling_keeps_one_entity_id(gen):
    cast, rendered, rec = _narrative(gen, 400, [KIND_MISSPELLING])
    p = cast.patients[0]
    bad = [m for m in rec["mentions"] if p.misspelled_last in m["text"]]
    good = [m for m in rec["mentions"] if m["text"] == p.forms["full"]]
    assert bad and good
    assert {m["entity_id"] for m in bad} == {m["entity_id"] for m in good}


def test_nickname_split_across_paragraphs(gen):
    cast, rendered, rec = _narrative(gen, 500, [KIND_NICKNAME])
    p = cast.patients[0]
    nick = [m for m in rendered.mentions if m.form in ("nickname", "nickname_first")]
    full = [m for m in rendered.mentions if m.form == "full"]
    assert nick and full
    assert nick[0].entity_key == full[0].entity_key == p.key
    assert nick[0].paragraph != full[0].paragraph


def test_initials_far_from_introduction(gen):
    cast, rendered, rec = _narrative(gen, 600, [KIND_INITIALS_FAR])
    full_para = min(m.paragraph for m in rendered.mentions if m.form == "full")
    assert any(m.form in ("initials", "initials_dot") and m.paragraph - full_para >= 2 for m in rendered.mentions)


def test_same_surname_two_patients(gen):
    for i in range(10):
        cast, rendered, rec = _narrative(gen, 700 + i, [KIND_SAME_SURNAME], n_patients=2)
        assert verify(KIND_SAME_SURNAME, cast, rendered)
        p0, p1 = cast.patients[:2]
        assert p0.last == p1.last and p0.first[0] != p1.first[0]
        # Never an ambiguous "Mr. Smith" style mention.
        for m in rendered.mentions:
            assert m.form != "title_last", m.text
        assert gate_entities([rec]).ok and gate_coverage([rec]).ok


def test_id_only_listing(gen):
    seed = doc_seed(MASTER, 800)
    hard = plan([KIND_ID_ONLY], CastSpec(n_patients=9), rng_from_seed(seed, "hard"))
    cast = gen.builder.build(800, hard.spec)
    rendered, layout = gen._render_listing(cast, hard, seed)
    assert not layout.names
    assert verify(KIND_ID_ONLY, cast, rendered, layout)
    assert not any(m.type == "PATIENT" for m in rendered.mentions)
    ids = assign_entity_ids(cast, [rendered])
    rec = build_record("h-800", cast, rendered, ids, [KIND_ID_ONLY], None, "test")
    assert gate_offsets([rec]).ok and gate_entities([rec]).ok and gate_coverage([rec]).ok


def test_namelike_distractor_is_unlabeled(gen):
    from synthgen.vocab import load_vocab
    v = load_vocab()
    cast, rendered, rec = _narrative(gen, 900, [KIND_NAMELIKE])
    drug = cast.clinical["drug"]
    assert drug in v.drugs_namelike and drug in rec["text"]
    assert not any(m["text"] == drug for m in rec["mentions"])
    assert cast.site.person_named
    site_mentions = [m for m in rec["mentions"] if m["type"] == "SITE" and m["text"] == cast.site.name]
    assert site_mentions
    # The person's surname inside the hospital name is not a separate entity.
    surname = cast.site.name.split()[0]
    assert not any(m["text"] == surname for m in rec["mentions"])


def test_all_kinds_are_covered_by_tests():
    assert set(ALL_KINDS) == {
        KIND_PROSE_DATE, KIND_MISSPELLING, KIND_NICKNAME, KIND_INITIALS_FAR,
        KIND_SAME_SURNAME, KIND_ID_ONLY, KIND_NAMELIKE,
    }
