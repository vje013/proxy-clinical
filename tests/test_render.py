import pytest

from synthgen.cast import CastBuilder, CastSpec, parse_master_seed, rng_from_seed
from synthgen.emit import assign_entity_ids, build_record
from synthgen.render import (
    Renderer, TemplateError, cross_reference_ok, load_templates, parse_token,
)
from synthgen.validate import gate_coverage, gate_distractors, gate_entities, gate_offsets

MASTER = parse_master_seed("cafebabe" * 4)


@pytest.fixture(scope="module")
def ts():
    return load_templates()


@pytest.fixture(scope="module")
def builder(ts):
    return CastBuilder(MASTER, extra_reserved=ts.static_words)


def _records(ts, builder, n=20, patients=1):
    r = Renderer(ts)
    skels = ts.skeletons_for(patients)
    out = []
    for i in range(n):
        cast = builder.build(1000 + i, CastSpec(n_patients=patients))
        sk = skels[i % len(skels)]
        rendered = r.render_narrative(cast, sk, rng_from_seed(cast.seed, "render"))
        ids = assign_entity_ids(cast, [rendered])
        out.append(build_record(f"t-{i:04d}", cast, rendered, ids, [], None, "test"))
    return out


def test_token_grammar():
    t = parse_token("patient2:initials|cap,poss")
    assert (t.role, t.index, t.form, t.mods) == ("patient", 1, "initials", ("cap", "poss"))
    t = parse_token("date:onset:any")
    assert (t.role, t.index, t.form, t.fmt) == ("date", 0, "onset", "any")
    for bad in ("patient", "date:onset", "site2:name", "c", "patient:full|shout", "bogus:x"):
        with pytest.raises(TemplateError):
            parse_token(bad)


def test_templates_load_and_lint(ts):
    assert len(ts.skeletons) >= 3
    for s in ts.skeletons:
        for para in s.paragraphs:
            for slot in para:
                assert len(slot.variants) >= 3


def test_gate1_offsets_exact_on_20_samples(ts, builder):
    recs = _records(ts, builder, 20)
    res = gate_offsets(recs)
    assert res.ok, res.failures[:5]
    # Belt and braces: re-check inline.
    for r in recs:
        for m in r["mentions"]:
            assert r["text"][m["start"]:m["end"]] == m["text"]


def test_gate3_coverage_on_20_samples(ts, builder):
    recs = _records(ts, builder, 20)
    res = gate_coverage(recs)
    assert res.ok, res.failures[:10]


def test_gate2_and_gate4_on_20_samples(ts, builder):
    recs = _records(ts, builder, 20)
    assert gate_entities(recs).ok, gate_entities(recs).failures[:10]
    assert gate_distractors(recs).ok, gate_distractors(recs).failures[:10]


def test_cross_reference_pattern_single(ts, builder):
    r = Renderer(ts)
    for i in range(30):
        cast = builder.build(2000 + i)
        for sk in ts.skeletons_for(1):
            rendered = r.render_narrative(cast, sk, rng_from_seed(cast.seed, sk.id))
            ok, why = cross_reference_ok(rendered, 1)
            assert ok, (sk.id, why)
            assert 150 <= rendered.words <= 450, (sk.id, rendered.words)


def test_entity_ids_follow_first_mention(ts, builder):
    recs = _records(ts, builder, 5)
    for r in recs:
        seen = []
        for m in sorted(r["mentions"], key=lambda m: m["start"]):
            if m["entity_id"] not in seen:
                seen.append(m["entity_id"])
        assert seen == [f"E{i + 1}" for i in range(len(seen))]


def test_render_is_deterministic(ts, builder):
    a = _records(ts, builder, 10)
    b = _records(ts, builder, 10)
    assert a == b
