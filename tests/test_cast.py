from synthgen.cast import (
    DATE_KEYS, PATIENT_ANY_FORMS, CastBuilder, CastSpec, doc_seed, parse_master_seed,
)
from synthgen.vocab import load_vocab, words_of

MASTER = parse_master_seed("cafebabe" * 4)


def _dump(cast):
    """Stable, comparable projection of a cast."""
    return {
        "locale": cast.locale,
        "patients": [(p.first, p.last, p.sex, p.age, p.subject_id, p.nickname, p.misspelled_last,
                      sorted(p.forms.items()), [(k, p.dates[k].iso) for k in DATE_KEYS])
                     for p in cast.patients],
        "inv": [(i.first, i.last, i.role, sorted(i.forms.items())) for i in cast.investigators],
        "site": (cast.site.number, cast.site.name),
        "loc": cast.location.forms["full"],
        "clinical": cast.clinical,
        "resamples": cast.resamples,
    }


def test_seed_derivation_is_hmac_and_index_sensitive():
    a = doc_seed(MASTER, 0)
    b = doc_seed(MASTER, 1)
    assert a != b and len(a) == 32
    assert doc_seed(MASTER, 0) == a


def test_determinism_across_builders():
    b1 = CastBuilder(MASTER)
    b2 = CastBuilder(MASTER)
    for i in range(30):
        spec = CastSpec(n_patients=1 + (i % 3))
        assert _dump(b1.build(i, spec)) == _dump(b2.build(i, spec))


def test_different_indices_differ():
    b = CastBuilder(MASTER)
    assert _dump(b.build(0)) != _dump(b.build(1))


def test_surface_form_set_completeness():
    b = CastBuilder(MASTER)
    for i in range(40):
        cast = b.build(i, CastSpec(n_patients=2))
        for p in cast.patients:
            for f in PATIENT_ANY_FORMS:
                assert f in p.forms
            assert p.forms["full"] == f"{p.first} {p.last}"
            assert p.forms["initials"] == f"{p.first[0]}. {p.last}"
            assert p.forms["initials_dot"] == f"{p.first[0]}.{p.last[0]}."
            assert p.forms["subject"] == f"Subject {p.subject_id}"
            assert p.forms["age"] == f"{p.age}-year-old"
            assert 18 <= p.age <= 89
            for k in DATE_KEYS:
                assert k in p.dates
        for inv in cast.investigators:
            assert inv.forms["full"].startswith("Dr. ")
            assert "@" in inv.forms["email"] and inv.forms["email"].endswith(".example")
        assert cast.site.forms["number"] == f"Site {cast.site.number}"


def test_date_graph_intervals_are_consistent():
    b = CastBuilder(MASTER)
    for i in range(40):
        cast = b.build(i)
        p = cast.patients[0]
        for k in DATE_KEYS:
            node = p.dates[k]
            if node.anchor_key is None:
                assert node.interval_days is None
                continue
            anchor = p.dates[node.anchor_key.split(".")[1]]
            assert (node.date - anchor.date).days == node.interval_days
            assert 1 <= abs(node.interval_days) <= 90
        assert p.dates["onset"].date > p.dates["first_dose"].date
        assert p.dates["screening"].date < p.dates["first_dose"].date
        assert p.dates["followup"].date >= p.dates["resolution"].date


def test_names_never_collide_with_reserved_or_each_other():
    v = load_vocab()
    b = CastBuilder(MASTER)
    for i in range(60):
        cast = b.build(i, CastSpec(n_patients=3))
        seen: set[str] = set()
        for p in cast.patients:
            toks = words_of(p.first) + words_of(p.last)
            assert not any(t in v.reserved_words for t in toks), (p.first, p.last)
            assert not any(t in seen for t in toks), (p.first, p.last, seen)
            seen.update(toks)
        for inv in cast.investigators:
            toks = words_of(inv.first) + words_of(inv.last)
            assert not any(t in seen for t in toks)
            seen.update(toks)
        loc_toks = set(words_of(cast.location.city) + words_of(cast.location.region))
        assert not (loc_toks & seen)


def test_same_surname_spec():
    b = CastBuilder(MASTER)
    for i in range(20):
        cast = b.build(i, CastSpec(n_patients=2, same_surname=True))
        p0, p1 = cast.patients
        assert p0.last == p1.last
        assert p0.first[0] != p1.first[0]
        assert "title_last" not in p0.allowed_any and "title_last" not in p1.allowed_any


def test_forced_nickname_and_misspelling():
    b = CastBuilder(MASTER)
    for i in range(20):
        cast = b.build(i, CastSpec(force_nickname=True, force_misspelling=True))
        p = cast.patients[0]
        assert p.nickname and "nickname" in p.forms and "nickname_first" in p.forms
        assert p.misspelled_last and p.misspelled_last != p.last
        assert p.forms["misspelled"] == f"{p.first} {p.misspelled_last}"


def test_person_named_site_and_namelike_drug():
    v = load_vocab()
    b = CastBuilder(MASTER)
    for i in range(10):
        cast = b.build(i, CastSpec(person_named_site=True, namelike_drug=True))
        assert cast.site.person_named
        assert cast.clinical["drug"] in v.drugs_namelike


def test_locale_share_roughly_twenty_percent():
    b = CastBuilder(MASTER)
    n = 300
    ca = sum(1 for i in range(n) if b.build(i).locale == "en_CA")
    assert 0.12 * n < ca < 0.30 * n
