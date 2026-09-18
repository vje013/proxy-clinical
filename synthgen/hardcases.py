"""Hard-case injectors 1-7.

An injector never edits rendered text. It shapes the *cast* (same surname,
forced nickname, misspelling, person-named site, name-like drug) and the
*render policy* (which ``any`` slots are pinned to which surface form), then
verifies after rendering that the intended pattern is actually present. The
generator retries with another skeleton or variant roll when it is not.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .cast import Cast, CastSpec
from .render import ListingLayout, RenderPolicy, Rendered
from .vocab import load_vocab

KIND_PROSE_DATE = "prose_relative_date"       # 1
KIND_MISSPELLING = "misspelled_name"          # 2
KIND_NICKNAME = "nickname_split"              # 3
KIND_INITIALS_FAR = "initials_far"            # 4
KIND_SAME_SURNAME = "same_surname"            # 5
KIND_ID_ONLY = "id_only_reference"            # 6
KIND_NAMELIKE = "namelike_distractor"         # 7

ALL_KINDS = (
    KIND_PROSE_DATE, KIND_MISSPELLING, KIND_NICKNAME, KIND_INITIALS_FAR,
    KIND_SAME_SURNAME, KIND_ID_ONLY, KIND_NAMELIKE,
)
NARRATIVE_KINDS = (KIND_PROSE_DATE, KIND_MISSPELLING, KIND_NICKNAME, KIND_INITIALS_FAR, KIND_NAMELIKE)
LISTING_KINDS = (KIND_ID_ONLY, KIND_NAMELIKE)


@dataclass
class HardCasePlan:
    kinds: list[str]
    spec: CastSpec
    policy: RenderPolicy = field(default_factory=RenderPolicy)
    listing_names: bool | None = None     # False forces an id-only layout


def plan(kinds: list[str], base_spec: CastSpec, rng: random.Random) -> HardCasePlan:
    spec = CastSpec(**vars(base_spec))
    policy = RenderPolicy()
    listing_names: bool | None = None
    forced: dict[str, list[tuple[str, str]]] = {}

    for kind in kinds:
        if kind == KIND_PROSE_DATE:
            forced.setdefault("date1", []).append(("late", "prose"))
        elif kind == KIND_MISSPELLING:
            spec.force_misspelling = True
            form = "misspelled" if rng.random() < 0.7 else "misspelled_title"
            forced.setdefault("patient1", []).append(("late", form))
        elif kind == KIND_NICKNAME:
            spec.force_nickname = True
            form = "nickname" if rng.random() < 0.6 else "nickname_first"
            forced.setdefault("patient1", []).append(("late", form))
        elif kind == KIND_INITIALS_FAR:
            form = "initials" if rng.random() < 0.6 else "initials_dot"
            forced.setdefault("patient1", []).append(("last", form))
        elif kind == KIND_SAME_SURNAME:
            spec.same_surname = True
            if spec.n_patients < 2:
                spec.n_patients = 2
            # Make both patients cycle through initials so the surname alone
            # never disambiguates.
            forced.setdefault("patient1", []).append(("late", "initials"))
            forced.setdefault("patient2", []).append(("late", "initials"))
        elif kind == KIND_ID_ONLY:
            listing_names = False
        elif kind == KIND_NAMELIKE:
            spec.namelike_drug = True
            spec.person_named_site = True
            policy.require_site_name = True
        else:
            raise ValueError(f"unknown hard-case kind {kind!r}")
    policy.forced_forms = forced
    return HardCasePlan(kinds=list(kinds), spec=spec, policy=policy, listing_names=listing_names)


# --------------------------------------------------------------------------- verification


def verify(kind: str, cast: Cast, rendered: Rendered, layout: ListingLayout | None = None) -> bool:
    ms = rendered.mentions
    p0 = cast.patients[0].key if cast.patients else None

    def mentions_of(pkey: str, forms: tuple[str, ...]) -> list:
        return [m for m in ms if m.entity_key == pkey and m.form in forms]

    if kind == KIND_PROSE_DATE:
        return any(m.form == "prose" for m in ms)

    if kind == KIND_MISSPELLING:
        bad = mentions_of(p0, ("misspelled", "misspelled_title"))
        good = mentions_of(p0, ("full", "title_last", "initials", "first"))
        return bool(bad) and bool(good)

    if kind == KIND_NICKNAME:
        nick = mentions_of(p0, ("nickname", "nickname_first"))
        full = mentions_of(p0, ("full",))
        if not nick or not full:
            return False
        return any(abs(n.paragraph - f.paragraph) >= 1 for n in nick for f in full)

    if kind == KIND_INITIALS_FAR:
        ini = mentions_of(p0, ("initials", "initials_dot"))
        full = mentions_of(p0, ("full",))
        if not ini or not full:
            return False
        first_full = min(f.paragraph for f in full)
        return any(i.paragraph - first_full >= 2 for i in ini)

    if kind == KIND_SAME_SURNAME:
        if len(cast.patients) < 2 or cast.patients[0].last != cast.patients[1].last:
            return False
        keys = {cast.patients[0].key, cast.patients[1].key}
        seen = {m.entity_key for m in ms if m.type in ("PATIENT", "ID")}
        return keys <= seen

    if kind == KIND_ID_ONLY:
        if layout is not None and layout.names:
            return False
        for p in cast.patients:
            if any(m.entity_key == p.key and m.type == "PATIENT" for m in ms):
                return False
        return any(m.type == "ID" for m in ms)

    if kind == KIND_NAMELIKE:
        v = load_vocab()
        drug_ok = cast.clinical["drug"] in v.drugs_namelike and cast.clinical["drug"] in rendered.text
        site_ok = cast.site.person_named and any(m.entity_key == cast.site.key and m.form == "name" for m in ms)
        return drug_ok or site_ok

    raise ValueError(f"unknown hard-case kind {kind!r}")


def verify_all(kinds: list[str], cast: Cast, rendered: Rendered, layout: ListingLayout | None = None) -> bool:
    return all(verify(k, cast, rendered, layout) for k in kinds)
