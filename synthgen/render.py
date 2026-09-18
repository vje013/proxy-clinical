"""Slot filling with offset-exact label emission.

The renderer walks a template (narrative skeleton or listing layout), fills
every ``{...}`` token from the cast, and records a mention for each token that
resolves to an entity *while the text is being assembled*. Offsets are exact by
construction; nothing is ever re-located by searching the text afterwards.

Token grammar (inside braces)::

    {role[:form[:fmt]][|mod,mod]}

roles
    patient, patient1..patient3        -> cast.patients[i]
    investigator, investigator1..2     -> cast.investigators[i]
    site, location
    date, date1..date3                 -> the date chain of patient i   (form = date role, fmt = format)
    c, c1..c3                          -> clinical context (doc-level or per-patient)
    row                                -> listing row context (row patient, AE, dates)
mods
    cap   capitalise the first character of the rendered piece (mention text included)
    poss  append "'s" outside the mention span
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from .cast import (
    DATE_FORMATS, INVESTIGATOR_ANY_FORMS, INVESTIGATOR_FORM_TYPES, LOCATION_FORM_TYPES,
    NUMBER_WORDS, PATIENT_FORM_TYPES, SITE_FORM_TYPES, Cast, DateNode, Patient,
)
from .vocab import MONTHS, WEEKDAYS, words_of

TEMPLATE_DIR = Path(__file__).parent / "templates"
TOKEN_RE = re.compile(r"\{([^{}]+)\}")

# Forms that gate 3 must not scan for (bare numbers would collide with labs).
NO_SCAN_FORMS = frozenset({"age_num"})

# Generic strings that may only ever be produced by a labelled slot.
GENERIC_STRINGS = (
    "the patient", "the subject", "the investigator", "the principal investigator",
    "the sub-investigator", "the first patient", "the second patient", "the third patient",
    "the first subject", "the second subject", "the third subject",
)


class TemplateError(ValueError):
    pass


# --------------------------------------------------------------------------- templates


@dataclass
class Slot:
    variants: list[str]
    forms: dict[str, str] = field(default_factory=dict)
    optional: float = 1.0
    bank: str | None = None


@dataclass
class Skeleton:
    id: str
    kind: str                      # "narrative"
    patients: int
    paragraphs: list[list[Slot]]
    tags: list[str] = field(default_factory=list)


@dataclass
class Column:
    header: str
    cell: str
    width: int


@dataclass
class ListingLayout:
    id: str
    kind: str                      # "listing"
    style: str                     # "pipe" | "fixed"
    titles: list[str]
    columns: list[Column]
    rows_per_patient: list[int]
    names: bool                    # whether any column carries a name-like patient form
    tags: list[str] = field(default_factory=list)


@dataclass
class TemplateSet:
    banks: dict[str, list[str]]
    skeletons: list[Skeleton]
    listings: list[ListingLayout]
    static_words: frozenset[str]

    def skeletons_for(self, n_patients: int) -> list[Skeleton]:
        return [s for s in self.skeletons if s.patients == n_patients]

    def listings_for(self, names: bool | None = None) -> list[ListingLayout]:
        if names is None:
            return list(self.listings)
        return [l for l in self.listings if l.names == names]


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def load_templates(template_dir: str | None = None) -> TemplateSet:
    tdir = Path(template_dir) if template_dir else TEMPLATE_DIR
    banks_doc = _load_yaml(tdir / "_banks.yaml")
    banks: dict[str, list[str]] = {k: list(v) for k, v in banks_doc["banks"].items()}

    skeletons: list[Skeleton] = []
    listings: list[ListingLayout] = []
    for path in sorted(tdir.glob("narr-skel-*.yaml")):
        doc = _load_yaml(path)
        paragraphs: list[list[Slot]] = []
        for para in doc["paragraphs"]:
            slots: list[Slot] = []
            for raw in para:
                if "bank" in raw:
                    if raw["bank"] not in banks:
                        raise TemplateError(f"{doc['id']}: unknown bank {raw['bank']!r}")
                    variants = list(banks[raw["bank"]])
                else:
                    variants = list(raw["variants"])
                slots.append(Slot(
                    variants=variants,
                    forms={str(k): str(v) for k, v in (raw.get("forms") or {}).items()},
                    optional=float(raw.get("optional", 1.0)),
                    bank=raw.get("bank"),
                ))
            paragraphs.append(slots)
        skeletons.append(Skeleton(
            id=doc["id"], kind=doc.get("kind", "narrative"), patients=int(doc["patients"]),
            paragraphs=paragraphs, tags=list(doc.get("tags", [])),
        ))
    for path in sorted(tdir.glob("listing-*.yaml")):
        doc = _load_yaml(path)
        cols = [Column(header=c["header"], cell=c["cell"], width=int(c.get("width", 12)))
                for c in doc["columns"]]
        names = any(_has_name_form(c.cell) for c in cols)
        listings.append(ListingLayout(
            id=doc["id"], kind="listing", style=doc.get("style", "pipe"),
            titles=list(doc["titles"]), columns=cols,
            rows_per_patient=list(doc.get("rows_per_patient", [1])), names=names,
            tags=list(doc.get("tags", [])),
        ))

    static: set[str] = set()
    for variants in banks.values():
        for v in variants:
            static.update(words_of(TOKEN_RE.sub(" ", v)))
    for s in skeletons:
        for para in s.paragraphs:
            for slot in para:
                for v in slot.variants:
                    static.update(words_of(TOKEN_RE.sub(" ", v)))
    for l in listings:
        for t in l.titles:
            static.update(words_of(TOKEN_RE.sub(" ", t)))
        for c in l.columns:
            static.update(words_of(c.header))
            static.update(words_of(TOKEN_RE.sub(" ", c.cell)))

    ts = TemplateSet(banks=banks, skeletons=skeletons, listings=listings, static_words=frozenset(static))
    lint_templates(ts)
    return ts


_NAME_FORMS = {"full", "first", "title_last", "initials", "initials_dot", "nickname", "named",
               "nickname_first", "misspelled", "misspelled_title", "any"}


def _has_name_form(cell: str) -> bool:
    for m in TOKEN_RE.finditer(cell):
        parts = m.group(1).split("|")[0].split(":")
        if parts[0].startswith("patient") and len(parts) > 1 and parts[1] in _NAME_FORMS:
            return True
        if parts[0] == "row" and len(parts) > 2 and parts[1] == "patient" and parts[2] in _NAME_FORMS:
            return True
    return False


def lint_templates(ts: TemplateSet) -> None:
    """Static template text must never contain a generic entity string, every
    slot needs >= 3 variants, and every token must parse."""
    def check_text(owner: str, text: str) -> None:
        static = TOKEN_RE.sub(" ", text).lower()
        for g in GENERIC_STRINGS:
            if re.search(r"(?<![a-z])" + re.escape(g) + r"(?![a-z])", static):
                raise TemplateError(f"{owner}: static text contains generic string {g!r}: {text!r}")
        for m in TOKEN_RE.finditer(text):
            parse_token(m.group(1))

    for name, variants in ts.banks.items():
        if len(variants) < 3:
            raise TemplateError(f"bank {name!r} has {len(variants)} variants; need >= 3")
        for v in variants:
            check_text(f"bank {name}", v)
    for s in ts.skeletons:
        for pi, para in enumerate(s.paragraphs):
            for si, slot in enumerate(para):
                if len(slot.variants) < 3:
                    raise TemplateError(f"{s.id} p{pi} s{si}: {len(slot.variants)} variants; need >= 3")
                for v in slot.variants:
                    check_text(f"{s.id} p{pi} s{si}", v)
    for l in ts.listings:
        for t in l.titles:
            check_text(l.id, t)
        for c in l.columns:
            check_text(l.id, c.cell)


# --------------------------------------------------------------------------- tokens


@dataclass(frozen=True)
class Token:
    role: str
    index: int            # 0-based patient/investigator index, -1 when n/a
    form: str | None
    fmt: str | None
    mods: tuple[str, ...]


_ROLE_RE = re.compile(r"^([a-z_]+?)([0-9])?$")


def parse_token(body: str) -> Token:
    head, _, mods = body.partition("|")
    parts = head.split(":")
    m = _ROLE_RE.match(parts[0])
    if not m:
        raise TemplateError(f"bad token {{{body}}}")
    role, idx = m.group(1), m.group(2)
    if role not in ("patient", "investigator", "site", "location", "date", "c", "row"):
        raise TemplateError(f"unknown role in token {{{body}}}")
    index = int(idx) - 1 if idx else (0 if role in ("patient", "investigator", "date", "c") else -1)
    if idx and role in ("site", "location", "row"):
        raise TemplateError(f"role {role} takes no index: {{{body}}}")
    form = parts[1] if len(parts) > 1 else None
    fmt = parts[2] if len(parts) > 2 else None
    if len(parts) > 3:
        raise TemplateError(f"too many parts in token {{{body}}}")
    mod_tuple = tuple(x for x in mods.split(",") if x) if mods else ()
    for mod in mod_tuple:
        if mod not in ("cap", "poss"):
            raise TemplateError(f"unknown mod {mod!r} in {{{body}}}")
    if role in ("patient", "investigator", "site", "location") and form is None:
        raise TemplateError(f"entity token needs a form: {{{body}}}")
    if role == "date" and (form is None or fmt is None):
        raise TemplateError(f"date token needs role and format: {{{body}}}")
    if role in ("c", "row") and form is None:
        raise TemplateError(f"context token needs a key: {{{body}}}")
    return Token(role=role, index=index, form=form, fmt=fmt, mods=mod_tuple)


# --------------------------------------------------------------------------- policy


@dataclass
class RenderPolicy:
    """Hard-case injectors express themselves through this."""
    # role key ("patient1", "date1") -> list of (when, form). when in {"late", "last", "first"}.
    forced_forms: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    # Restrict a patient (index) to ID-only references (listing hard case 6).
    id_only_patients: tuple[int, ...] = ()
    # Listing titles must carry the institution name (hard case 7 in listings,
    # where the person-named hospital is the only name-like distractor).
    require_site_name: bool = False


# --------------------------------------------------------------------------- output


@dataclass
class Mention:
    start: int
    end: int
    text: str
    type: str
    entity_key: str
    form: str
    paragraph: int
    attrs: dict = field(default_factory=dict)

    def to_json(self, key_to_id: dict[str, str]) -> dict:
        d: dict = {
            "start": self.start, "end": self.end, "text": self.text, "type": self.type,
            "entity_id": key_to_id[self.entity_key],
        }
        if self.attrs:
            attrs = dict(self.attrs)
            if "anchor" in attrs:
                attrs["anchor"] = key_to_id[attrs["anchor"]]
            d["attrs"] = attrs
        return d


@dataclass
class Rendered:
    template_id: str
    doc_type: str
    text: str
    mentions: list[Mention]
    paragraph_count: int
    words: int


# --------------------------------------------------------------------------- renderer


class Renderer:
    def __init__(self, templates: TemplateSet):
        self.ts = templates

    # ---- narrative

    def render_narrative(self, cast: Cast, skeleton: Skeleton, rng: random.Random,
                         policy: RenderPolicy | None = None) -> Rendered:
        policy = policy or RenderPolicy()
        if skeleton.patients != len(cast.patients):
            raise ValueError(f"{skeleton.id} needs {skeleton.patients} patients, cast has {len(cast.patients)}")
        state = _DocState(cast, rng, policy)

        # Pass 1: choose variants and inclusion so the forced-form planner can
        # see the whole document before anything is rendered.
        chosen: list[list[tuple[Slot, str]]] = []
        for para in skeleton.paragraphs:
            row: list[tuple[Slot, str]] = []
            for slot in para:
                if slot.optional < 1.0 and rng.random() >= slot.optional:
                    continue
                row.append((slot, rng.choice(slot.variants)))
            chosen.append(row)
        chosen = [row for row in chosen if row]
        state.plan_forced(chosen)

        # Pass 2: render.
        text_parts: list[str] = []
        mentions: list[Mention] = []
        pos = 0
        for pi, row in enumerate(chosen):
            if pi > 0:
                text_parts.append("\n\n")
                pos += 2
            first_in_para = True
            for slot, variant in row:
                if not first_in_para:
                    text_parts.append(" ")
                    pos += 1
                first_in_para = False
                pieces = state.fill(variant, slot.forms, paragraph=pi)
                pos = _append_pieces(pieces, text_parts, mentions, pos, pi)
        text = "".join(text_parts)
        _assert_offsets(text, mentions)
        return Rendered(
            template_id=skeleton.id, doc_type="narrative", text=text, mentions=mentions,
            paragraph_count=len(chosen), words=len(text.split()),
        )

    # ---- listing

    def render_listing(self, cast: Cast, layout: ListingLayout, rng: random.Random,
                       policy: RenderPolicy | None = None,
                       row_patients: list[int] | None = None) -> Rendered:
        policy = policy or RenderPolicy()
        state = _DocState(cast, rng, policy)
        order = list(row_patients) if row_patients is not None else list(range(len(cast.patients)))
        text_parts: list[str] = []
        mentions: list[Mention] = []
        pos = 0

        # Title line
        titles = layout.titles
        if policy.require_site_name:
            titles = [t for t in titles if "{site:name}" in t] or titles
        title = rng.choice(titles)
        pieces = state.fill(title, {}, paragraph=0)
        pos = _append_pieces(pieces, text_parts, mentions, pos, 0)
        text_parts.append("\n\n")
        pos += 2

        sep = " | " if layout.style == "pipe" else "  "
        # Header
        header_cells: list[str] = []
        for c in layout.columns:
            header_cells.append(c.header.ljust(c.width) if layout.style == "fixed" else c.header)
        header = sep.join(header_cells).rstrip()
        text_parts.append(header)
        pos += len(header)
        text_parts.append("\n")
        pos += 1
        if layout.style == "fixed":
            rule = sep.join("-" * c.width for c in layout.columns)
        else:
            rule = sep.join("-" * max(3, len(c.header)) for c in layout.columns)
        text_parts.append(rule)
        pos += len(rule)

        line_no = 1
        for p_idx in order:
            n_rows = rng.choice(layout.rows_per_patient)
            for r in range(n_rows):
                text_parts.append("\n")
                pos += 1
                state.set_row(p_idx, r)
                for ci, col in enumerate(layout.columns):
                    if ci > 0:
                        text_parts.append(sep)
                        pos += len(sep)
                    cell_pieces = state.fill(col.cell, {}, paragraph=line_no)
                    cell_len = sum(len(t) for t, _ in cell_pieces)
                    pos = _append_pieces(cell_pieces, text_parts, mentions, pos, line_no)
                    if layout.style == "fixed" and ci < len(layout.columns) - 1:
                        pad = max(col.width - cell_len, 0)
                        text_parts.append(" " * pad)
                        pos += pad
                line_no += 1
        text = "".join(text_parts)
        _assert_offsets(text, mentions)
        return Rendered(
            template_id=layout.id, doc_type="listing", text=text, mentions=mentions,
            paragraph_count=line_no, words=len(text.split()),
        )


def _append_pieces(pieces: list[tuple[str, Mention | None]], text_parts: list[str],
                   mentions: list[Mention], pos: int, paragraph: int) -> int:
    prev_text = ""
    for text, mention in pieces:
        # Collapse ".." produced by an initials form at sentence end.
        if mention is None and text.startswith(".") and prev_text.endswith("."):
            text = text[1:]
        if mention is not None:
            mention.start = pos
            mention.end = pos + len(text)
            mention.text = text
            mention.paragraph = paragraph
            mentions.append(mention)
        text_parts.append(text)
        pos += len(text)
        if text:
            prev_text = text
    return pos


def _assert_offsets(text: str, mentions: list[Mention]) -> None:
    for m in mentions:
        if text[m.start:m.end] != m.text:
            raise AssertionError(f"offset drift: {m}")


# --------------------------------------------------------------------------- per-document state


class _DocState:
    def __init__(self, cast: Cast, rng: random.Random, policy: RenderPolicy):
        self.cast = cast
        self.rng = rng
        self.policy = policy
        self.used_forms: dict[str, list[str]] = {}     # role key -> forms already emitted
        self.any_counter: dict[str, int] = {}          # role key -> number of `any` resolved so far
        self.any_total: dict[str, int] = {}            # role key -> total `any` tokens in doc
        self.any_paragraphs: dict[str, list[int]] = {} # role key -> paragraph of each `any`
        self.first_mention_para: dict[str, int] = {}
        self.first_para: dict[str, int] = {}          # role key -> first paragraph with any token for it
        self.forced_plan: dict[tuple[str, int], str] = {}  # (role key, any ordinal) -> form
        # Per-document house style for dates: most dates share one format.
        weights = [0.40, 0.35, 0.15, 0.10]
        self.primary_date_fmt = rng.choices(DATE_FORMATS, weights=weights, k=1)[0]
        self.generic_toggle: dict[str, int] = {}
        self.row_patient: int = 0
        self.row_index: int = 0

    # ---- planning

    def plan_forced(self, chosen: list[list[tuple[Slot, str]]]) -> None:
        """Locate every ``any`` token per role so 'late'/'last' forced forms can
        be pinned to a concrete occurrence."""
        first_para: dict[str, int] = self.first_para
        any_roles: dict[str, list[str]] = {}
        for pi, row in enumerate(chosen):
            for slot, variant in row:
                for m in TOKEN_RE.finditer(variant):
                    tok = parse_token(m.group(1))
                    if tok.role not in ("patient", "investigator", "date"):
                        continue
                    key = f"{tok.role}{tok.index + 1}"
                    first_para.setdefault(key, pi)
                    if tok.role == "date":
                        is_any = tok.fmt == "any" and not slot.forms.get("date")
                    else:
                        is_any = tok.form == "any" and _override_for(slot.forms, tok) in (None, "any")
                    if is_any:
                        self.any_paragraphs.setdefault(key, []).append(pi)
                        any_roles.setdefault(key, []).append(tok.form or "")
        for key, paras in self.any_paragraphs.items():
            self.any_total[key] = len(paras)
        for key, wants in self.policy.forced_forms.items():
            paras = self.any_paragraphs.get(key, [])
            if not paras:
                continue
            taken: set[int] = set()
            for when, form in wants:
                candidates = list(range(len(paras)))
                if key.startswith("date") and form == "prose":
                    # Prose-relative rendering needs an anchored node.
                    candidates = [i for i in candidates if any_roles[key][i] != "first_dose"]
                candidates = [i for i in candidates if i not in taken]
                if not candidates:
                    continue
                ordinal: int | None = None
                if when == "first":
                    ordinal = candidates[0]
                elif when == "last":
                    ordinal = candidates[-1]
                elif when == "late":
                    fp = first_para.get(key, 0)
                    later = [i for i in candidates if paras[i] >= fp + 2]
                    ordinal = later[0] if later else candidates[-1]
                if ordinal is None:
                    continue
                taken.add(ordinal)
                self.forced_plan[(key, ordinal)] = form

    def set_row(self, patient_idx: int, row_index: int) -> None:
        self.row_patient = patient_idx
        self.row_index = row_index

    # ---- filling

    def fill(self, template: str, slot_forms: dict[str, str], paragraph: int) -> list[tuple[str, Mention | None]]:
        pieces: list[tuple[str, Mention | None]] = []
        last = 0
        for m in TOKEN_RE.finditer(template):
            if m.start() > last:
                pieces.append((template[last:m.start()], None))
            tok = parse_token(m.group(1))
            resolved = self.resolve(tok, slot_forms, paragraph)
            # "on three weeks after the March visit" -> "three weeks after the
            # March visit". The preposition belongs to absolute dates only.
            if (tok.role == "date" and resolved and resolved[0][1] is not None
                    and resolved[0][1].form == "prose"
                    and not resolved[0][0].lower().startswith("the following")
                    and pieces and pieces[-1][1] is None):
                prev = pieces[-1][0]
                if prev.endswith("On "):
                    pieces[-1] = (prev[:-3], None)
                    text, mention = resolved[0]
                    resolved[0] = (text[0].upper() + text[1:], mention)
                elif prev.endswith(" on "):
                    pieces[-1] = (prev[:-3], None)
            pieces.extend(resolved)
            last = m.end()
        if last < len(template):
            pieces.append((template[last:], None))
        return pieces

    def resolve(self, tok: Token, slot_forms: dict[str, str], paragraph: int) -> list[tuple[str, Mention | None]]:
        if tok.role == "patient":
            out = self._patient(tok, slot_forms, paragraph)
        elif tok.role == "investigator":
            out = self._investigator(tok, slot_forms, paragraph)
        elif tok.role == "site":
            out = self._site(tok)
        elif tok.role == "location":
            out = self._location(tok)
        elif tok.role == "date":
            out = self._date(tok, slot_forms, paragraph)
        elif tok.role == "c":
            out = [(self._context(tok), None)]
        elif tok.role == "row":
            out = self._row(tok)
        else:
            raise TemplateError(f"unhandled role {tok.role}")
        return _apply_mods(out, tok.mods)

    # ---- entity resolvers

    def _record(self, key: str, form: str) -> None:
        self.used_forms.setdefault(key, []).append(form)

    def _pick_any(self, key: str, allowed: tuple[str, ...], paragraph: int) -> str:
        ordinal = self.any_counter.get(key, 0)
        self.any_counter[key] = ordinal + 1
        forced = self.forced_plan.get((key, ordinal))
        if forced:
            return forced
        used = self.used_forms.get(key, [])
        unused = [f for f in allowed if f not in used]
        # Mostly reach for a form not yet used (diversity), sometimes repeat
        # one (real narratives repeat "the patient" and the subject ID).
        if unused and (len(used) < 2 or self.rng.random() < 0.65):
            return self.rng.choice(unused)
        return self.rng.choice(list(allowed))

    def _patient(self, tok: Token, slot_forms: dict[str, str], paragraph: int) -> list[tuple[str, Mention | None]]:
        idx = min(tok.index, len(self.cast.patients) - 1)
        p = self.cast.patients[idx]
        key = f"patient{idx + 1}"
        form = tok.form or "any"
        # Non-entity helpers.
        if form in ("pron_subj", "pron_obj", "pron_poss"):
            return [(p.pronoun(form.split("_")[1]), None)]
        if form == "sex_noun":
            return [(p.sex_noun, None)]
        if form == "sex_word":
            return [(p.sex_word, None)]
        if form == "sex_letter":
            return [(p.sex, None)]
        if form == "an_age":
            article = "an" if str(p.age)[0] == "8" else "a"
            return [(article + " ", None), (p.forms["age"], self._m(p.key, "AGE", "age", paragraph))]
        if form == "age_num":
            return [(str(p.age), self._m(p.key, "AGE", "age_num", paragraph))]

        override = _override_for(slot_forms, tok)
        allowed = self._allowed_forms(p, idx, paragraph)
        if form == "named":
            # Like ``any`` but never a generic ("the patient") form: used inside
            # parentheses and other spots where a generic reads badly. Not
            # eligible for forced forms, so it consumes no ``any`` ordinal.
            named_allowed = tuple(f for f in allowed if not f.startswith("generic")) or allowed
            if override not in (None, "any") and override in p.forms and not override.startswith("generic") \
                    and (override in allowed or override in _EXTRA_OVERRIDE_OK):
                form = override
            else:
                used = self.used_forms.get(key, [])
                pool = [f for f in named_allowed if f not in used] or list(named_allowed)
                form = self.rng.choice(pool)
        if form == "any":
            if override in (None, "any"):
                form = self._pick_any(key, allowed, paragraph)      # consumes an ordinal
            elif override in p.forms and (override in allowed or override in _EXTRA_OVERRIDE_OK):
                form = override
            else:
                # Override not applicable to this cast (e.g. title_last in a
                # same-surname pair, or a missing nickname): free choice.
                used = self.used_forms.get(key, [])
                pool = [f for f in allowed if f not in used] or list(allowed)
                form = self.rng.choice(pool)
        if form in ("generic_patient", "generic_subject") and idx in self.policy.id_only_patients:
            form = "subject"
        if form in ("generic_patient", "generic_subject") and self._generic_ambiguous(idx, paragraph):
            # A pinned generic in a paragraph where another patient is already
            # in play would be unresolvable; use the ordinal instead.
            form = "ordinal_patient" if form == "generic_patient" else "ordinal_subject"
        if form == "title_last" and form not in p.allowed_any:
            form = "initials"
        if form not in p.forms:
            # Requested a nickname/misspelling the cast lacks: degrade to a safe form.
            form = "full" if form in ("nickname", "misspelled") else "title_last" if form == "misspelled_title" else "first"
            if form not in p.forms:
                form = "full"
        text = p.forms[form]
        mtype = PATIENT_FORM_TYPES.get(form, "PATIENT")
        self._record(key, form)
        self.first_mention_para.setdefault(key, paragraph)
        return [(text, self._m(p.key, mtype, form, paragraph))]

    def _generic_ambiguous(self, idx: int, paragraph: int) -> bool:
        """True when another patient has been introduced at or before this
        paragraph, so "the patient" would not resolve."""
        if len(self.cast.patients) < 2:
            return False
        for j in range(len(self.cast.patients)):
            if j == idx:
                continue
            fp = self.first_para.get(f"patient{j + 1}")
            if fp is not None and fp <= paragraph:
                return True
        return False

    def _allowed_forms(self, p: Patient, idx: int, paragraph: int) -> tuple[str, ...]:
        if idx in self.policy.id_only_patients:
            return ("subject", "id")
        allowed = p.allowed_any
        if self._generic_ambiguous(idx, paragraph):
            allowed = tuple(f for f in allowed if not f.startswith("generic"))
            if "ordinal_patient" in p.forms:
                allowed = allowed + ("ordinal_patient", "ordinal_subject")
        return allowed

    def _investigator(self, tok: Token, slot_forms: dict[str, str], paragraph: int) -> list[tuple[str, Mention | None]]:
        idx = min(tok.index, len(self.cast.investigators) - 1)
        inv = self.cast.investigators[idx]
        key = f"investigator{idx + 1}"
        form = tok.form or "any"
        override = _override_for(slot_forms, tok)
        if form == "any":
            if override in (None, "any"):
                form = self._pick_any(key, INVESTIGATOR_ANY_FORMS, paragraph)
            elif override in inv.forms:
                form = override
            else:
                form = self.rng.choice(INVESTIGATOR_ANY_FORMS)
        if form not in inv.forms:
            raise TemplateError(f"unknown investigator form {form!r}")
        self._record(key, form)
        return [(inv.forms[form], self._m(inv.key, INVESTIGATOR_FORM_TYPES[form], form, paragraph))]

    def _site(self, tok: Token) -> list[tuple[str, Mention | None]]:
        s = self.cast.site
        form = tok.form or "name"
        if form == "any":
            form = self.rng.choice(("name", "number", "number"))
        if form == "bare_number":
            # Listing "Site" column: just the digits. Known to gate 2 through
            # no_scan_forms, never scanned by gate 3.
            return [(s.number, self._m(s.key, "SITE", "bare_number", 0))]
        if form not in s.forms:
            raise TemplateError(f"unknown site form {form!r}")
        return [(s.forms[form], self._m(s.key, SITE_FORM_TYPES[form], form, 0))]

    def _location(self, tok: Token) -> list[tuple[str, Mention | None]]:
        loc = self.cast.location
        form = tok.form or "full"
        if form == "any":
            form = self.rng.choice(("full", "full", "city"))
        if form not in loc.forms:
            raise TemplateError(f"unknown location form {form!r}")
        return [(loc.forms[form], self._m(loc.key, LOCATION_FORM_TYPES[form], form, 0))]

    def _date(self, tok: Token, slot_forms: dict[str, str], paragraph: int) -> list[tuple[str, Mention | None]]:
        idx = min(tok.index, len(self.cast.patients) - 1)
        p = self.cast.patients[idx]
        key = f"date{idx + 1}"
        node = p.dates.get(tok.form or "")
        if node is None:
            raise TemplateError(f"unknown date role {tok.form!r}")
        fmt = tok.fmt or "any"
        override = slot_forms.get("date")
        if fmt == "any":
            if override:
                fmt = override
            else:
                ordinal = self.any_counter.get(key, 0)
                self.any_counter[key] = ordinal + 1
                forced = self.forced_plan.get((key, ordinal))
                if forced:
                    fmt = forced
                else:
                    fmt = self.primary_date_fmt if self.rng.random() < 0.7 else self.rng.choice(DATE_FORMATS)
        if fmt == "prose" and node.anchor_key is None:
            fmt = self.primary_date_fmt
        return [self._render_date(node, p, fmt, paragraph)]

    def _render_date(self, node: DateNode, p: Patient, fmt: str, paragraph: int) -> tuple[str, Mention]:
        if fmt in DATE_FORMATS:
            return node.render(fmt, p.first_dose), self._m(node.key, "DATE", fmt, paragraph)
        if fmt == "day":
            text = node.render("day", p.first_dose)
            return text, self._m(node.key, "DATE", "day", paragraph,
                                 {"relative": True, "anchor": p.dates["first_dose"].key})
        if fmt == "prose":
            assert node.anchor_key is not None
            anchor = p.dates[node.anchor_key.split(".")[1]]
            text = self._prose_relative(node, anchor)
            return text, self._m(node.key, "DATE", "prose", paragraph,
                                 {"relative": True, "anchor": anchor.key})
        raise TemplateError(f"unknown date format {fmt!r}")

    def _prose_relative(self, node: DateNode, anchor: DateNode) -> str:
        n = node.interval_days or 0
        month = MONTHS[anchor.date.month - 1]
        opts = [f"the {month} visit", f"the {month} visit"]
        opts.append("the first dose visit" if anchor.role == "first_dose" else f"the {month} assessment")
        anchor_ref = self.rng.choice(opts)
        if 1 <= n <= 7 and self.rng.random() < 0.5:
            return f"the following {WEEKDAYS[node.date.weekday()]}"
        if n > 0 and n % 7 == 0 and n // 7 <= 12:
            w = n // 7
            unit = "week" if w == 1 else "weeks"
            return f"{NUMBER_WORDS[w]} {unit} after {anchor_ref}"
        if 1 <= n <= 12:
            unit = "day" if n == 1 else "days"
            return f"{NUMBER_WORDS[n]} {unit} after {anchor_ref}"
        if n > 12:
            return f"{n} days after {anchor_ref}"
        m = -n
        if m % 7 == 0 and m // 7 <= 12:
            w = m // 7
            unit = "week" if w == 1 else "weeks"
            return f"{NUMBER_WORDS[w]} {unit} before {anchor_ref}"
        return f"{m} days before {anchor_ref}"

    def _m(self, entity_key: str, mtype: str, form: str, paragraph: int, attrs: dict | None = None) -> Mention:
        return Mention(start=-1, end=-1, text="", type=mtype, entity_key=entity_key, form=form,
                       paragraph=paragraph, attrs=attrs or {})

    # ---- distractor context

    def _context(self, tok: Token) -> str:
        c = self.cast.clinical
        key = tok.form or ""
        if key in ("drug", "dose", "route", "freq", "indication", "study"):
            return c[key]
        idx = min(tok.index, len(c["per_patient"]) - 1)
        pp = c["per_patient"][idx]
        if key in ("ae", "ae_other", "severity", "outcome", "causality", "action",
                   "seriousness", "weight", "smoking", "ecog", "exposure_days"):
            return pp[key]
        if key == "history":
            return " and ".join(pp["history"])
        if key == "history1":
            return pp["history"][0]
        if key == "conmed":
            return pp["conmeds"][0]
        if key == "conmed2":
            return pp["conmeds"][1]
        if key == "conmeds":
            return " and ".join(pp["conmeds"])
        if key in ("lab", "lab1"):
            return pp["labs"][0]
        if key == "lab2":
            return pp["labs"][1]
        if key == "lab3":
            return pp["labs"][2]
        if key == "labs2":
            return f"{pp['labs'][0]} and {pp['labs'][1]}"
        raise TemplateError(f"unknown context key {key!r}")

    # ---- listing rows

    def _row(self, tok: Token) -> list[tuple[str, Mention | None]]:
        p_idx = self.row_patient
        p = self.cast.patients[p_idx]
        pp = self.cast.clinical["per_patient"][p_idx]
        key = tok.form or ""
        second = self.row_index > 0
        if key == "ae":
            return [(pp["ae_other"] if second else pp["ae"], None)]
        if key in ("severity", "outcome", "causality", "action"):
            return [(pp[key], None)]
        if key == "start":
            node = p.dates["action"] if second else p.dates["onset"]
            return [self._render_date(node, p, tok.fmt or "dmy", self.row_index)]
        if key == "end":
            node = p.dates["followup"] if second else p.dates["resolution"]
            return [self._render_date(node, p, tok.fmt or "dmy", self.row_index)]
        if key == "first_dose":
            return [self._render_date(p.dates["first_dose"], p, tok.fmt or "dmy", self.row_index)]
        if key == "start_day":
            node = p.dates["action"] if second else p.dates["onset"]
            return [self._render_date(node, p, "day", self.row_index)]
        if key == "serious":
            return [("Y" if pp["hospitalized"] else "N", None)]
        if key == "patient":
            # Delegates to the patient resolver for the row's patient.
            sub = Token(role="patient", index=p_idx, form=tok.fmt, fmt=None, mods=())
            return self._patient(sub, {}, self.row_index)
        if key == "investigator":
            inv = self.cast.investigators[0]
            form = tok.fmt or "last"
            return [(inv.forms[form], self._m(inv.key, INVESTIGATOR_FORM_TYPES[form], form, self.row_index))]
        raise TemplateError(f"unknown row key {key!r}")


_EXTRA_OVERRIDE_OK = frozenset({
    "nickname", "nickname_first", "misspelled", "misspelled_title", "id", "screening", "mrn",
    "ordinal_patient", "ordinal_subject", "full", "first", "initials", "initials_dot",
    "generic_patient", "generic_subject", "subject",
})


def _override_for(slot_forms: dict[str, str], tok: Token) -> str | None:
    """Skeleton slot override for a token: ``patient`` and ``patient1`` both
    address the first patient; ``patient2`` only the second."""
    if not slot_forms:
        return None
    key = f"{tok.role}{tok.index + 1}"
    if key in slot_forms:
        return slot_forms[key]
    if tok.index == 0 and tok.role in slot_forms:
        return slot_forms[tok.role]
    return None


def _apply_mods(pieces: list[tuple[str, Mention | None]], mods: tuple[str, ...]) -> list[tuple[str, Mention | None]]:
    if not mods:
        return pieces
    out = list(pieces)
    if "cap" in mods and out:
        text, mention = out[0]
        if text:
            out[0] = (text[0].upper() + text[1:], mention)
    if "poss" in mods:
        out.append(("'s", None))
    return out


# --------------------------------------------------------------------------- selection helpers


def cross_reference_ok(rendered: Rendered, n_patients: int) -> tuple[bool, str]:
    """Spec 4.3: each patient >= 4 name-like mentions with >= 3 distinct forms,
    and at least one mention pair >= 2 paragraphs apart."""
    for i in range(n_patients):
        key = f"P{i}"
        ms = [m for m in rendered.mentions if m.entity_key == key and m.type in ("PATIENT", "ID")]
        if len(ms) < 4:
            return False, f"{key}: {len(ms)} mentions"
        forms = {m.text.lower() for m in ms}
        if len(forms) < 3:
            return False, f"{key}: {len(forms)} distinct forms"
        paras = sorted({m.paragraph for m in ms})
        if paras[-1] - paras[0] < 2:
            return False, f"{key}: paragraph span {paras}"
    return True, ""
