"""Surrogate engine: every labelled mention is replaced by a surrogate that
keeps the mention's *form* (full name / initials / title + surname / ID
scheme / date format / age phrasing) while the identifying atoms inside it
(name words, city, site number, digit runs, the date itself) are replaced.

Keys. One HMAC-SHA256 chain, no lookup table:

    scope_key   = HMAC(master_secret, scope_id)          scope_id = bundle_id or sample_id
    entity_seed = HMAC(scope_key, entity_id)             seeds the per-entity RNG and Faker
    shift_seed  = HMAC(scope_key, "date-shift")          one offset per scope, whole weeks

The same entity id therefore maps to the same surrogate everywhere inside a
scope and to an unrelated one in any other scope; the mapping is recomputable
only with the secret and is never written down.

Input is mention-driven on purpose: ``(text, mentions)`` with mentions of
``{start, end, type, entity_id}``. Gold labels and relocated model output are
interchangeable. Nothing here reads the corpus's entity records or date
graph; those are ground truth for the tests, not inputs to production.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from faker import Faker

from synthgen.types import ENTITY_TYPE_SET
from synthgen.vocab import MONTHS, MONTHS_ABBR, WEEKDAYS, load_vocab, words_of

from .policy import Policy

# --------------------------------------------------------------------------- constants

TITLES = {"dr", "mr", "mrs", "ms", "prof", "miss"}
MALE_TITLES = {"mr"}
FEMALE_TITLES = {"mrs", "ms", "miss"}
GENERIC_RE = re.compile(
    r"^the (?:(?:first|second|third|fourth|fifth) )?(?:patient|subject|principal investigator|sub-investigator|investigator)$",
    re.IGNORECASE,
)
INSTITUTION_WORDS = {
    "memorial", "hospital", "clinical", "research", "institute", "center", "centre", "investigational",
    "medicine", "medical", "unit", "north", "south", "east", "west", "central", "site", "the", "general",
    "university", "regional", "health", "sciences", "clinic",
}
DIGITS_RE = re.compile(r"\d+")
NAME_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*\.?|\s+|[^A-Za-z\s]+")
INITIALS_DOT_RE = re.compile(r"^([A-Z])\.([A-Z])\.$")
EMAIL_RE = re.compile(r"^([a-z]+)\.([a-z]+)@site(\d+)\.example$")

# Absolute date formats, detected by shape (not by strptime order, which is
# ambiguous between dmy and us on some inputs).
_DATE_SHAPES: tuple[tuple[str, re.Pattern], ...] = (
    ("long", re.compile(r"^(\d{1,2}) ([A-Z][a-z]+) (\d{4})$")),          # 13 July 2024
    ("dmy", re.compile(r"^(\d{2})-([A-Z][a-z]{2})-(\d{4})$")),            # 13-Jul-2024
    ("us", re.compile(r"^(\d{2})/(\d{2})/(\d{4})$")),                     # 07/13/2024
    ("iso", re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")),                    # 2024-07-13
    ("mdy_long", re.compile(r"^([A-Z][a-z]+) (\d{1,2}), (\d{4})$")),      # July 13, 2024
)
_DAY_RE = re.compile(r"^Day (-?\d+)$")
_FOLLOWING_RE = re.compile(r"^the following (Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)$")
_PROSE_RE = re.compile(
    r"^(?P<n>[A-Za-z]+|\d+) (?P<unit>day|days|week|weeks) (?P<dir>after|before) the "
    r"(?P<anchor>(?:(?P<month>January|February|March|April|May|June|July|August|September|October|November|December) )?"
    r"(?:visit|assessment)|first dose visit)$"
)
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}

_FAKERS: dict[str, Faker] = {}


def _faker(locale: str) -> Faker:
    if locale not in _FAKERS:
        _FAKERS[locale] = Faker(locale)
    return _FAKERS[locale]


def _first_name_sets() -> tuple[frozenset[str], frozenset[str]]:
    from faker.providers.person.en_US import Provider
    return frozenset(Provider.first_names_male), frozenset(Provider.first_names_female)


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# --------------------------------------------------------------------------- key derivation


def scope_key(master_secret: bytes, scope_id: str) -> bytes:
    return hmac.new(master_secret, scope_id.encode("utf-8"), hashlib.sha256).digest()


def entity_seed(skey: bytes, entity_id: str) -> bytes:
    return hmac.new(skey, entity_id.encode("utf-8"), hashlib.sha256).digest()


def shift_days_for_scope(skey: bytes, policy: Policy) -> int:
    rng = random.Random(hmac.new(skey, b"date-shift", hashlib.sha256).digest())
    weeks = rng.randint(policy.date_shift.min_weeks, policy.date_shift.max_weeks)
    direction = policy.date_shift.direction
    if direction == "backward":
        sign = -1
    elif direction == "forward":
        sign = 1
    else:
        sign = rng.choice((-1, 1))
    return sign * 7 * weeks


def secret_key_id(master_secret: bytes) -> str:
    """Identifies which secret produced a run without revealing it (32 random
    bytes are not recoverable from a truncated hash)."""
    return hashlib.sha256(master_secret).hexdigest()[:16]


# --------------------------------------------------------------------------- date helpers


def parse_absolute_date(text: str) -> tuple[dt.date, str] | None:
    for shape, rx in _DATE_SHAPES:
        m = rx.match(text)
        if not m:
            continue
        try:
            if shape == "long":
                d, mon, y = m.groups()
                return dt.date(int(y), MONTHS.index(mon) + 1, int(d)), shape
            if shape == "dmy":
                d, mon, y = m.groups()
                return dt.date(int(y), MONTHS_ABBR.index(mon) + 1, int(d)), shape
            if shape == "us":
                mo, d, y = m.groups()
                return dt.date(int(y), int(mo), int(d)), shape
            if shape == "iso":
                y, mo, d = m.groups()
                return dt.date(int(y), int(mo), int(d)), shape
            if shape == "mdy_long":
                mon, d, y = m.groups()
                return dt.date(int(y), MONTHS.index(mon) + 1, int(d)), shape
        except ValueError:
            return None
    return None


def render_date(d: dt.date, shape: str) -> str:
    if shape == "long":
        return f"{d.day} {MONTHS[d.month - 1]} {d.year}"
    if shape == "dmy":
        return f"{d.day:02d}-{MONTHS_ABBR[d.month - 1]}-{d.year}"
    if shape == "us":
        return f"{d.month:02d}/{d.day:02d}/{d.year}"
    if shape == "iso":
        return d.isoformat()
    if shape == "mdy_long":
        return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"
    raise ValueError(shape)


def classify_relative(text: str) -> tuple[str, dict] | None:
    """('study_day'|'weekday'|'prose', details) or None when not a relative form."""
    if _DAY_RE.match(text):
        return "study_day", {}
    if _FOLLOWING_RE.match(text):
        return "weekday", {}
    m = _PROSE_RE.match(text[0].lower() + text[1:])
    if m:
        n_raw = m.group("n").lower()
        n = int(n_raw) if n_raw.isdigit() else _NUMBER_WORDS.get(n_raw)
        if n is None:
            return None
        days = n * 7 if m.group("unit").startswith("week") else n
        if m.group("dir") == "before":
            days = -days
        return "prose", {"delta_days": days, "month": m.group("month"), "anchor": m.group("anchor")}
    return None


# --------------------------------------------------------------------------- data


@dataclass
class DocIn:
    sample_id: str
    text: str
    mentions: list[dict]              # start, end, type, entity_id (+ ignored extras)
    bundle_id: str | None = None


@dataclass
class ScopeReport:
    scope_id: str
    documents: list[str]
    entities: int = 0
    mentions_by_type: dict = field(default_factory=dict)
    passthrough_generic: int = 0
    ages_capped: int = 0
    prose_month_rerendered: int = 0
    relative_unresolvable: list[dict] = field(default_factory=list)
    unparsed_dates: list[dict] = field(default_factory=list)
    unclassified_name_tokens: list[dict] = field(default_factory=list)
    residual_hits: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unparsed_dates and not self.residual_hits

    def as_dict(self) -> dict:
        d = dict(vars(self))
        d["ok"] = self.ok
        return d


@dataclass
class ScopeResult:
    scope_id: str
    shift_days: int                   # in memory only; never written to output or receipt
    docs: list[dict]                  # {"sample_id","text","mentions":[{start,end,text,type,entity_id}]}
    report: ScopeReport


# --------------------------------------------------------------------------- entity profiles


@dataclass
class PersonProfile:
    entity_id: str
    kind: str                                  # PATIENT | INVESTIGATOR
    first_tokens: Counter = field(default_factory=Counter)   # original first-position name words
    last_tokens: Counter = field(default_factory=Counter)    # original last-position name words
    titles: Counter = field(default_factory=Counter)
    first_map: dict[str, str] = field(default_factory=dict)  # lower original -> surrogate word
    last_map: dict[str, str] = field(default_factory=dict)
    first_sur: str = ""
    last_sur: str = ""
    nick_sur: str = ""
    digit_map: dict[str, str] = field(default_factory=dict)  # original digit run -> surrogate run


@dataclass
class PlaceProfile:
    entity_id: str
    city: str | None = None
    region: str | None = None
    country: str | None = None
    city_sur: str = ""
    region_sur: str = ""


@dataclass
class SiteProfile:
    entity_id: str
    numbers: set[str] = field(default_factory=set)
    number_map: dict[str, str] = field(default_factory=dict)
    city_phrase: str | None = None
    city_sur: str = ""
    name_tokens: set[str] = field(default_factory=set)       # proper-name tokens (person-named sites)
    name_map: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------- engine


class SurrogateEngine:
    def __init__(self, policy: Policy, master_secret: bytes):
        if len(master_secret) < 16:
            raise ValueError("master_secret must be at least 16 bytes")
        self.policy = policy
        self.secret = master_secret
        self.vocab = load_vocab()
        self.male_names, self.female_names = _first_name_sets()
        self.us_cities = {c for c, _ in self.vocab.places_us}
        self.ca_cities = {c for c, _ in self.vocab.places_ca}
        self.us_regions = {r for _, r in self.vocab.places_us}
        self.ca_regions = {r for _, r in self.vocab.places_ca}
        self.all_cities_lower = {c.lower() for c in self.us_cities | self.ca_cities}

    # ---- public

    @staticmethod
    def scope_id_of(doc: DocIn) -> str:
        return doc.bundle_id or doc.sample_id

    def process_scope(self, docs: list[DocIn]) -> ScopeResult:
        if not docs:
            raise ValueError("empty scope")
        scope_ids = {self.scope_id_of(d) for d in docs}
        if len(scope_ids) != 1:
            raise ValueError(f"documents from different scopes in one call: {sorted(scope_ids)}")
        scope_id = scope_ids.pop()
        skey = scope_key(self.secret, scope_id)
        shift = shift_days_for_scope(skey, self.policy)
        report = ScopeReport(scope_id=scope_id, documents=[d.sample_id for d in docs])

        # Gather mentions per entity across the scope.
        by_entity: dict[str, list[tuple[int, dict, str]]] = defaultdict(list)
        for di, doc in enumerate(docs):
            prev_end = -1
            for m in sorted(doc.mentions, key=lambda m: (m["start"], m["end"])):
                if m["type"] not in ENTITY_TYPE_SET:
                    raise ValueError(f"{doc.sample_id}: mention of unknown type {m['type']!r}")
                if m["start"] < prev_end:
                    raise ValueError(f"{doc.sample_id}: overlapping mentions at {m['start']}")
                if not (0 <= m["start"] < m["end"] <= len(doc.text)):
                    raise ValueError(f"{doc.sample_id}: mention span out of range")
                prev_end = m["end"]
                by_entity[m["entity_id"]].append((di, m, doc.text[m["start"]:m["end"]]))
        report.entities = len(by_entity)
        report.mentions_by_type = dict(Counter(m["type"] for ms in by_entity.values() for _, m, _ in ms))

        locale = self._infer_locale(by_entity)
        forbidden = self._forbidden_atoms(by_entity)
        used: set[str] = set()
        # Original initials pairs anywhere in scope ("J.G.", "Julian Gray" -> "jg");
        # no surrogate may reproduce one, and no two surrogates share one.
        used_initials: set[str] = self._initials_pairs(by_entity)

        # Deterministic entity order: numeric part of the id, then the id.
        order = sorted(by_entity, key=lambda e: (int(re.sub(r"\D", "", e) or 0), e))
        persons: dict[str, PersonProfile] = {}
        places: dict[str, PlaceProfile] = {}
        sites: dict[str, SiteProfile] = {}
        date_entities: set[str] = set()
        misc_digits: dict[str, dict[str, str]] = {}

        # Places first (sites follow their city), then sites, then people.
        for eid in order:
            kind = self._entity_kind(by_entity[eid])
            if kind == "LOCATION":
                places[eid] = self._place_profile(eid, by_entity[eid], skey, forbidden, used, report)
        site_numbers: dict[str, str] = {}
        for eid in order:
            kind = self._entity_kind(by_entity[eid])
            if kind == "SITE":
                prof = self._site_profile(eid, by_entity[eid], skey, forbidden, used, places, locale, report)
                sites[eid] = prof
                site_numbers.update(prof.number_map)
        for eid in order:
            kind = self._entity_kind(by_entity[eid])
            if kind in ("PATIENT", "INVESTIGATOR"):
                persons[eid] = self._person_profile(eid, kind, by_entity[eid], skey, locale, forbidden, used,
                                                    used_initials, report)
            elif kind == "DATE":
                date_entities.add(eid)
            elif kind in ("ID", "AGE", "CONTACT"):
                misc_digits[eid] = {}

        # Absolute date value per date entity (for month-anchored prose).
        entity_dates: dict[str, dt.date] = {}
        for eid in date_entities:
            vals = []
            for _, m, txt in by_entity[eid]:
                p = parse_absolute_date(txt)
                if p:
                    vals.append(p[0])
            if vals:
                entity_dates[eid] = Counter(vals).most_common(1)[0][0]

        study_day_dates = self._dates_from_study_days(by_entity, entity_dates)

        # Render every mention.
        out_docs: list[dict] = []
        for di, doc in enumerate(docs):
            pieces: list[str] = []
            new_mentions: list[dict] = []
            cursor = 0
            new_len = 0
            for m in sorted(doc.mentions, key=lambda m: (m["start"], m["end"])):
                original = doc.text[m["start"]:m["end"]]
                eid = m["entity_id"]
                t = m["type"]
                if t == "DATE":
                    rep = self._render_date_mention(original, eid, shift, entity_dates, study_day_dates, report, doc.sample_id)
                elif t == "AGE":
                    rep = self._render_age(original, report)
                elif GENERIC_RE.match(original):
                    rep = original
                    report.passthrough_generic += 1
                elif t == "ID" or t == "CONTACT":
                    rep = self._render_id_like(original, eid, t, persons, site_numbers, misc_digits, skey, forbidden, used)
                elif t in ("PATIENT", "INVESTIGATOR"):
                    prof = persons.get(eid)
                    if prof is None:   # entity only has name mentions under an unexpected mix
                        prof = persons[eid] = self._person_profile(eid, t, by_entity[eid], skey, locale, forbidden, used,
                                                                   used_initials, report)
                    rep = self._render_name(original, prof, report, doc.sample_id)
                elif t == "SITE":
                    rep = self._render_site(original, sites[eid])
                elif t == "LOCATION":
                    rep = self._render_place(original, places[eid])
                else:  # pragma: no cover
                    raise AssertionError(t)
                gap = doc.text[cursor:m["start"]]
                pieces.append(gap)
                new_len += len(gap)
                start = new_len
                pieces.append(rep)
                new_len += len(rep)
                new_mentions.append({"start": start, "end": new_len, "text": rep, "type": t, "entity_id": eid})
                cursor = m["end"]
            tail = doc.text[cursor:]
            pieces.append(tail)
            new_text = "".join(pieces)
            for nm in new_mentions:
                assert new_text[nm["start"]:nm["end"]] == nm["text"]
            out_docs.append({"sample_id": doc.sample_id, "text": new_text, "mentions": new_mentions})

        self._residual_scan(out_docs, by_entity, persons, sites, places, report)
        return ScopeResult(scope_id=scope_id, shift_days=shift, docs=out_docs, report=report)

    # ---- profiles

    @staticmethod
    def _entity_kind(ms: list[tuple[int, dict, str]]) -> str:
        types = Counter(m["type"] for _, m, _ in ms)
        for t in ("PATIENT", "INVESTIGATOR", "SITE", "LOCATION", "DATE"):
            if types.get(t):
                return t
        # Only ID / AGE / CONTACT mentions: a person referenced by identifier alone.
        return types.most_common(1)[0][0]

    def _infer_locale(self, by_entity) -> str:
        for ms in by_entity.values():
            for _, m, txt in ms:
                if m["type"] == "LOCATION":
                    parts = [p.strip() for p in txt.split(",")]
                    if parts[-1] == "Canada" or any(p in self.ca_regions for p in parts) or parts[0] in self.ca_cities:
                        return "en_CA"
                if m["type"] == "SITE" and "Centre" in txt:
                    return "en_CA"
        return "en_US"

    def _forbidden_atoms(self, by_entity) -> set[str]:
        """Lower-cased original identifying words and digit runs in the scope.
        No surrogate may equal any of them."""
        f: set[str] = set()
        for ms in by_entity.values():
            for _, m, txt in ms:
                if m["type"] in ("PATIENT", "INVESTIGATOR", "SITE", "LOCATION", "ID", "CONTACT") and not GENERIC_RE.match(txt):
                    for w in words_of(txt):
                        if w not in TITLES and w not in INSTITUTION_WORDS and len(w) > 1:
                            f.add(w)
                    for run in DIGITS_RE.findall(txt):
                        if len(run) >= 3:
                            f.add(run)
        return f

    @staticmethod
    def _rng(skey: bytes, eid: str, salt: str = "") -> random.Random:
        return random.Random(entity_seed(skey, eid) + salt.encode("utf-8"))

    def _fake(self, skey: bytes, eid: str, locale: str) -> Faker:
        fake = _faker(locale)
        fake.seed_instance(int.from_bytes(entity_seed(skey, eid)[:8], "big"))
        return fake

    def _fresh_word(self, gen, forbidden: set[str], used: set[str], avoid: set[str] = frozenset(),
                    min_len: int = 3, bad_initials: set[str] = frozenset()) -> str:
        for _ in range(400):
            w = gen()
            if not w.isalpha() or len(w) < min_len:
                continue
            lw = w.lower()
            if lw in forbidden or lw in used or lw in avoid or lw in self.vocab.reserved_words:
                continue
            if lw[0] in bad_initials:
                continue
            used.add(lw)
            return w
        raise RuntimeError("could not draw a fresh surrogate word")

    def _fresh_digits(self, rng: random.Random, original: str, forbidden: set[str], used: set[str],
                      lo: int | None = None, hi: int | None = None) -> str:
        n = len(original)
        for _ in range(500):
            if lo is not None and hi is not None:
                cand = str(rng.randint(lo, hi))
            else:
                cand = "".join(rng.choice("0123456789") for _ in range(n))
                if original[0] != "0" and cand[0] == "0":
                    cand = str(rng.randint(1, 9)) + cand[1:]
            if cand == original or cand in forbidden or cand in used:
                continue
            used.add(cand)
            return cand
        raise RuntimeError("could not draw a fresh digit run")

    @staticmethod
    def _initials_pairs(by_entity) -> set[str]:
        pairs: set[str] = set()
        for ms in by_entity.values():
            for _, m, txt in ms:
                if m["type"] not in ("PATIENT", "INVESTIGATOR") or GENERIC_RE.match(txt):
                    continue
                im = INITIALS_DOT_RE.match(txt)
                if im:
                    pairs.add((im.group(1) + im.group(2)).lower())
                    continue
                words = [w for w in txt.replace(".", " ").split() if w.lower() not in TITLES]
                if len(words) >= 2:
                    pairs.add((words[0][0] + words[-1][0]).lower())
        return pairs

    def _person_profile(self, eid, kind, ms, skey, locale, forbidden, used, used_initials, report) -> PersonProfile:
        prof = PersonProfile(entity_id=eid, kind=kind)
        single: Counter = Counter()
        for _, m, txt in ms:
            if m["type"] not in ("PATIENT", "INVESTIGATOR") or GENERIC_RE.match(txt):
                continue
            words = [w for w in NAME_TOKEN_RE.findall(txt) if w.strip() and re.match(r"[A-Za-z]", w)]
            words_clean = []
            for w in words:
                bare = w[:-1] if w.endswith(".") else w
                if bare.lower() in TITLES:
                    prof.titles[bare.lower()] += 1
                    continue
                words_clean.append(w)
            # drop initials ("J.") from position votes; keep alphabetic words
            alpha = [w for w in words_clean if not (len(w) == 2 and w.endswith("."))]
            if len(alpha) >= 2:
                prof.first_tokens[alpha[0].lower()] += 1
                prof.last_tokens[alpha[-1].lower()] += 1
            elif len(alpha) == 1:
                single[alpha[0].lower()] += 1
        # Single words: a word already seen in a position keeps it; a word seen
        # only alone is a surname when it ever follows a title, else a first name.
        nick_values = {v.lower() for v in self.vocab.nicknames.values()}
        for w, c in single.items():
            if w in prof.last_tokens:
                prof.last_tokens[w] += c
            elif w in prof.first_tokens:
                prof.first_tokens[w] += c
            else:
                titled = any(re.search(rf"\b(?:{'|'.join(TITLES)})\.?\s+{re.escape(w)}$", txt, re.IGNORECASE)
                             for _, m, txt in ms)
                near_last = any(_edit_distance(w, l) <= 2 for l in prof.last_tokens)
                near_first = any(_edit_distance(w, f) <= 2 or (len(w) >= 3 and f.startswith(w[:3])) for f in prof.first_tokens)
                if titled or near_last:
                    prof.last_tokens[w] += c                   # "Mr. Grey", a misspelled surname
                elif w in nick_values or near_first or w.title() in self.male_names | self.female_names:
                    prof.first_tokens[w] += c                  # "Maddie", "Bob", a bare first name
                elif not prof.first_tokens and prof.last_tokens:
                    prof.last_tokens[w] += c
                else:
                    prof.first_tokens[w] += c

        # Sex: title first, then the most frequent first-position word.
        sex = None
        if any(t in MALE_TITLES for t in prof.titles):
            sex = "M"
        elif any(t in FEMALE_TITLES for t in prof.titles):
            sex = "F"
        if sex is None and prof.first_tokens:
            top = prof.first_tokens.most_common(1)[0][0].title()
            if top in self.male_names and top not in self.female_names:
                sex = "M"
            elif top in self.female_names and top not in self.male_names:
                sex = "F"
        rng = self._rng(skey, eid)
        if sex is None:
            sex = rng.choice("MF")
        fake = self._fake(skey, eid, locale)
        avoid_first = set(prof.first_tokens) | set(prof.last_tokens)
        # Initials are identity too: the surrogate's initials differ from every
        # original first-word and last-word initial, so "J.G." never survives as "J.G.".
        first_initials = {w[0] for w in prof.first_tokens}
        last_initials = {w[0] for w in prof.last_tokens}
        prof.first_sur = self._fresh_word(fake.first_name_male if sex == "M" else fake.first_name_female,
                                          forbidden, used, avoid_first, bad_initials=first_initials)
        for _ in range(100):
            cand = self._fresh_word(fake.last_name, forbidden, used, avoid_first, bad_initials=last_initials)
            pair = (prof.first_sur[0] + cand[0]).lower()
            if pair not in used_initials:
                prof.last_sur = cand
                used_initials.add(pair)
                break
            used.discard(cand.lower())
        else:  # pragma: no cover
            raise RuntimeError("could not draw surrogate initials")
        prof.nick_sur = self.vocab.nicknames.get(prof.first_sur, prof.first_sur)
        # Every distinct original word maps to exactly one surrogate word.
        if prof.first_tokens:
            ordered = [w for w, _ in prof.first_tokens.most_common()]
            prof.first_map[ordered[0]] = prof.first_sur
            for w in ordered[1:]:
                prof.first_map[w] = prof.nick_sur if prof.nick_sur.lower() != prof.first_sur.lower() else prof.first_sur
        for w in prof.last_tokens:
            prof.last_map[w] = prof.last_sur
        return prof

    def _place_profile(self, eid, ms, skey, forbidden, used, report) -> PlaceProfile:
        prof = PlaceProfile(entity_id=eid)
        parts_seen: list[list[str]] = []
        for _, m, txt in ms:
            if m["type"] == "LOCATION":
                parts_seen.append([p.strip() for p in txt.split(",")])
        full = max(parts_seen, key=len) if parts_seen else []
        if full and full[-1] in ("Canada", "United States"):
            prof.country = full[-1]
            full = full[:-1]
        if len(full) >= 2:
            prof.city, prof.region = full[0], full[1]
        elif len(full) == 1:
            w = full[0]
            if w in self.us_regions | self.ca_regions and w not in self.us_cities | self.ca_cities:
                prof.region = w
            else:
                prof.city = w
        # Single-part mentions may name the other component.
        for parts in parts_seen:
            if len(parts) == 1:
                w = parts[0]
                if w == prof.city or w == prof.region or w in ("Canada", "United States"):
                    continue
                if w in self.us_regions | self.ca_regions and prof.region is None:
                    prof.region = w
                elif prof.city is None:
                    prof.city = w
        canada = prof.country == "Canada" or (prof.region in self.ca_regions) or (prof.city in self.ca_cities and prof.city not in self.us_cities)
        pool = list(self.vocab.places_ca if canada else self.vocab.places_us)
        rng = self._rng(skey, eid)
        for _ in range(500):
            city, region = rng.choice(pool)
            if city == prof.city or region == prof.region:
                continue
            toks = set(words_of(city)) | set(words_of(region))
            if toks & forbidden or toks & used:
                continue
            prof.city_sur, prof.region_sur = city, region
            used.update(toks)
            break
        else:  # pragma: no cover
            raise RuntimeError("could not draw a fresh place")
        return prof

    def _site_profile(self, eid, ms, skey, forbidden, used, places, locale, report) -> SiteProfile:
        prof = SiteProfile(entity_id=eid)
        names: list[str] = []
        for _, m, txt in ms:
            if m["type"] != "SITE":
                continue
            if DIGITS_RE.fullmatch(txt) or re.fullmatch(r"Site \d+", txt):
                prof.numbers.add(DIGITS_RE.search(txt).group())
            else:
                names.append(txt)
        rng = self._rng(skey, eid)
        for n in sorted(prof.numbers):
            prof.number_map[n] = self._fresh_digits(rng, n, forbidden, used, lo=10 ** (len(n) - 1) + 1, hi=10 ** len(n) - 1)
        # City phrase: a LOCATION entity's city that appears in the name, else a
        # known city name at the head of the name.
        city_candidates = [(p.city, p.city_sur) for p in places.values() if p.city]
        for name in names:
            body = re.sub(r"^the ", "", name)
            body = re.sub(r" site$", "", body)
            for city, sur in city_candidates:
                if re.search(rf"\b{re.escape(city)}\b", body):
                    prof.city_phrase, prof.city_sur = city, sur
                    break
            if prof.city_phrase:
                break
            # head phrase before the first institution word
            toks = body.split()
            head = []
            for t in toks:
                if t.lower() in INSTITUTION_WORDS:
                    break
                head.append(t)
            phrase = " ".join(head)
            if phrase and phrase.lower() in self.all_cities_lower:
                prof.city_phrase = phrase
                # draw a city surrogate of the same country for this site alone
                pool = list(self.vocab.places_ca if locale == "en_CA" else self.vocab.places_us)
                for _ in range(500):
                    c, r = rng.choice(pool)
                    if c.lower() != phrase.lower() and not (set(words_of(c)) & (forbidden | used)):
                        prof.city_sur = c
                        used.update(words_of(c))
                        break
                break
            for t in head:
                if t[:1].isupper() and t.isalpha():
                    prof.name_tokens.add(t)
        if prof.name_tokens:
            fake = self._fake(skey, eid, locale)
            for t in sorted(prof.name_tokens):
                prof.name_map[t.lower()] = self._fresh_word(fake.last_name, forbidden, used)
        return prof

    @staticmethod
    def _dates_from_study_days(by_entity, entity_dates: dict[str, dt.date]) -> dict[str, dt.date]:
        """Date values for entities that appear only as "Day N": Day N is
        first_dose + (N - 1) for N > 0 (Day -k is first_dose - k). The first-dose
        date is the unique value F in scope for which every "Day N" mention of an
        entity with a known date satisfies that arithmetic. No unique F, no
        inference."""
        day_mentions: list[tuple[str, int]] = []
        for eid, ms in by_entity.items():
            for _, m, txt in ms:
                if m["type"] != "DATE":
                    continue
                dm = _DAY_RE.match(txt)
                if dm:
                    day_mentions.append((eid, int(dm.group(1))))
        if not day_mentions:
            return {}

        def offset(n: int) -> int:
            return n - 1 if n > 0 else n

        known = [(eid, n) for eid, n in day_mentions if eid in entity_dates]
        candidates = {d for d in entity_dates.values()}
        if known:
            candidates = {F for F in candidates if all(entity_dates[e] == F + dt.timedelta(days=offset(n)) for e, n in known)}
        if len(candidates) != 1:
            return {}
        F = candidates.pop()
        return {eid: F + dt.timedelta(days=offset(n)) for eid, n in day_mentions if eid not in entity_dates}

    # ---- renderers

    def _render_name(self, text: str, prof: PersonProfile, report: ScopeReport, sample_id: str) -> str:
        m = INITIALS_DOT_RE.match(text)
        if m:
            return f"{prof.first_sur[0]}.{prof.last_sur[0]}."
        toks = NAME_TOKEN_RE.findall(text)
        alpha_words = [t for t in toks if re.match(r"[A-Za-z]", t)]
        has_last_word = any((t[:-1] if t.endswith(".") else t).lower() in prof.last_map for t in alpha_words)
        out: list[str] = []
        initial_index = 0
        word_index = -1
        n_words = len(alpha_words)
        for t in toks:
            if not re.match(r"[A-Za-z]", t):
                out.append(t)
                continue
            word_index += 1
            dotted = t.endswith(".")
            bare = t[:-1] if dotted else t
            low = bare.lower()
            if low in TITLES:
                out.append(t)
                continue
            if len(bare) == 1 and dotted:
                # "J. Gray" -> first initial; "J.G." handled above; a lone "J." is a first initial
                which = prof.first_sur if (has_last_word or initial_index == 0) else prof.last_sur
                out.append(which[0] + ".")
                initial_index += 1
                continue
            if low in prof.first_map:
                rep = prof.first_map[low]
            elif low in prof.last_map:
                rep = prof.last_map[low]
            else:
                # Unclassified name word: never let it through. Position decides.
                rep = prof.last_sur if (word_index == n_words - 1 and n_words > 1) else prof.first_sur
                report.unclassified_name_tokens.append({"sample_id": sample_id, "entity_id": prof.entity_id, "token": bare})
            if bare.isupper() and len(bare) > 1:
                rep = rep.upper()
            out.append(rep)
        return "".join(out)

    def _render_place(self, text: str, prof: PlaceProfile) -> str:
        out = text
        if prof.city:
            out = re.sub(rf"\b{re.escape(prof.city)}\b", prof.city_sur, out)
        if prof.region:
            out = re.sub(rf"\b{re.escape(prof.region)}\b", prof.region_sur, out)
        return out

    def _render_site(self, text: str, prof: SiteProfile) -> str:
        out = text
        for n, sur in prof.number_map.items():
            out = re.sub(rf"(?<!\d){n}(?!\d)", sur, out)
        if prof.city_phrase:
            out = re.sub(rf"\b{re.escape(prof.city_phrase)}\b", prof.city_sur, out)
        for tok, sur in prof.name_map.items():
            out = re.sub(rf"\b{re.escape(tok)}\b", sur, out, flags=re.IGNORECASE)
        return out

    def _render_age(self, text: str, report: ScopeReport) -> str:
        if self.policy.method("AGE") == "passthrough":
            return text
        m = DIGITS_RE.search(text)
        if not m:
            return text
        if int(m.group()) >= (self.policy.age_threshold or 90):
            report.ages_capped += 1
            return text[:m.start()] + self.policy.age_cap_label + text[m.end():]
        return text

    def _render_id_like(self, text, eid, t, persons, site_numbers, misc_digits, skey, forbidden, used) -> str:
        prof = persons.get(eid)
        dmap = prof.digit_map if prof else misc_digits.setdefault(eid, {})
        rng = self._rng(skey, eid, "digits")
        if t == "CONTACT":
            em = EMAIL_RE.match(text)
            if em and prof and prof.first_sur:
                site_no = em.group(3)
                site_sur = site_numbers.get(site_no) or dmap.get(site_no) or self._fresh_digits(rng, site_no, forbidden, used)
                dmap.setdefault(site_no, site_sur)
                return f"{prof.first_sur.lower()}.{prof.last_sur.lower()}@site{site_sur}.example"

        def sub(m: re.Match) -> str:
            run = m.group()
            if run in site_numbers:
                return site_numbers[run]
            if t == "CONTACT" and run in ("1", "555") or len(run) < 2:
                return run
            if run not in dmap:
                dmap[run] = self._fresh_digits(rng, run, forbidden, used)
            return dmap[run]
        return DIGITS_RE.sub(sub, text)

    def _render_date_mention(self, text, eid, shift, entity_dates, study_day_dates, report, sample_id) -> str:
        parsed = parse_absolute_date(text)
        if parsed:
            d, shape = parsed
            return render_date(d + dt.timedelta(days=shift), shape)
        rel = classify_relative(text)
        if rel is None:
            report.unparsed_dates.append({"sample_id": sample_id, "entity_id": eid, "text": text})
            return text
        kind, info = rel
        if kind in ("study_day", "weekday"):
            return text                       # interval-relative; whole-week shift keeps weekdays true
        # month-anchored prose: the month names the anchor's month, which moves with the shift
        if info["month"] is None:
            return text                       # "the first dose visit": no month to move
        target = entity_dates.get(eid)
        if target is None:
            target = study_day_dates.get(eid)
        if target is not None:
            anchor = target - dt.timedelta(days=info["delta_days"])
            new_month = MONTHS[(anchor + dt.timedelta(days=shift)).month - 1]
        else:
            # The target has no absolute mention in scope. The phrase names the
            # anchor's month; every date value in scope that falls in that month
            # is a candidate anchor. The phrase only needs the anchor's *shifted
            # month*, so the candidates need only agree on that.
            month_no = MONTHS.index(info["month"]) + 1
            new_months = {MONTHS[(d + dt.timedelta(days=shift)).month - 1]
                          for d in entity_dates.values() if d.month == month_no}
            if len(new_months) != 1:
                action = self.policy.relative_action("unresolvable")
                entry = {"sample_id": sample_id, "entity_id": eid, "text": text, "candidate_months": sorted(new_months),
                         "action": action}
                report.relative_unresolvable.append(entry)
                if action == "drop_month_and_report":
                    # "nine days after the March assessment" -> "nine days after the assessment":
                    # the interval stays true and no stale month is asserted.
                    dropped = re.sub(rf"\b{info['month']} ", "", text, count=1)
                    entry["rendered"] = dropped
                    return dropped
                return text
            new_month = new_months.pop()
        report.prose_month_rerendered += 1
        return re.sub(rf"\b{info['month']}\b", new_month, text, count=1)

    # ---- residual scan

    def _residual_scan(self, docs_out: list[dict], by_entity, persons, sites, places, report: ScopeReport) -> None:
        """No original identifying surface string survives anywhere in the
        scope's output. Scanned, word-bounded and case-insensitive: every
        non-generic name mention and each of its name words; every site and
        location mention as a phrase, plus the city, region, site number and
        proper-name tokens behind them; every ID/CONTACT mention and each digit
        run of 3+ digits. Exempt by policy: generic references, pass-through
        ages, relative date forms; absolute dates are checked by the utility
        module (shifted, not absent)."""
        needles: set[str] = set()
        for eid, ms in by_entity.items():
            for _, m, txt in ms:
                t = m["type"]
                if t in ("AGE", "DATE") or GENERIC_RE.match(txt):
                    continue
                needles.add(txt)
                if t in ("PATIENT", "INVESTIGATOR"):
                    for w in words_of(txt):
                        if w not in TITLES and len(w) > 1:
                            needles.add(w)
                elif t in ("ID", "CONTACT"):
                    for run in DIGITS_RE.findall(txt):
                        if len(run) >= 3 and not (t == "CONTACT" and run == "555"):   # fictional exchange, kept by design
                            needles.add(run)
        for prof in places.values():
            for ph in (prof.city, prof.region):
                if ph:
                    needles.add(ph)
        for prof in sites.values():
            needles.update(prof.numbers)
            needles.update(prof.name_tokens)
            if prof.city_phrase:
                needles.add(prof.city_phrase)
        for prof in persons.values():
            needles.update(prof.first_tokens)
            needles.update(prof.last_tokens)
        for od in docs_out:
            text = od["text"]
            for n in sorted(needles):
                rx = re.compile(rf"(?<![A-Za-z0-9]){re.escape(n)}(?![A-Za-z0-9])", re.IGNORECASE)
                for hit in rx.finditer(text):
                    report.residual_hits.append({"sample_id": od["sample_id"], "needle": n, "at": hit.start()})


# --------------------------------------------------------------------------- convenience


def group_scopes(docs: list[DocIn]) -> list[list[DocIn]]:
    groups: dict[str, list[DocIn]] = defaultdict(list)
    for d in docs:
        groups[SurrogateEngine.scope_id_of(d)].append(d)
    return [groups[k] for k in sorted(groups)]


def doc_from_record(rec: dict) -> DocIn:
    """A corpus record (gold) or an evaluated prediction record as engine input.
    Only text and mentions are read; entity records and date graph are not."""
    return DocIn(
        sample_id=rec["sample_id"], text=rec["text"],
        mentions=[{"start": m["start"], "end": m["end"], "type": m["type"], "entity_id": m["entity_id"]}
                  for m in rec["mentions"]],
        bundle_id=rec.get("bundle_id"),
    )
