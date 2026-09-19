"""Identity pool: seeded casts, surface-form sets, date graphs.

A *cast* is everything about one document unit that must be de-identified,
sampled deterministically from ``HMAC-SHA256(master_seed, doc_index)``. The
renderer never invents identity text; it only picks surface forms that are
precomputed here, which is what makes offset-exact labelling possible.

Entity keys used inside a cast are stable strings (``P0``, ``I1``, ``S``,
``L``, ``P0.onset``). They are mapped to ``E1, E2, ...`` at emit time, ordered
by first mention in the rendered text.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import random
from dataclasses import dataclass

from faker import Faker

from .types import ENTITY_TYPE_SET
from .vocab import MONTHS, MONTHS_ABBR, Vocab, load_vocab, words_of

# One Faker per locale, created once. ``seed_instance`` resets its RNG per
# document, so this is deterministic and avoids per-document construction cost.
_FAKERS: dict[str, Faker] = {}


def _faker(locale: str) -> Faker:
    if locale not in _FAKERS:
        _FAKERS[locale] = Faker(locale)
    return _FAKERS[locale]


# --------------------------------------------------------------------------- seeds


def parse_master_seed(value: str) -> bytes:
    """Hex string when it is valid hex of even length, otherwise UTF-8 bytes."""
    v = value.strip()
    try:
        if len(v) % 2 == 0 and len(v) > 0:
            return bytes.fromhex(v)
    except ValueError:
        pass
    return v.encode("utf-8")


def doc_seed(master_seed: bytes, doc_index: int) -> bytes:
    return hmac.new(master_seed, doc_index.to_bytes(8, "big"), hashlib.sha256).digest()


def rng_from_seed(seed: bytes, salt: str = "") -> random.Random:
    """A ``random.Random`` seeded from bytes. Python hashes bytes seeds with
    SHA-512 internally, so this is independent of PYTHONHASHSEED."""
    return random.Random(seed + salt.encode("utf-8"))


# --------------------------------------------------------------------------- types

PATIENT_FORM_TYPES: dict[str, str] = {
    "full": "PATIENT",
    "first": "PATIENT",
    "title_last": "PATIENT",
    "initials": "PATIENT",
    "initials_dot": "PATIENT",
    "generic_patient": "PATIENT",
    "generic_subject": "PATIENT",
    "ordinal_patient": "PATIENT",
    "ordinal_subject": "PATIENT",
    "nickname": "PATIENT",
    "nickname_first": "PATIENT",
    "misspelled": "PATIENT",
    "misspelled_title": "PATIENT",
    "subject": "ID",
    "id": "ID",
    "screening": "ID",
    "mrn": "ID",
    "age": "AGE",
    "aged": "AGE",
}

# Forms the ``any`` directive may choose from (name-like references only).
PATIENT_ANY_FORMS: tuple[str, ...] = (
    "full", "first", "title_last", "initials", "initials_dot",
    "generic_patient", "generic_subject", "subject",
)

INVESTIGATOR_FORM_TYPES: dict[str, str] = {
    "full": "INVESTIGATOR",
    "last": "INVESTIGATOR",
    "plain": "INVESTIGATOR",
    "generic": "INVESTIGATOR",
    "phone": "CONTACT",
    "email": "CONTACT",
}
INVESTIGATOR_ANY_FORMS: tuple[str, ...] = ("full", "last", "generic")

SITE_FORM_TYPES: dict[str, str] = {"name": "SITE", "number": "SITE", "city_site": "SITE"}
LOCATION_FORM_TYPES: dict[str, str] = {"full": "LOCATION", "city": "LOCATION", "region": "LOCATION"}

# Every type the generator can emit must be in the closed set. Checked at import
# so a typo in a form table fails before any corpus is written.
for _table in (PATIENT_FORM_TYPES, INVESTIGATOR_FORM_TYPES, SITE_FORM_TYPES, LOCATION_FORM_TYPES):
    for _form, _type in _table.items():
        if _type not in ENTITY_TYPE_SET:
            raise ImportError(f"form {_form!r} maps to unknown entity type {_type!r}")
del _table, _form, _type

DATE_KEYS: tuple[str, ...] = ("screening", "first_dose", "onset", "action", "resolution", "followup")
DATE_ANCHORS: dict[str, str | None] = {
    "screening": "first_dose",
    "first_dose": None,
    "onset": "first_dose",
    "action": "onset",
    "resolution": "onset",
    "followup": "resolution",
}
DATE_FORMATS: tuple[str, ...] = ("long", "dmy", "us", "iso")
ORDINALS = ("first", "second", "third", "fourth", "fifth")
NUMBER_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
    8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
}


@dataclass
class DateNode:
    key: str                # "P0.onset"
    role: str               # "onset"
    patient_key: str        # "P0"
    date: dt.date
    anchor_key: str | None  # "P0.first_dose"
    interval_days: int | None

    @property
    def iso(self) -> str:
        return self.date.isoformat()

    def render(self, fmt: str, first_dose: dt.date) -> str:
        d = self.date
        if fmt == "long":
            return f"{d.day} {MONTHS[d.month - 1]} {d.year}"
        if fmt == "dmy":
            return f"{d.day:02d}-{MONTHS_ABBR[d.month - 1]}-{d.year}"
        if fmt == "us":
            return f"{d.month:02d}/{d.day:02d}/{d.year}"
        if fmt == "iso":
            return d.isoformat()
        if fmt == "day":
            delta = (d - first_dose).days
            n = delta + 1 if delta >= 0 else delta
            return f"Day {n}"
        raise ValueError(f"unknown date format {fmt!r}")

    def all_absolute_forms(self) -> list[str]:
        return [self.render(f, self.date) for f in DATE_FORMATS]


@dataclass
class Patient:
    key: str
    idx: int
    first: str
    last: str
    sex: str                # "M" | "F"
    age: int
    title: str
    subject_id: str
    screening_no: str
    mrn: str
    nickname: str | None
    misspelled_last: str | None
    forms: dict[str, str]
    dates: dict[str, DateNode]
    allowed_any: tuple[str, ...]

    @property
    def canonical(self) -> str:
        return f"{self.first} {self.last}"

    @property
    def first_dose(self) -> dt.date:
        return self.dates["first_dose"].date

    def pronoun(self, case: str) -> str:
        table = {
            "M": {"subj": "he", "obj": "him", "poss": "his"},
            "F": {"subj": "she", "obj": "her", "poss": "her"},
        }
        return table[self.sex][case]

    @property
    def sex_noun(self) -> str:
        return "male" if self.sex == "M" else "female"

    @property
    def sex_word(self) -> str:
        return "man" if self.sex == "M" else "woman"


@dataclass
class Investigator:
    key: str
    idx: int
    first: str
    last: str
    role: str               # "principal" | "sub"
    forms: dict[str, str]

    @property
    def canonical(self) -> str:
        return f"{self.first} {self.last}"


@dataclass
class Site:
    key: str
    number: str
    name: str
    person_named: bool
    forms: dict[str, str]


@dataclass
class Location:
    key: str
    city: str
    region: str
    country: str
    forms: dict[str, str]


@dataclass
class CastSpec:
    n_patients: int = 1
    same_surname: bool = False
    force_nickname: bool = False        # patient 0 gets a nickname form
    force_misspelling: bool = False     # patient 0 gets a misspelled form
    person_named_site: bool = False
    namelike_drug: bool = False


@dataclass
class Cast:
    doc_index: int
    seed: bytes
    locale: str
    spec: CastSpec
    patients: list[Patient]
    investigators: list[Investigator]
    site: Site
    location: Location
    clinical: dict
    resamples: int

    @property
    def seed_hex(self) -> str:
        return self.seed.hex()

    def date_nodes(self) -> list[DateNode]:
        out: list[DateNode] = []
        for p in self.patients:
            for k in DATE_KEYS:
                out.append(p.dates[k])
        return out

    def entity_records(self) -> list[dict]:
        """Every entity with type, canonical and its full surface-form set.
        Order is cast order; emit renumbers by first mention."""
        recs: list[dict] = []
        for p in self.patients:
            forms = [v for k, v in p.forms.items()]
            recs.append({"key": p.key, "type": "PATIENT", "canonical": p.canonical, "forms": forms})
        for i in self.investigators:
            recs.append({"key": i.key, "type": "INVESTIGATOR", "canonical": i.canonical,
                         "forms": list(i.forms.values())})
        recs.append({"key": self.site.key, "type": "SITE", "canonical": self.site.name,
                     "forms": list(self.site.forms.values())})
        recs.append({"key": self.location.key, "type": "LOCATION",
                     "canonical": self.location.forms["full"],
                     "forms": list(self.location.forms.values())})
        for d in self.date_nodes():
            recs.append({"key": d.key, "type": "DATE", "canonical": d.iso,
                         "forms": d.all_absolute_forms(), "role": d.role, "patient": d.patient_key})
        return recs

    def scan_forms(self) -> list[tuple[str, str]]:
        """(entity_key, surface string) pairs gate 3 scans the text for.
        Relative date renderings are excluded (they are context-bound)."""
        pairs: list[tuple[str, str]] = []
        for rec in self.entity_records():
            for f in rec["forms"]:
                pairs.append((rec["key"], f))
        return pairs


# --------------------------------------------------------------------------- helpers


def _typo(word: str, rng: random.Random) -> str:
    """One edit-distance-1 change to ``word`` that keeps it capitalised."""
    ops = ["swap", "delete", "double", "vowel"]
    vowels = "aeiou"
    for _ in range(12):
        op = rng.choice(ops)
        w = list(word)
        if op == "swap" and len(w) >= 4:
            i = rng.randint(1, len(w) - 2)
            w[i], w[i + 1] = w[i + 1], w[i]
        elif op == "delete" and len(w) >= 5:
            i = rng.randint(1, len(w) - 2)
            del w[i]
        elif op == "double" and len(w) >= 4:
            i = rng.randint(1, len(w) - 2)
            w.insert(i, w[i])
        elif op == "vowel":
            idx = [i for i, c in enumerate(w) if c in vowels and i > 0]
            if not idx:
                continue
            i = rng.choice(idx)
            w[i] = rng.choice([v for v in vowels if v != w[i]])
        else:
            continue
        cand = "".join(w)
        if cand != word and cand.isalpha():
            return cand
    return word


def _tokens_collide(tokens: list[str], pool: set[str]) -> bool:
    return any(t in pool for t in tokens)


def _person_tokens(*names: str | None) -> list[str]:
    out: list[str] = []
    for n in names:
        if n:
            out.extend(words_of(n))
    return out


# --------------------------------------------------------------------------- builder


class CastBuilder:
    """Builds one deterministic cast for a document index."""

    def __init__(self, master_seed: bytes, vocab: Vocab | None = None,
                 extra_reserved: frozenset[str] = frozenset()):
        self.master_seed = master_seed
        self.vocab = vocab or load_vocab()
        self.reserved = self.vocab.reserved_words | extra_reserved

    # ---- public

    def build(self, doc_index: int, spec: CastSpec | None = None) -> Cast:
        spec = spec or CastSpec()
        seed = doc_seed(self.master_seed, doc_index)
        rng = rng_from_seed(seed, "cast")
        locale = "en_CA" if rng.random() < 0.20 else "en_US"
        fake = _faker(locale)
        fake.seed_instance(int.from_bytes(seed[:8], "big"))
        resamples = 0

        used_tokens: set[str] = set()

        location = self._location(rng, locale)
        used_tokens.update(words_of(location.city))
        used_tokens.update(words_of(location.region))

        site_number = str(rng.randint(1001, 4999))

        # Investigators first so a patient never shares a surname with one.
        investigators: list[Investigator] = []
        n_inv = 1 if rng.random() < 0.65 else 2
        for i in range(n_inv):
            while True:
                first, last = self._person_name(fake, rng)
                toks = _person_tokens(first, last)
                if _tokens_collide(toks, self.reserved) or _tokens_collide(toks, used_tokens):
                    resamples += 1
                    continue
                break
            used_tokens.update(toks)
            role = "principal" if i == 0 else "sub"
            investigators.append(self._investigator(i, first, last, role, site_number, rng))

        site, s_res = self._site(fake, rng, locale, location, site_number, spec.person_named_site, used_tokens)
        resamples += s_res
        used_tokens.update(words_of(site.name))

        patients: list[Patient] = []
        resamples += self._make_patients(
            fake, rng, spec, site_number, used_tokens, patients, range(spec.n_patients), set(), set(), set(),
        )

        clinical = self._clinical(rng, spec, len(patients))
        for p, pp in zip(patients, clinical["per_patient"]):
            pp["exposure_days"] = str(p.dates["onset"].interval_days)

        return Cast(
            doc_index=doc_index, seed=seed, locale=locale, spec=spec,
            patients=patients, investigators=investigators, site=site,
            location=location, clinical=clinical, resamples=resamples,
        )

    def extend(self, cast: Cast, doc_index: int, n_extra: int) -> Cast:
        """A new cast for a bundle listing: the same site, investigators,
        location and clinical context, the same patients (same keys), plus
        ``n_extra`` new patients drawn from ``doc_index``'s seed."""
        seed = doc_seed(self.master_seed, doc_index)
        rng = rng_from_seed(seed, "cast-extend")
        fake = _faker(cast.locale)
        fake.seed_instance(int.from_bytes(seed[:8], "big"))
        used_tokens: set[str] = set()
        used_tokens.update(words_of(cast.location.city))
        used_tokens.update(words_of(cast.location.region))
        used_tokens.update(words_of(cast.site.name))
        for inv in cast.investigators:
            used_tokens.update(_person_tokens(inv.first, inv.last))
        used_subj: set[int] = set()
        used_scr: set[int] = set()
        used_mrn: set[int] = set()
        initials_seen: set[str] = set()
        for p in cast.patients:
            used_tokens.update(_person_tokens(p.first, p.last, p.nickname, p.misspelled_last))
            used_subj.add(int(p.subject_id.split("-")[1]))
            used_scr.add(int(p.screening_no.split("-")[1]))
            used_mrn.add(int(p.mrn))
            initials_seen.add(p.first[0] + p.last[0])
        spec = CastSpec(n_patients=len(cast.patients) + n_extra, namelike_drug=cast.spec.namelike_drug,
                        person_named_site=cast.spec.person_named_site)
        patients = list(cast.patients)
        resamples = self._make_patients(
            fake, rng, spec, cast.site.number, used_tokens, patients,
            range(len(cast.patients), len(cast.patients) + n_extra), used_subj, used_scr, used_mrn,
            initials_seen=initials_seen,
        )
        extra_clinical = self._clinical(rng, spec, len(patients))
        clinical = dict(cast.clinical)
        clinical["per_patient"] = list(cast.clinical["per_patient"]) + extra_clinical["per_patient"][len(cast.patients):]
        for p, pp in zip(patients, clinical["per_patient"]):
            pp["exposure_days"] = str(p.dates["onset"].interval_days)
        return Cast(
            doc_index=doc_index, seed=seed, locale=cast.locale, spec=spec,
            patients=patients, investigators=list(cast.investigators), site=cast.site,
            location=cast.location, clinical=clinical, resamples=cast.resamples + resamples,
        )

    def _make_patients(self, fake: Faker, rng: random.Random, spec: CastSpec, site_number: str,
                       used_tokens: set[str], patients: list[Patient], indices: range,
                       used_subj: set[int], used_scr: set[int], used_mrn: set[int],
                       initials_seen: set[str] | None = None) -> int:
        """Append one patient per index in ``indices`` to ``patients``.
        Returns the number of resamples spent on collisions."""
        resamples = 0
        initials_seen = set() if initials_seen is None else initials_seen
        n_new = len(indices)
        subj_nums = rng.sample([n for n in range(1, 61) if n not in used_subj], n_new)
        scr_nums = rng.sample([n for n in range(10, 999) if n not in used_scr], n_new)
        mrns: list[int] = []
        while len(mrns) < n_new:
            m = rng.randint(1_000_000, 9_999_999)
            if m not in used_mrn and m not in mrns:
                mrns.append(m)
        for k, idx in enumerate(indices):
            while True:
                sex = "M" if rng.random() < 0.5 else "F"
                first = fake.first_name_male() if sex == "M" else fake.first_name_female()
                if spec.same_surname and idx > 0:
                    last = patients[0].last
                else:
                    last = fake.last_name()
                if not (first.isalpha() and last.isalpha()):
                    resamples += 1
                    continue
                initials = first[0] + last[0]
                want_nick = spec.force_nickname and idx == 0
                want_typo = spec.force_misspelling and idx == 0
                if want_nick and first not in self.vocab.nicknames:
                    # Pick a first name from the nickname map of the right sex.
                    pool = [k2 for k2 in self.vocab.nicknames if self._name_sex(k2) == sex]
                    first = rng.choice(pool)
                    initials = first[0] + last[0]
                nickname = self.vocab.nicknames.get(first)
                has_extra = want_nick or want_typo or rng.random() < 0.10
                if not has_extra:
                    nickname = None
                toks = _person_tokens(first, last, nickname)
                if spec.same_surname and idx > 0:
                    # Surname collision is deliberate; only the first name and
                    # the first initial must be distinct.
                    check_pool = used_tokens - set(words_of(last))
                    if first[0] == patients[0].first[0] or first == patients[0].first:
                        resamples += 1
                        continue
                else:
                    check_pool = used_tokens
                if (_tokens_collide(toks, self.reserved) or _tokens_collide(toks, check_pool)
                        or initials in initials_seen or len(last) < 3):
                    resamples += 1
                    continue
                misspelled = None
                if has_extra or want_typo:
                    for _ in range(6):
                        cand = _typo(last, rng)
                        ctoks = words_of(cand)
                        if cand != last and not _tokens_collide(ctoks, self.reserved) \
                                and not _tokens_collide(ctoks, used_tokens) \
                                and not _tokens_collide(ctoks, set(toks)):
                            misspelled = cand
                            break
                    if want_typo and misspelled is None:
                        resamples += 1
                        continue
                break
            used_tokens.update(toks)
            if misspelled:
                used_tokens.update(words_of(misspelled))
            initials_seen.add(initials)
            patients.append(self._patient(
                idx, first, last, sex, nickname, misspelled, site_number,
                subj_nums[k], scr_nums[k], mrns[k], rng, spec,
            ))
        return resamples

    # ---- pieces

    @staticmethod
    def _name_sex(first: str) -> str:
        female = {
            "Elizabeth", "Margaret", "Katherine", "Catherine", "Patricia", "Jennifer",
            "Jessica", "Rebecca", "Deborah", "Susan", "Barbara", "Dorothy", "Victoria",
            "Kimberly", "Christina", "Stephanie", "Samantha", "Alexandra", "Jacqueline",
            "Melissa", "Theresa", "Pamela", "Cynthia", "Abigail", "Madeline", "Eleanor",
            "Josephine",
        }
        return "F" if first in female else "M"

    @staticmethod
    def _person_name(fake: Faker, rng: random.Random) -> tuple[str, str]:
        for _ in range(50):
            first = fake.first_name()
            last = fake.last_name()
            if first.isalpha() and last.isalpha() and len(last) >= 3:
                return first, last
        return "Alden", "Ferrick"

    def _location(self, rng: random.Random, locale: str) -> Location:
        if locale == "en_CA":
            city, region = rng.choice(self.vocab.places_ca)
            country = "Canada"
            full = f"{city}, {region}, {country}"
        else:
            city, region = rng.choice(self.vocab.places_us)
            country = "United States"
            full = f"{city}, {region}"
        forms = {"full": full, "city": city, "region": region}
        return Location(key="L", city=city, region=region, country=country, forms=forms)

    def _site(self, fake: Faker, rng: random.Random, locale: str, loc: Location,
              number: str, person_named: bool, used: set[str]) -> tuple[Site, int]:
        centre = "Centre" if locale == "en_CA" else "Center"
        resamples = 0
        patterns = ["surname_memorial", "surname_institute", "city_crc", "city_imu", "city_compass"]
        if person_named:
            pattern = rng.choice(patterns[:2])
        else:
            pattern = rng.choice(patterns)
        if pattern.startswith("surname"):
            while True:
                surname = fake.last_name()
                toks = words_of(surname)
                if not surname.isalpha() or _tokens_collide(toks, self.reserved) or _tokens_collide(toks, used):
                    resamples += 1
                    continue
                break
            if pattern == "surname_memorial":
                name = f"{surname} Memorial Hospital"
            else:
                name = f"{surname} Clinical Research Institute"
            is_person = True
        elif pattern == "city_crc":
            name = f"{loc.city} Clinical Research {centre}"
            is_person = False
        elif pattern == "city_imu":
            name = f"{loc.city} Investigational Medicine Unit"
            is_person = False
        else:
            compass = rng.choice(["North", "South", "East", "West", "Central"])
            name = f"{loc.city} {compass} Medical {centre}"
            is_person = False
        forms = {"name": name, "number": f"Site {number}", "city_site": f"the {loc.city} site"}
        return Site(key="S", number=number, name=name, person_named=is_person, forms=forms), resamples

    def _investigator(self, idx: int, first: str, last: str, role: str,
                      site_number: str, rng: random.Random) -> Investigator:
        generic = "the principal investigator" if role == "principal" else "the sub-investigator"
        area = rng.randint(201, 989)
        phone = f"+1 ({area}) 555-01{rng.randint(0, 99):02d}"
        email = f"{first.lower()}.{last.lower()}@site{site_number}.example"
        forms = {
            "full": f"Dr. {first} {last}",
            "last": f"Dr. {last}",
            "plain": f"{first} {last}",
            "generic": generic,
            "phone": phone,
            "email": email,
        }
        return Investigator(key=f"I{idx}", idx=idx, first=first, last=last, role=role, forms=forms)

    def _patient(self, idx: int, first: str, last: str, sex: str, nickname: str | None,
                 misspelled: str | None, site_number: str, subj_num: int, scr_num: int,
                 mrn: int, rng: random.Random, spec: CastSpec) -> Patient:
        key = f"P{idx}"
        age = int(rng.triangular(18, 89, 62))
        if sex == "M":
            title = "Mr."
        else:
            title = "Mrs." if rng.random() < 0.35 else "Ms."
        subject_id = f"{site_number}-{subj_num:03d}"
        screening_no = f"SCR-{scr_num:04d}"
        forms: dict[str, str] = {
            "full": f"{first} {last}",
            "first": first,
            "title_last": f"{title} {last}",
            "initials": f"{first[0]}. {last}",
            "initials_dot": f"{first[0]}.{last[0]}.",
            "generic_patient": "the patient",
            "generic_subject": "the subject",
            "subject": f"Subject {subject_id}",
            "id": subject_id,
            "screening": f"Screening No. {screening_no}",
            "mrn": f"MRN {mrn}",
            "age": f"{age}-year-old",
            "aged": f"aged {age}",
        }
        if idx < len(ORDINALS):
            forms["ordinal_patient"] = f"the {ORDINALS[idx]} patient"
            forms["ordinal_subject"] = f"the {ORDINALS[idx]} subject"
        if nickname:
            forms["nickname"] = f"{nickname} {last}"
            forms["nickname_first"] = nickname
        if misspelled:
            forms["misspelled"] = f"{first} {misspelled}"
            forms["misspelled_title"] = f"{title} {misspelled}"

        allowed_any = list(PATIENT_ANY_FORMS)
        if spec.same_surname and spec.n_patients > 1:
            # "Mr. Chen" cannot be resolved between two Chens; never teach guessing.
            allowed_any.remove("title_last")
        if spec.n_patients > 1:
            # In multi-patient documents a bare first name is fine, but a bare
            # "the patient" is only used inside paragraphs about one patient;
            # the skeleton controls that, so nothing to remove here.
            pass

        dates = self._date_chain(key, rng)
        return Patient(
            key=key, idx=idx, first=first, last=last, sex=sex, age=age, title=title,
            subject_id=subject_id, screening_no=screening_no, mrn=str(mrn),
            nickname=nickname, misspelled_last=misspelled, forms=forms, dates=dates,
            allowed_any=tuple(allowed_any),
        )

    @staticmethod
    def _date_chain(pkey: str, rng: random.Random) -> dict[str, DateNode]:
        year = rng.randint(2018, 2024)
        first_dose = dt.date(year, rng.randint(1, 12), rng.randint(1, 28))
        screening = first_dose - dt.timedelta(days=rng.randint(7, 28))
        onset = first_dose + dt.timedelta(days=rng.randint(1, 60))
        action = onset + dt.timedelta(days=rng.randint(1, 5))
        resolution = action + dt.timedelta(days=rng.randint(1, 40))
        followup = resolution + dt.timedelta(days=rng.randint(7, 90))
        dates = {
            "screening": screening, "first_dose": first_dose, "onset": onset,
            "action": action, "resolution": resolution, "followup": followup,
        }
        nodes: dict[str, DateNode] = {}
        for role in DATE_KEYS:
            anchor_role = DATE_ANCHORS[role]
            anchor_key = f"{pkey}.{anchor_role}" if anchor_role else None
            interval = (dates[role] - dates[anchor_role]).days if anchor_role else None
            nodes[role] = DateNode(
                key=f"{pkey}.{role}", role=role, patient_key=pkey, date=dates[role],
                anchor_key=anchor_key, interval_days=interval,
            )
        return nodes

    def _clinical(self, rng: random.Random, spec: CastSpec, n_patients: int) -> dict:
        v = self.vocab
        c = v.clinical
        drug = rng.choice(v.drugs_namelike) if spec.namelike_drug else rng.choice(v.drugs)
        prefix = rng.choice(c["study_prefixes"])
        study = f"{prefix}-{rng.randint(1000, 9999)}-{rng.randint(101, 399)}"
        ae_pool = list(v.ae_terms)
        rng.shuffle(ae_pool)
        per_patient: list[dict] = []
        for i in range(n_patients):
            ae = ae_pool[i]
            ae_other = ae_pool[n_patients + i]
            labs = rng.sample(v.labs, 3)
            hospitalized = rng.random() < 0.4
            if hospitalized:
                seriousness = "serious (hospitalisation)"
            else:
                seriousness = rng.choice(["not serious", "not serious", "serious (medically important event)"])
            per_patient.append({
                "ae": ae,
                "ae_other": ae_other,
                "severity": rng.choice(c["severities"]),
                "outcome": rng.choice(c["outcomes"]),
                "causality": rng.choice(c["causality"]),
                "action": rng.choice(c["actions"]),
                "history": rng.sample(c["histories"], 2),
                "conmeds": rng.sample(c["conmeds"], 2),
                "labs": labs,
                "hospitalized": hospitalized,
                "seriousness": seriousness,
                "weight": str(rng.randint(48, 118)),
                "exposure_days": None,  # filled after date chains exist

                "smoking": rng.choice(["non-smoker", "non-smoker", "former smoker", "current smoker"]),
                "ecog": str(rng.choice([0, 0, 1, 1, 2])),
            })
        return {
            "drug": drug,
            "dose": rng.choice(c["doses"]),
            "route": rng.choice(c["routes"]),
            "freq": rng.choice(c["frequencies"]),
            "indication": rng.choice(c["indications"]),
            "study": study,
            "per_patient": per_patient,
        }


def build_listing_cast(builder: CastBuilder, doc_index: int, n_rows: int,
                       spec: CastSpec | None = None) -> Cast:
    """A cast with ``n_rows`` patients for a standalone listing."""
    spec = spec or CastSpec()
    spec.n_patients = n_rows
    return builder.build(doc_index, spec)
