"""Fixed vocabulary lists shipped with the generator.

Everything in here is either a distractor (drugs, AE terms, labs, clinical
filler) or an identity ingredient (nickname map, places). Loading is done once
and cached; all lists preserve file order so downstream sampling is
deterministic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

_HERE = Path(__file__).parent

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def _read_lines(name: str) -> list[str]:
    out: list[str] = []
    for raw in (_HERE / name).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


@dataclass(frozen=True)
class Vocab:
    drugs: tuple[str, ...]
    drugs_namelike: tuple[str, ...]
    ae_terms: tuple[str, ...]
    labs: tuple[str, ...]
    nicknames: dict[str, str]
    places_us: tuple[tuple[str, str], ...]
    places_ca: tuple[tuple[str, str], ...]
    clinical: dict[str, list[str]]
    # Every distractor phrase, for gate 4 (distractor purity).
    distractor_phrases: tuple[str, ...] = field(default=())
    # Lower-cased single words that a cast name may never equal.
    reserved_words: frozenset[str] = field(default=frozenset())

    @property
    def all_drugs(self) -> tuple[str, ...]:
        return self.drugs + self.drugs_namelike


MONTHS = (
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
)
MONTHS_ABBR = tuple(m[:3] for m in MONTHS)
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

# Words that appear in surface-form scaffolding or date rendering; a Faker name
# equal to one of these would make gate 3 ambiguous, so they are reserved.
_STRUCTURAL_RESERVED = (
    "the", "patient", "subject", "site", "day", "dr", "mr", "ms", "mrs",
    "screening", "no", "mrn", "study", "protocol", "visit", "dose", "first",
    "second", "third", "both", "principal", "investigator", "following",
    "weeks", "days", "after", "before", "later", "prior", "approximately",
)


def words_of(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


@lru_cache(maxsize=1)
def load_vocab() -> Vocab:
    drugs = tuple(_read_lines("drugs.txt"))
    drugs_namelike = tuple(_read_lines("drugs_namelike.txt"))
    ae_terms = tuple(_read_lines("ae_terms.txt"))
    labs = tuple(_read_lines("labs.txt"))
    nick: dict[str, str] = {}
    for line in _read_lines("nicknames.tsv"):
        formal, short = line.split("\t")
        nick[formal.strip()] = short.strip()
    places_us = tuple(tuple(p.split("|", 1)) for p in _read_lines("places_us.txt"))  # type: ignore[misc]
    places_ca = tuple(tuple(p.split("|", 1)) for p in _read_lines("places_ca.txt"))  # type: ignore[misc]
    clinical = yaml.safe_load((_HERE / "clinical.yaml").read_text(encoding="utf-8"))

    distractors: list[str] = []
    distractors.extend(drugs)
    distractors.extend(drugs_namelike)
    distractors.extend(ae_terms)
    distractors.extend(labs)
    for key in ("indications", "histories", "conmeds"):
        distractors.extend(clinical[key])

    reserved: set[str] = set(_STRUCTURAL_RESERVED)
    for phrase in distractors:
        reserved.update(words_of(phrase))
    for key, values in clinical.items():
        for phrase in values:
            reserved.update(words_of(phrase))
    reserved.update(m.lower() for m in MONTHS)
    reserved.update(m.lower() for m in MONTHS_ABBR)
    reserved.update(w.lower() for w in WEEKDAYS)

    return Vocab(
        drugs=drugs,
        drugs_namelike=drugs_namelike,
        ae_terms=ae_terms,
        labs=labs,
        nicknames=nick,
        places_us=places_us,  # type: ignore[arg-type]
        places_ca=places_ca,  # type: ignore[arg-type]
        clinical=clinical,
        distractor_phrases=tuple(distractors),
        reserved_words=frozenset(reserved),
    )
