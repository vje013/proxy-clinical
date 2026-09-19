"""The closed entity-type set. One definition, imported everywhere a type
name is produced or consumed: the generator's form tables, the validator,
the training-view emitter, the strict prediction parsers (v1 and v2) and
the de-identification policy file. A type that is not in this tuple is a
contract violation at every one of those points, never a soft warning.

Adding a type is a deliberate act: it changes the model's output contract
(``instruction_version`` must bump), the policy file must say how the new
type is surrogated, and the corpus must be regenerated.
"""
from __future__ import annotations

ENTITY_TYPES: tuple[str, ...] = (
    "PATIENT",        # a trial participant, by name or by a generic/ordinal reference
    "INVESTIGATOR",   # site staff named in the document
    "SITE",           # the investigational site, by name or number
    "LOCATION",       # city / region / country of the site
    "DATE",           # absolute or relative date expression
    "ID",             # subject / screening / medical-record identifiers
    "AGE",            # a participant's age
    "CONTACT",        # phone / email of a person
)

ENTITY_TYPE_SET: frozenset[str] = frozenset(ENTITY_TYPES)


def is_entity_type(value: object) -> bool:
    return isinstance(value, str) and value in ENTITY_TYPE_SET


def check_entity_type(value: object, where: str = "") -> str:
    """Return ``value`` if it is a known type, else raise ValueError."""
    if not is_entity_type(value):
        raise ValueError(f"unknown entity type {value!r}{(' in ' + where) if where else ''}; "
                         f"allowed: {', '.join(ENTITY_TYPES)}")
    return value  # type: ignore[return-value]
