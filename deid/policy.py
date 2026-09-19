"""Policy file loader. The policy is data, the code only executes it.

The loader is strict: an unknown method, a type without an entry, a type set
that differs from the closed enum, or a nonsense shift window is a
``PolicyError``. The policy's identity for receipts is (policy_id,
sha256 of the file bytes exactly as shipped).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from synthgen.types import ENTITY_TYPES

DEFAULT_POLICY = Path(__file__).resolve().parent.parent / "policies" / "ema-0070-v0.yaml"

METHODS_BY_TYPE = {
    "PATIENT": {"surrogate_name"},
    "INVESTIGATOR": {"surrogate_name"},
    "SITE": {"surrogate_site"},
    "LOCATION": {"surrogate_place"},
    "DATE": {"shift"},
    "ID": {"digit_substitution"},
    "AGE": {"safe_harbor_90", "passthrough"},
    "CONTACT": {"surrogate_contact"},
}
SCOPES = {"document_bundle", "document"}
RELATIVE_ACTIONS = {
    "study_day": {"passthrough"},
    "weekday": {"passthrough"},
    "month_anchored_prose": {"rerender_month"},
    "unresolvable": {"drop_month_and_report", "passthrough_and_report"},
}


class PolicyError(ValueError):
    pass


@dataclass(frozen=True)
class DateShift:
    unit: str
    min_weeks: int
    max_weeks: int
    direction: str


@dataclass(frozen=True)
class Policy:
    policy_id: str
    version: int
    sha256: str
    path: str
    scope: str
    types: dict            # raw per-type config, validated
    date_shift: DateShift
    age_threshold: int | None
    age_cap_label: str
    raw: dict

    def method(self, entity_type: str) -> str:
        return self.types[entity_type]["method"]

    def relative_action(self, kind: str) -> str:
        return self.types["DATE"]["relative_expressions"][kind]

    def identity(self) -> dict:
        return {"id": self.policy_id, "version": self.version, "sha256": self.sha256}


def load_policy(path: str | Path = DEFAULT_POLICY) -> Policy:
    p = Path(path)
    if not p.is_file():
        raise PolicyError(f"policy file {p} not found")
    data = p.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    try:
        raw = yaml.safe_load(data)
    except yaml.YAMLError as exc:
        raise PolicyError(f"{p}: not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise PolicyError(f"{p}: top level must be a mapping")

    for key in ("policy_id", "policy_version", "entity_types", "surrogate_consistency", "types",
                "generic_references", "residual_scan"):
        if key not in raw:
            raise PolicyError(f"{p}: missing required key {key!r}")
    pid = raw["policy_id"]
    if not isinstance(pid, str) or not pid:
        raise PolicyError("policy_id must be a non-empty string")
    version = raw["policy_version"]
    if not isinstance(version, int) or isinstance(version, bool) or version < 0:
        raise PolicyError("policy_version must be a non-negative integer")

    if tuple(raw["entity_types"]) != ENTITY_TYPES:
        raise PolicyError(f"entity_types {raw['entity_types']} != closed set {list(ENTITY_TYPES)}")

    sc = raw["surrogate_consistency"]
    if not isinstance(sc, dict) or sc.get("scope") not in SCOPES:
        raise PolicyError(f"surrogate_consistency.scope must be one of {sorted(SCOPES)}")

    if raw["generic_references"].get("action") != "passthrough":
        raise PolicyError("generic_references.action must be 'passthrough' (the only implemented action)")

    types = raw["types"]
    if not isinstance(types, dict) or set(types) != set(ENTITY_TYPES):
        missing = set(ENTITY_TYPES) - set(types or {})
        extra = set(types or {}) - set(ENTITY_TYPES)
        raise PolicyError(f"types must have exactly one entry per entity type; missing {sorted(missing)}, extra {sorted(extra)}")
    for t, cfg in types.items():
        if not isinstance(cfg, dict) or "method" not in cfg:
            raise PolicyError(f"types.{t} must be a mapping with a 'method'")
        if cfg["method"] not in METHODS_BY_TYPE[t]:
            raise PolicyError(f"types.{t}.method {cfg['method']!r} not in {sorted(METHODS_BY_TYPE[t])}")

    d = types["DATE"]
    sh = d.get("shift")
    if not isinstance(sh, dict):
        raise PolicyError("types.DATE.shift is required")
    if sh.get("unit") != "weeks":
        raise PolicyError("types.DATE.shift.unit must be 'weeks' (day-of-week preservation is an invariant)")
    mn, mx = sh.get("min_weeks"), sh.get("max_weeks")
    if not (isinstance(mn, int) and isinstance(mx, int) and 1 <= mn <= mx):
        raise PolicyError("types.DATE.shift.min_weeks/max_weeks must be integers with 1 <= min <= max")
    if sh.get("direction") not in ("backward", "forward", "either"):
        raise PolicyError("types.DATE.shift.direction must be backward, forward or either")
    rel = d.get("relative_expressions")
    if not isinstance(rel, dict) or set(rel) != set(RELATIVE_ACTIONS):
        raise PolicyError(f"types.DATE.relative_expressions must have exactly {sorted(RELATIVE_ACTIONS)}")
    for k, v in rel.items():
        if v not in RELATIVE_ACTIONS[k]:
            raise PolicyError(f"types.DATE.relative_expressions.{k} {v!r} not in {sorted(RELATIVE_ACTIONS[k])}")

    a = types["AGE"]
    threshold: int | None = None
    cap_label = ""
    if a["method"] == "safe_harbor_90":
        threshold = a.get("threshold")
        if threshold != 90:
            raise PolicyError("safe_harbor_90 requires threshold: 90 (45 CFR 164.514(b)(2)(i)(C))")
        cap_label = a.get("cap_label")
        if not isinstance(cap_label, str) or not cap_label:
            raise PolicyError("types.AGE.cap_label must be a non-empty string")
    if "jitter" in (a.get("excluded_methods") or []) and a["method"] == "jitter":  # pragma: no cover
        raise PolicyError("AGE method jitter is excluded by the policy itself")

    if raw["residual_scan"].get("action") != "fail_on_hit":
        raise PolicyError("residual_scan.action must be 'fail_on_hit'")

    return Policy(
        policy_id=pid, version=version, sha256=digest, path=str(p), scope=sc["scope"], types=types,
        date_shift=DateShift(unit="weeks", min_weeks=mn, max_weeks=mx, direction=sh["direction"]),
        age_threshold=threshold, age_cap_label=cap_label, raw=raw,
    )
