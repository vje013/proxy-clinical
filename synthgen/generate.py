"""Sample plan and generation loop.

``build_plan`` turns (n, composition, master seed) into an ordered list of
sample specs: which slice, how many patients, which hard-case kinds, and which
bundle a sample belongs to. ``generate`` renders them. Everything downstream of
the master seed is deterministic; nothing here touches the network.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import GENERATOR_VERSION
from .cast import Cast, CastBuilder, CastSpec, doc_seed, rng_from_seed
from .emit import DEFAULT_INSTRUCTION_VERSION, INSTRUCTIONS, assign_entity_ids, build_record
from .hardcases import (
    KIND_ID_ONLY, KIND_INITIALS_FAR, KIND_MISSPELLING, KIND_NAMELIKE, KIND_NICKNAME,
    KIND_PROSE_DATE, KIND_SAME_SURNAME, HardCasePlan, plan as plan_hard, verify_all,
)
from .render import ListingLayout, Rendered, Renderer, TemplateSet, cross_reference_ok, load_templates


# --------------------------------------------------------------------------- composition


@dataclass(frozen=True)
class Composition:
    """Slice shares. Scale the corpus by changing ``n`` only."""
    single: float = 0.52          # single-patient narratives
    multi: float = 0.24           # multi-patient narratives
    listing: float = 0.16         # standalone listings
    bundle_docs: float = 0.08     # narrative+listing bundle documents (pairs = half)
    hard: float = 0.10            # hard-case flagged share, taken across the slices above
    three_patient_share: float = 0.15   # of multi-patient narratives

    def counts(self, n: int) -> dict[str, int]:
        shares = {"single": self.single, "multi": self.multi, "listing": self.listing,
                  "bundle_docs": self.bundle_docs}
        raw = {k: n * v for k, v in shares.items()}
        counts = {k: int(v) for k, v in raw.items()}
        # Largest-remainder rounding so the slices sum to n.
        remainder = n - sum(counts.values())
        for k, _ in sorted(raw.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True):
            if remainder <= 0:
                break
            counts[k] += 1
            remainder -= 1
        if counts["bundle_docs"] % 2 == 1:
            counts["bundle_docs"] -= 1
            counts["single"] += 1
        counts["hard"] = int(round(n * self.hard))
        return counts


@dataclass
class SampleSpec:
    slice: str                       # single | multi | listing | bundle_narrative | bundle_listing
    n_patients: int
    hard_kinds: list[str] = field(default_factory=list)
    bundle_no: int | None = None
    listing_rows: int = 0
    doc_index: int = -1
    sample_id: str = ""
    template_hint: int = 0           # rotation offset so templates are used evenly within a slice


_SINGLE_HARD_CYCLE = (
    [KIND_PROSE_DATE], [KIND_MISSPELLING], [KIND_NICKNAME], [KIND_INITIALS_FAR], [KIND_NAMELIKE],
    [KIND_PROSE_DATE, KIND_INITIALS_FAR], [KIND_MISSPELLING, KIND_NICKNAME],
    [KIND_NICKNAME, KIND_PROSE_DATE], [KIND_INITIALS_FAR, KIND_NAMELIKE], [KIND_MISSPELLING, KIND_PROSE_DATE],
)
_MULTI_HARD_CYCLE = (
    [KIND_SAME_SURNAME], [KIND_SAME_SURNAME, KIND_PROSE_DATE], [KIND_SAME_SURNAME, KIND_INITIALS_FAR],
    [KIND_SAME_SURNAME, KIND_NAMELIKE], [KIND_PROSE_DATE, KIND_NAMELIKE], [KIND_SAME_SURNAME, KIND_NICKNAME],
)
_LISTING_HARD_CYCLE = ([KIND_ID_ONLY], [KIND_ID_ONLY, KIND_NAMELIKE])


def build_plan(n: int, composition: Composition, master_seed: bytes, prefix: str = "pilot") -> list[SampleSpec]:
    counts = composition.counts(n)
    rng = rng_from_seed(doc_seed(master_seed, 0xFFFF_FFFF), "plan")

    units: list[list[SampleSpec]] = []
    for _ in range(counts["single"]):
        units.append([SampleSpec("single", 1)])
    for _ in range(counts["multi"]):
        k = 3 if rng.random() < composition.three_patient_share else 2
        units.append([SampleSpec("multi", k)])
    for _ in range(counts["listing"]):
        units.append([SampleSpec("listing", 0, listing_rows=rng.randint(5, 20))])
    for b in range(counts["bundle_docs"] // 2):
        k = 2 if rng.random() < 0.3 else 1
        units.append([
            SampleSpec("bundle_narrative", k, bundle_no=b),
            SampleSpec("bundle_listing", k, bundle_no=b, listing_rows=k + rng.randint(4, 12)),
        ])

    # Hard-case allocation: 40% single, 30% multi, 20% listing, 10% bundle listings.
    hard = counts["hard"]
    alloc = {"single": int(hard * 0.4), "multi": int(hard * 0.3), "listing": int(hard * 0.2)}
    alloc["bundle_listing"] = hard - sum(alloc.values())
    cycles = {"single": _SINGLE_HARD_CYCLE, "multi": _MULTI_HARD_CYCLE,
              "listing": _LISTING_HARD_CYCLE, "bundle_listing": ([KIND_ID_ONLY],)}
    for slice_name, want in alloc.items():
        candidates = [s for unit in units for s in unit if s.slice == slice_name]
        chosen = candidates[:want] if want <= len(candidates) else candidates
        cyc = cycles[slice_name]
        for i, s in enumerate(chosen):
            kinds = list(cyc[i % len(cyc)])
            if KIND_SAME_SURNAME in kinds and s.n_patients < 2:
                s.n_patients = 2
            s.hard_kinds = kinds

    rng.shuffle(units)
    plan: list[SampleSpec] = []
    hint_counters: dict[tuple[str, int], int] = {}
    for unit in units:
        for s in unit:
            s.doc_index = len(plan)
            s.sample_id = f"{prefix}-{s.doc_index:06d}"
            hk = (s.slice, s.n_patients)
            s.template_hint = hint_counters.get(hk, 0)
            hint_counters[hk] = s.template_hint + 1
            plan.append(s)
    return plan


# --------------------------------------------------------------------------- generation


class GenerationError(RuntimeError):
    pass


class Generator:
    def __init__(self, master_seed: bytes, templates: TemplateSet | None = None):
        self.ts = templates or load_templates()
        self.builder = CastBuilder(master_seed, extra_reserved=self.ts.static_words)
        self.renderer = Renderer(self.ts)
        self.master_seed = master_seed

    # ---- narratives

    def _render_narrative(self, cast: Cast, hard: HardCasePlan, seed: bytes, hint: int = 0) -> Rendered:
        skels = self.ts.skeletons_for(len(cast.patients))
        if not skels:
            raise GenerationError(f"no skeleton for {len(cast.patients)} patients")
        # Even rotation through the skeletons (hint), falling through to the
        # others when a hard-case pattern cannot be satisfied.
        start = hint % len(skels)
        order = skels[start:] + skels[:start]
        for sk in order:
            for attempt in range(8):
                rng = rng_from_seed(seed, f"render:{sk.id}:{attempt}")
                rendered = self.renderer.render_narrative(cast, sk, rng, hard.policy)
                ok, _ = cross_reference_ok(rendered, len(cast.patients))
                if ok and verify_all(hard.kinds, cast, rendered):
                    return rendered
        raise GenerationError(f"could not satisfy {hard.kinds} for doc {cast.doc_index}")

    def _render_listing(self, cast: Cast, hard: HardCasePlan, seed: bytes,
                        row_patients: list[int] | None = None, hint: int = 0) -> tuple[Rendered, ListingLayout]:
        layouts = self.ts.listings_for(hard.listing_names)
        if not layouts:
            raise GenerationError("no listing layout matches the plan")
        start = hint % len(layouts)
        for layout in layouts[start:] + layouts[:start]:
            for attempt in range(6):
                rrng = rng_from_seed(seed, f"render:{layout.id}:{attempt}")
                rendered = self.renderer.render_listing(cast, layout, rrng, hard.policy, row_patients)
                if verify_all(hard.kinds, cast, rendered, layout):
                    return rendered, layout
        raise GenerationError(f"could not satisfy {hard.kinds} for listing {cast.doc_index}")

    # ---- samples

    def generate(self, plan: list[SampleSpec]) -> list[dict]:
        records: list[dict] = []
        bundle_state: dict[int, tuple[Cast, dict[str, str]]] = {}
        for spec in plan:
            seed = doc_seed(self.master_seed, spec.doc_index)
            hrng = rng_from_seed(seed, "hard")
            bundle_id = None if spec.bundle_no is None else f"{spec.sample_id.rsplit('-', 1)[0]}-bundle-{spec.bundle_no:04d}"

            if spec.slice in ("single", "multi", "bundle_narrative"):
                hard = plan_hard(spec.hard_kinds, CastSpec(n_patients=spec.n_patients), hrng)
                cast = self.builder.build(spec.doc_index, hard.spec)
                rendered = self._render_narrative(cast, hard, seed, spec.template_hint)
                ids = assign_entity_ids(cast, [rendered])
                if spec.slice == "bundle_narrative":
                    bundle_state[spec.bundle_no] = (cast, ids)  # type: ignore[index]
                records.append(build_record(spec.sample_id, cast, rendered, ids, spec.hard_kinds, bundle_id, spec.slice))

            elif spec.slice == "listing":
                hard = plan_hard(spec.hard_kinds, CastSpec(n_patients=spec.listing_rows), hrng)
                cast = self.builder.build(spec.doc_index, hard.spec)
                rendered, _ = self._render_listing(cast, hard, seed, hint=spec.template_hint)
                ids = assign_entity_ids(cast, [rendered])
                records.append(build_record(spec.sample_id, cast, rendered, ids, spec.hard_kinds, bundle_id, spec.slice))

            elif spec.slice == "bundle_listing":
                if spec.bundle_no not in bundle_state:
                    raise GenerationError("bundle listing planned before its narrative")
                narr_cast, narr_ids = bundle_state[spec.bundle_no]
                n_extra = max(spec.listing_rows - len(narr_cast.patients), 3)
                cast = self.builder.extend(narr_cast, spec.doc_index, n_extra)
                hard = plan_hard(spec.hard_kinds, cast.spec, hrng)
                order = list(range(len(cast.patients)))
                rng_from_seed(seed, "rows").shuffle(order)
                rendered, _ = self._render_listing(cast, hard, seed, row_patients=order, hint=spec.template_hint)
                ids = assign_entity_ids(cast, [rendered], existing=narr_ids)
                records.append(build_record(spec.sample_id, cast, rendered, ids, spec.hard_kinds, bundle_id, spec.slice))
            else:
                raise GenerationError(f"unknown slice {spec.slice}")
        return records


def corpus_meta(n: int, master_seed_hex: str, composition: Composition, counts: dict[str, int],
                instruction_version: str = DEFAULT_INSTRUCTION_VERSION) -> dict:
    return {
        "generator_version": GENERATOR_VERSION,
        "master_seed": master_seed_hex,
        "n": n,
        "composition": vars(composition),
        "counts": counts,
        "instruction_version": instruction_version,
        "instruction": INSTRUCTIONS[instruction_version],
        "split": {"rule": "sha256(bundle_id or sample_id) mod 10000 < 1000 -> val", "val_share": 0.10},
        "entity_id_rule": "corpus: first mention order (bundle-wide); training view: first mention order per sample",
    }
