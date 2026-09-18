# Proxy Clinical: synthetic de-identification corpus generator (Block 1)

`synthgen` produces clinical-trial text (CSR-style adverse-event narratives and
patient listings) in which every identity is planted by the generator, so every
label is known exactly and for free. It is the training-data stage for the
Darwin Proxy Clinical de-identification pipeline. Nothing in the required path
touches the network or a language model.

## Install and run

```bash
pip install -e ".[dev]"
pytest
synthgen generate --n 500 --seed cafebabecafebabecafebabecafebabecafebabecafebabecafebabecafebabe --out data/pilot/
synthgen validate --corpus data/pilot/corpus.jsonl
synthgen check-determinism --n 100 --seed <hex>
```

`generate` writes `corpus.jsonl` (labelled corpus), `train.jsonl` / `val.jsonl`
(instruction-format training view), `meta.json`, and `pilot_report.md`. It runs
every gate, regenerates the corpus a second time and byte-compares it, and
exits non-zero if anything fails. Scaling to 3,000 to 5,000 samples is
`--n 5000`; the slice shares live in `Composition` and can be overridden with
`--single/--multi/--listing/--bundle-docs/--hard`.

## How it works

`cast.py` builds a per-document cast from `HMAC-SHA256(master_seed, doc_index)`:
patients (with a precomputed surface-form set: full name, `Mr. Chen`,
`R. Chen`, `R.C.`, first name, `the patient`, `the subject`, `Subject 1104-002`,
screening number, MRN, age forms, and for some patients a nickname and a
single-edit misspelling), investigators, a site, a location, and a date chain
per patient (screening, first dose, onset, action, resolution, follow-up) whose
intervals are stored as ground truth. Faker names that collide with the vocab,
with words used in the templates, or with another cast member are resampled
and the count is reported.

`render.py` fills YAML skeletons (`templates/narr-skel-*.yaml`) composed from
sentence banks (`templates/_banks.yaml`) and listing layouts
(`templates/listing-*.yaml`). Each skeleton pins the surface form for its
slots (`{patient:initials}`) or leaves it open (`{patient:any}`), so the
skeleton controls the cross-reference pattern. Labels are emitted while the
text is assembled; offsets are exact by construction and nothing is ever
re-located by searching the text. Distractors (invented drug names, MedDRA-like
terms, lab values, conmeds) are filled from `vocab/` and never labelled.

`hardcases.py` implements injectors 1 to 7 by shaping the cast and pinning
forms through a render policy, then verifying the pattern after rendering.
`generate.py` builds the plan (slices, bundles, hard-case allocation, even
template rotation) and drives generation. `validate.py` holds the gates and the
report writer. `emit.py` builds records, assigns entity ids by first mention,
produces the training view, and splits train/val by hash of `bundle_id` (when
present) or `sample_id`, so a bundle never straddles the split.

## Conventions worth knowing

- Entity ids in the corpus follow first-mention order; in a bundle the listing
  continues the narrative's numbering. The training view renumbers per sample
  so a model always sees `E1` first.
- `ID` and `AGE` mentions carry the patient's entity id. Dates are their own
  entities; the same date rendered as `14 March 2023` and `14-Mar-2023` shares
  one id. `Day 28` and prose-relative dates carry
  `attrs: {relative: true, anchor: <entity>}` and are checked arithmetically.
- In multi-patient documents `the patient` / `the subject` are only used before
  a second patient has been introduced; afterwards the renderer uses
  `the first patient` / `the second subject` style ordinals, so no reference is
  left ambiguous. Same-surname pairs never use `Mr. Chen`.
- A hospital named after a person is one `SITE` span; the person inside it is
  not an entity.
- The bare age number in a listing column is labelled `AGE`; it is known to gate 2
  through `no_scan_forms` and excluded from the gate 3 scan.
- `--paraphrase` sends narratives to the Anthropic API and keeps a paraphrase
  only if every span relocates exactly and the result passes the coverage and
  distractor gates. It is off by default and the pilot ships without it.

## Not in this block

Model training, the surrogate engine, receipts, and the utility check.
