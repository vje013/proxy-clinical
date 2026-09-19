# Proxy Clinical

## Block 1: synthetic de-identification corpus generator

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

Model training (Block 2), the surrogate engine, receipts and the utility check (Block 3).

---

# Block 2: LoRA tagger training, deterministic inference, evaluation

`lora/` fine-tunes a 3B instruct model on the Block 1 training view and
evaluates it with production decoding. Target hardware is a Colab A100
(high-RAM); the repo is the source of truth and `notebooks/colab_launcher.ipynb`
is a thin launcher that mounts Drive, clones the bundle, installs
`requirements-colab.txt`, copies the pilot JSONL, and runs one command.
`notebooks/colab_launcher_files.ipynb` does the same from a tarball uploaded to
the Colab Files panel (no Drive); it zips and downloads the run folder at the end,
since `/content` does not outlive the runtime.

```bash
pip uninstall -y torchvision torchaudio torchao bitsandbytes   # Colab preinstalls; unused here and break once torch is replaced
pip install -r requirements-colab.txt            # exact pins; the run manifest records what actually loaded
bash scripts/preflight.sh                        # import probes + one-step CPU training on the tiny stand-in; fails fast
bash scripts/run_pilot.sh configs/pilot.yaml runs/pilot-qwen2.5-3b
```

`run_pilot.sh` is train, then greedy inference on `val.jsonl`, then per-slice
evaluation, then the two-run determinism test. Re-running with the same output
directory resumes from the latest checkpoint (`--resume-if-exists`); pointing
the output directory at Drive is how a Colab run survives a disconnect.
Checkpoints are written every `save_steps` (100). Every run leaves one folder
with `adapter/`, `run_manifest.json`, `config.yaml`, `predictions.jsonl`,
`eval_report.md`, `eval.json`, `determinism.json`, and `pip_freeze.txt`.

Configs: `configs/pilot.yaml` (Qwen2.5-3B-Instruct, pinned to commit
`aa8e7253…`), `configs/pilot-llama.yaml` (Llama-3.2-3B-Instruct fallback,
`0cb88a4f…`, gated: needs `HF_TOKEN`), `configs/smoke-cpu.yaml` (tiny random
stand-in for CPU proofs). A Hub model without a 40-hex revision is a config
error. LoRA is r=16, alpha=32, dropout 0.05 on q/k/v/o and gate/up/down, trained
with TRL's SFT loop, completion-only loss, no packing.

Data contract: every line of `train.jsonl`/`val.jsonl` must carry the configured
`instruction_version` (v2 by default) and the identical instruction text; a mismatch is a
hard error at load, as is any sample longer than `train.max_length` (truncating
a completion would teach the model to stop early). The user message the model
sees is built in exactly one place (`lora.data.build_user_message`) for both
training and inference.

Output contract (v2, text anchors). The model emits
`{"mentions":[{"text":S,"n":K,"type":T,"id":E},...]}`: the exact mention
string, which occurrence of that exact string it is (1 = first, counting every
occurrence including overlapping ones), the type, and the entity id. Offsets are
recovered deterministically by `synthgen.anchors.nth_occurrence`; the emitted
target round-trips to the gold offsets exactly on every mention of the pilot
(tested). The v1 contract (character offsets) is kept for reproducibility of the
first pilot run, which showed a token-level model generating offsets with the
right shape and no grounding (span F1 0.03 with the correct type mix and id
structure). `synthgen generate --instruction-version v1|v2` selects the view;
the corpus itself is identical either way.

Inference is the production config: `do_sample=False`, `num_beams=1` (this is
what temperature 0 means, exactly), fixed `max_new_tokens`, and the tokenizer
copy saved beside the adapter. Batches are formed by a fixed sort so the same
inputs always form the same batches. A run directory carries a `run_lock.json`
(contract, data hashes, model, revision, LoRA shape, max_length, seed) and a
checkpoint can only be resumed by a run that matches it.

Evaluation reports span F1 (exact start, end, type) and entity-consistency
accuracy per slice: single, multi, listing, bundle, hard_case, all, plus per
type and per hard-case kind. Under v2 the predictions are relocated to offsets
first; a mention whose string or occurrence index is not in the document is
"unlocatable" and scores as a false positive of its type (its gold counterpart
stays a miss), and the report shows the unlocatable rate, which is the direct
measure of occurrence miscounting. A gold entity is consistent only when every one
of its mention spans was predicted, all under one predicted id, and that id is
used for no other gold entity. Output parsing is strict: anything that is not
`{"mentions":[{"span":[s,e],"type":T,"id":E},…]}` with integer offsets inside
the text is malformed, and a malformed sample counts every gold mention as
missed. There is no repair path.

The determinism test runs inference twice on the same checkpoint and inputs in
the same process and environment and requires byte-identical raw output. That
is the whole claim: same checkpoint, same input, same pinned environment. It is
not a cross-hardware claim, and `determinism.json` records the GPU and library
versions so it cannot be read as one.

CPU proof: `bash scripts/smoke_cpu.sh` builds a 1M-parameter random Qwen2
stand-in and runs the entire pipeline in about a minute. Its numbers are
meaningless (a random model emits no JSON); its purpose is that every code path,
including resume and the malformed-output branch, runs.

Known pilot property: the hash split put no bundle pair in `val.jsonl` for the
cafebabe seed, so the `bundle` slice is empty in the pilot eval. At 3k to 5k
samples it will be populated.

---

# Block 3: de-identification runtime (surrogates, receipts, utility check)

`deid/` turns `(text, mentions)` into a de-identified document, signs what it
did, and proves the document is still analysable. Gold labels and relocated
model output are interchangeable inputs; the whole block runs on CPU and is
tested against the 500 pilot gold labels without a model.

```bash
pip install -r requirements.txt                      # Faker, PyYAML, pytest, cryptography (pinned)
python -m deid.cli keygen --out keys/                # Ed25519 signing key pair + 32-byte surrogate secret (never committed)
python -m deid.cli run --corpus data/pilot/corpus.jsonl --keys keys/ --out runs/deid-pilot/
python -m deid.cli run --corpus ... --predictions runs/<run>/predictions.jsonl --keys keys/ --out ...   # mentions from the model
python -m deid.cli verify --run runs/deid-pilot/ --public-key keys/signing_key.pub                    # offline
```

**Policy** (`policies/ema-0070-v0.yaml`) is data the code executes; a change in
how any type is handled is a new policy id. It states the closed type set (must
equal `synthgen.types.ENTITY_TYPES`, the same tuple the generator, the validator
and both strict parsers import), the surrogate consistency scope (`document_bundle`:
the same entity maps to the same surrogate inside a bundle and to an unrelated one
elsewhere), per-type methods, the date shift (whole weeks, backward, 4 to 52), and
`age_handling: safe_harbor_90` (pass through under 90, `90+` at and above; jitter
excluded). Receipts carry the policy id and the sha256 of the file bytes.

**Surrogate engine** (`deid/surrogate.py`). Keys form one HMAC chain:
`scope_key = HMAC(master_secret, bundle_id or sample_id)`,
`entity_seed = HMAC(scope_key, entity_id)`, `shift_seed = HMAC(scope_key, "date-shift")`.
Each entity's Faker is seeded from its entity seed, so no mapping table exists
anywhere. Rendering is form-preserving: `Julian Gray`, `Julian`, `Mr. Gray`,
`J. Gray`, `J.G.`, `Jules Gray` and `Mr. Grey` all re-render from one surrogate
pair (with fresh initials); IDs keep their scheme and digit widths and carry the
surrogate site number; a site named after its city follows the city's surrogate;
places come from the real-place pool of the same country; generic references
(`the patient`, `the second subject`) pass through. Dates shift by one offset per
scope in whole weeks, so every interval and every weekday survives; `Day 14` is
unchanged; `nine days after the March visit` has its month re-rendered to the
shifted anchor's month, and when the anchor cannot be resolved from the text the
month word is dropped (`nine days after the visit`) and reported, never left
stale. A residual scan over the output fails the run if any original identifying
string survives.

**Receipts** (`deid/receipt.py`). One Ed25519 signature per shard (a shard is one
consistency scope) over the canonical JSON of: policy id + hash, code versions and
git commit, scope id and document ids, sha256 of the input and of the output
`(text, mentions)`, mention counts by type, tagger provenance (gold, or adapter
and predictions hashes under a contract), the surrogate secret's id and the
signing key's id, and a timestamp. A run receipt signs the ordered list of shard
receipt hashes plus the utility and residual verdicts. `verify` needs only the
public key, the receipts, the output and the policy file, and re-derives every
hash; it fails on an edited surrogate, a dropped document, a removed or reordered
shard, a re-signed shard, a different policy file, or a recorded utility failure.
Receipts contain counts and hashes, never document strings. Signatures come from
the `cryptography` package; nothing cryptographic is implemented in this repo.

**Utility check** (`deid/utility.py`). From the surrogated text alone it parses
every absolute date, checks that all of them in a scope moved by one whole-week
offset in the policy window, reconstructs every DATE entity's value, rebuilds the
per-patient timeline (screening, first dose, onset, action, resolution,
follow-up) and diffs every interval against the corpus date graph; every
relative expression must still resolve to the right day against the reconstructed
anchor; ages must be unchanged under the threshold; mention structure and all text
outside mentions must be identical. Zero tolerance: there is no tolerance parameter.

Pilot result (gold mentions, 500 documents, 480 shards): 8,855 intervals checked,
0 mismatched; 6,560 absolute dates shifted; 459 relative expressions still true
(446 study days, 2 weekdays, 11 anchored prose: 10 months re-rendered, 1 dropped);
2,060 ages unchanged; 0 residual hits; all receipts verify. About 4 seconds.

`runs/`, `keys/` are git-ignored. `engine_report.json` in a run folder lists the
reported phrases and so contains original strings; it is controller-side only.
