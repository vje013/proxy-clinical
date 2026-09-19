# Proxy Clinical: v2 pilot results (Qwen2.5-3B-Instruct, LoRA)

**What this document is.** Numbers transcribed from the session log; run artifacts not retained. The Colab runtime that produced this run was reclaimed before the run folder was downloaded, so the adapter, `run_manifest.json`, `pip_freeze.txt`, `predictions.jsonl`, `eval.json`, and `determinism.json` no longer exist anywhere. Every figure below comes from the cell output pasted into the session. Nothing here is a manifest, a freeze file, or a checksum you can verify against an artifact. Where a fact is checkable against the retained repository and corpus, the text says so. Where it is only in the log, the text says that too.

Run id as printed: `pilot-qwen2.5-3b-v2-20260919`.

## 1. What was run

**Verifiable against the retained repo and data.** The tarball delivered for this run was built from commit `9556a7b` ("v2 output contract: text anchors with occurrence index, deterministic relocation"). The transcript excerpt does not include the git log line cell 1 prints, so the commit on the runtime is my record of what was shipped, not a runtime-side confirmation. The configuration is `configs/pilot.yaml` at that commit:

| item | value | source |
|---|---|---|
| base model | `Qwen/Qwen2.5-3B-Instruct`, HF revision `aa8e72537993ba99e69dfaafa59ed015b17504d1` | repo config |
| LoRA | r 16, alpha 32, dropout 0.05, targets q/k/v/o/gate/up/down proj | repo config |
| optimizer schedule | lr 2e-4, cosine, 10 warmup steps, 3 epochs, batch 2 x accum 8 (effective 16), bf16, gradient checkpointing, seed 1234 | repo config |
| max sequence length | 6144 tokens (over-budget sample is a hard error) | repo config |
| inference | greedy, `max_new_tokens` 4096, batch 8, bf16, deterministic algorithms on | repo config |
| output contract | `instruction_version: v2` (text anchor + occurrence index), asserted on every training line at load | repo config |
| corpus | `data/pilot/corpus.jsonl`, sha256 `fe0dad56f62d7bc5...` | retained file |
| train view | `data/pilot/train.jsonl`, 448 samples, sha256 `2b24a41670e462a3...` | retained file |
| val view | `data/pilot/val.jsonl`, 52 samples, sha256 `43c1f81b2a279cfb...` | retained file |

**Verifiable val composition** (computed from the retained corpus with the same split function): 34 single-narrative, 11 multi-patient, 7 listings, 0 bundle documents. 8 of the 52 are hard-case samples (kinds present: prose_relative_date 5, misspelled_name 3, initials_far 2, same_surname 2, nickname_split 1; a sample can carry more than one kind). Gold in val: 2,113 mentions across 769 entities. Per type: PATIENT 659, DATE 656, ID 228, AGE 183, INVESTIGATOR 182, SITE 156, LOCATION 49.

The bundle slice has no representative in val for this seed. The bundle result in this pilot is therefore untested, not passing.

**From the log only.** GPU `NVIDIA A100-SXM4-80GB`; torch `2.14.0+cu130`, transformers `5.17.0`, peft `0.21.0`, trl `1.13.0`. These match the pins in `requirements-colab.txt`, but the actual freeze file was not retained, so the rest of the environment is unrecorded. Trainable parameters 29,933,568 of 3,115,872,256 (0.961%).

Token lengths as logged: train min 993, median 1,381, p95 3,779, max 5,742; val min 1,002, median 1,318, p95 3,897, max 4,660. All under the 6,144 budget, which the loader enforces.

## 2. Training trajectory (transcribed)

84 optimizer steps (448 samples / 16 effective batch = 28 steps per epoch, 3 epochs). Wall time 758.1 s for training. Every eval below is loss on the 52 val samples under teacher forcing, which is not the same thing as the extraction metrics in section 4.

| step | epoch | train loss | eval loss | eval token acc |
|---|---|---|---|---|
| 5 | 0.18 | 0.154 | | |
| 10 | 0.36 | 0.1097 | | |
| 15 | 0.54 | 0.04897 | | |
| 20 | 0.71 | 0.02662 | 0.02985 | 0.9902 |
| 25 | 0.89 | 0.01958 | | |
| 30 | 1.07 | 0.01435 | | |
| 35 | 1.25 | 0.009026 | | |
| 40 | 1.43 | 0.00709 | 0.007903 | 0.9978 |
| 45 | 1.61 | 0.00639 | | |
| 50 | 1.79 | 0.006519 | | |
| 55 | 1.96 | 0.004672 | | |
| 60 | 2.14 | 0.003561 | 0.004489 | 0.9989 |
| 65 | 2.32 | 0.002917 | | |
| 70 | 2.50 | 0.003331 | | |
| 75 | 2.68 | 0.003097 | | |
| 80 | 2.86 | 0.002217 | 0.003967 | 0.9991 |
| 84 | 3.00 | | 0.003971 | 0.9991 |

Reported `train_loss` (mean over the run) 0.02525. Eval loss flattens between step 60 and 84 (0.0045 to 0.0040); the third epoch bought little. Train and eval loss track each other closely, so there is no sign of overfitting at this scale, but with 448 templated training documents that is a weak statement.

## 3. Inference (transcribed)

52 val documents, greedy decoding, batch 8, length-sorted batching. 474 s total. `finished_rate=1.0`: no output was cut off by the 4,096-token cap. Malformed JSON: 0 of 52.

## 4. Extraction metrics (transcribed)

Definitions, as printed in the report header: v2 predictions are text anchors (exact string plus occurrence index) relocated to character offsets deterministically before scoring. A predicted mention whose string or occurrence is not in the document is *unlocatable* and is scored as a false positive of its predicted type. Span F1 is exact match on (start, end, type). An entity is *consistent* only when every one of its gold mention spans was predicted, all under one predicted id, and that id is not shared with another gold entity. *Split*: one gold entity under two or more predicted ids. *Merged*: one predicted id covering two or more gold entities. *Incomplete*: at least one gold mention of the entity was missed.

The report header also printed an adapter checksum prefix, `dd96984f019ee87c...`. The adapter is gone, so that value cannot be checked against anything. It is recorded here as a transcript line, not as provenance.

### Per slice

| slice | samples | malformed | unfinished | unlocatable | span P | span R | span F1 | entity acc | consistent | split | merged | incomplete |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 52 | 0 | 0 | 4/2103 | 0.985 | 0.980 | 0.983 | 0.940 | 723/769 | 4 | 6 | 36 |
| single | 34 | 0 | 0 | 4/958 | 0.991 | 0.980 | 0.986 | 0.944 | 305/323 | 1 | 2 | 15 |
| multi | 11 | 0 | 0 | 0/428 | 0.991 | 0.988 | 0.990 | 0.943 | 149/158 | 2 | 2 | 5 |
| listing | 7 | 0 | 0 | 0/717 | 0.975 | 0.975 | 0.975 | 0.934 | 269/288 | 1 | 2 | 16 |
| hard_case | 8 | 0 | 0 | 2/246 | 0.984 | 0.972 | 0.978 | 0.886 | 78/88 | 2 | 2 | 6 |

`hard_case` overlaps the other rows (its 8 samples are also counted in single, multi, or listing). No bundle row exists because val holds no bundle documents.

### Per type, all 52 samples

| type | tp | fp | fn | P | R | F1 |
|---|---|---|---|---|---|---|
| AGE | 167 | 16 | 16 | 0.913 | 0.913 | 0.913 |
| DATE | 645 | 4 | 11 | 0.994 | 0.983 | 0.989 |
| DRUG | 0 | 1 | 0 | 0.000 | 0.000 | 0.000 |
| ID | 228 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| INVESTIGATOR | 182 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| LOCATION | 48 | 2 | 1 | 0.960 | 0.980 | 0.970 |
| PATIENT | 648 | 8 | 11 | 0.988 | 0.983 | 0.986 |
| SITE | 153 | 0 | 3 | 1.000 | 0.981 | 0.990 |

Gold per type (tp + fn) matches the retained val corpus exactly, which is the one cross-check the transcript allows against retained data.

One arithmetic note. The slice table counts 2,103 predicted mentions; the type table sums to 2,102 (2,071 tp + 31 fp). The scorer collapses exact duplicate predictions (same start, end, type) into one, so the difference means the model emitted one exact duplicate mention somewhere. That is derivable from the transcript plus the retained scorer code. Which sample it was is not.

### Entity consistency by hard-case kind

| kind | entities | consistent | split | merged | incomplete | acc |
|---|---|---|---|---|---|---|
| initials_far | 19 | 17 | 0 | 0 | 2 | 0.895 |
| misspelled_name | 29 | 28 | 0 | 0 | 1 | 0.966 |
| nickname_split | 10 | 10 | 0 | 0 | 0 | 1.000 |
| prose_relative_date | not captured | | | | | |
| same_surname | not captured | | | | | |

`run_pilot.sh` printed only the first 40 lines of the report at the time, and the table ended at line 40. Val contains 5 prose_relative_date and 2 same_surname samples, so those rows existed in `eval_report.md`; their values are lost with it. The "Malformed output reasons" section was cut the same way, but every slice row shows 0 malformed, so that section listed nothing. The script now prints the whole report.

## 5. Determinism (transcribed)

Two inference passes over 32 val samples, same adapter, same session, same environment. Digest `e824cc189f7b8e01` on both runs, 0 samples differ, PASS. The claim proven is exactly that: same checkpoint, same inputs, same pinned environment, same session, byte-identical output. It is not a cross-GPU or cross-driver claim, and `determinism.json` itself was not retained.

## 6. v1 to v2 (transcribed from the earlier session)

The v1 run, same base, same LoRA config, same corpus, same seed, differed in one variable: the output contract asked for character offsets `{"span":[start,end],"type","id"}`. Its span F1 was 0.027. The predictions showed the model producing offsets that looked plausible and drifted, which is a known failure mode for autoregressive decoders asked to count characters. v2 replaced offsets with `{"text","n","type","id"}`: the exact mention string and which occurrence of that string it is. Relocation to offsets happens deterministically in the scorer and in the production pipeline, never in the model. Same model, same data, one contract change: 0.027 to 0.983.

The v1 run's artifacts were also not retained. Its 0.027 is a transcript number under the same caveat as everything else here.

## 7. Reading the numbers

Span F1 0.983 means that of the mentions the model emitted, 98.5% landed on an exact gold span with the right type, and 98.0% of gold mentions were recovered. On this templated pilot corpus that is close to the ceiling the eval can measure with 2,113 gold mentions; the remaining 1.7% is 31 false positives and 42 misses.

Entity accuracy 0.940 is mostly the same errors seen at entity granularity. Of the 46 non-consistent entities, 36 are *incomplete*, meaning at least one of their mentions was missed or mis-anchored. Only 10 are identity errors proper: 4 split, 6 merged. Coreference is not where this model is losing.

AGE is the outlier at 0.913, and the shape of it is telling: 16 false positives and 16 false negatives, symmetric. That is the signature of a correct string with the wrong occurrence index. A bare age such as `67` recurs in a listing (as an age, inside an identifier, inside a date, inside a dose), so an off-by-one in `n` relocates the mention to a different `67`, producing one FP and one FN per miscount. This is a hypothesis. Confirming it needs `predictions.jsonl`, which does not exist for this run. If it holds, the fix is either a left-context anchor for the AGE type or a listing layout that keeps bare ages from colliding. The decision was deferred to the error analysis, and the error analysis is blocked on the rerun.

DRUG: the model emitted one mention with a type that is not in the label set. It was scored as a false positive and nothing else. One invented type in 2,103 predictions is a curiosity on a pilot and a hard error in production, where the type set is closed and the pipeline should reject the output rather than skip the mention.

Unlocatable 4/2,103 (0.19%): four predicted anchors pointed at a string or occurrence not present in the document. All four fell in single-narrative samples, two of them in hard-case samples. Their type breakdown is in the per-type FP column somewhere but is not separable from the transcript.

Listings are the weakest slice (0.975 / 0.934) and carry the most incomplete entities (16 of 36) with only 7 samples. They are also where AGE occurrence collisions would live. These two observations are probably one observation.

## 8. What this run cannot claim

No adapter. The weights cannot be reloaded, so nothing above can be reproduced from the artifact; it can only be reproduced by rerunning training, which under the pinned environment and seed should regenerate the same adapter but has not been shown to.

No `run_manifest.json`. The resolved HF commit of the downloaded base weights, the training log history, and the per-file data hashes recorded by the trainer are all lost. The data hashes are recoverable from the retained files; the resolved commit and log history are not.

No `pip_freeze.txt`. Four library versions are in the transcript. The rest of the environment is unrecorded.

No `predictions.jsonl`. No error analysis is possible. Every statement in section 7 about *why* a type failed is inference from aggregate counts, not from examples.

Two hard-case kinds have no entity-consistency figures, for the reason given in section 4.

The bundle slice was not evaluated in this pilot at all, because this seed's split placed no bundle pair in val.

## 9. Next actions

Rerun v2 on the same runtime class. The notebook now packs the run folder the moment the pipeline finishes and downloads `predictions.jsonl` and the zip from cell 4 itself, so the artifacts leave the runtime before it can be reclaimed. The zip carries the manifest, freeze, lock, predictions with their manifest, eval, and determinism files; that folder is the provenance record this document is not.

With `predictions.jsonl` in hand: confirm or refute the AGE occurrence-miscount hypothesis on the 16 pairs, inspect the 4 split and 6 merged entities, find the DRUG sample, and identify the duplicate prediction. Then decide between left-context anchors for AGE and a listing layout change.

After that: a 5k corpus with a harder held-out slice and a val split that contains bundle pairs, then the base-model bake-off (Phi-4-mini as the candidate) if the pilot numbers leave anything worth comparing.
