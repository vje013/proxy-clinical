# Proxy Clinical: scale run results (Qwen2.5-3B-Instruct, LoRA, contract v3, 5k corpus)

**What this document is.** A writeup of a run whose artifacts exist and are committed: `docs/runs/scale5k-v3/` in the attached repo holds the run manifest, lock, config, pip freeze, predictions with their manifest, eval, determinism result and adapter hashes exactly as the Colab runtime wrote them; the 120 MB adapter weights live in the zip you downloaded and are pinned by hash. Every number below is in those files. The two Block 3 runs on this model's output and on the gold labels are committed beside it, and `python -m deid.cli verify` passes on both. Both Presidio scans over the model-output run are committed as run, the first a FAIL and the second a PASS, with signed attestations that reference the run receipt by hash.

## 1. What was run

Base Qwen2.5-3B-Instruct at HF revision `aa8e7253…` (pinned and resolved identical), LoRA r 16 / alpha 32 / dropout 0.05 on q, k, v, o, gate, up, down; 29,933,568 trainable of 3,115,872,256 parameters. Corpus `data/scale5k`: 5,000 synthetic documents (2,600 single-patient narratives, 1,200 multi-patient, 800 listings, 400 bundle documents in 200 pairs), hard-case share 15% weighted toward initials_far and same_surname, master seed `ec1c2398…`, contract v3 (text anchors with word-bounded occurrence counting). Split 4,543 train / 457 val by hash of bundle id or sample id; val holds 16 bundles and 74 hard-case documents across all seven kinds. Two epochs, 568 steps at effective batch 16, learning rate 2e-4 cosine with 20 warmup steps, max sequence 6,656 tokens (longest sample 6,219), bf16, gradient checkpointing, seed 1234. A100-SXM4-80GB; torch 2.14.0+cu130, transformers 5.17.0, peft 0.21.0, trl 1.13.0.

Training took 87 minutes. Eval loss fell from 0.00165 at step 100 to 0.00027 at step 568, flat from step 400. Inference on 457 held-out documents took 34 minutes at batch 16 with greedy decoding and a 4,864-token cap; every output finished.

## 2. Results

| slice | documents | span F1 | entity accuracy | consistent / entities |
|---|---|---|---|---|
| all | 457 | 0.999 | 0.995 | 7,726 / 7,763 |
| single | 240 | 0.999 | 0.994 | 2,225 / 2,238 |
| multi | 120 | 1.000 | 0.998 | 1,761 / 1,765 |
| listing | 65 | 0.997 | 0.994 | 3,031 / 3,048 |
| bundle | 32 | 0.999 | 0.996 | 709 / 712 |
| hard_case | 74 | 0.997 | 0.991 | 1,563 / 1,578 |

Per type F1: PATIENT 0.999, DATE 1.000, ID 1.000, INVESTIGATOR 1.000, SITE 1.000, LOCATION 1.000, CONTACT 1.000, AGE 0.991. Of 19,597 predicted mentions, 5 were unlocatable and 0 outputs were malformed.

Hard-case kinds, entity accuracy, with the pilot in parentheses: same_surname 0.997 (0.800), initials_far 0.979 (0.895), prose_relative_date 0.966 (0.845), nickname_split 1.000 (1.000), misspelled_name 0.935 (0.966; 43 of 46 entities, 3 incomplete), namelike_distractor 0.998, id_only_reference 0.994.

Determinism: two inference passes on 32 documents in the same session, digest `322bd31d…` both, PASS.

Against the pilot (span F1 0.983, entity accuracy 0.938, AGE 0.913): the corpus grew ten-fold and the contract changed at the same time, so this run does not separate the two. The diagnostic in the pilot's error analysis, which re-scored the pilot model's own outputs under v3 counting, attributes AGE 0.913 to 0.978 to the contract alone; the rest is data.

## 3. The pipeline on this model's output

The full chain on the 457 predictions: strict parse, word-bounded relocation, cross-document entity linking within each bundle (the model's ids are document-local; 20 exact-string links merged 629 entities into 609 across 14 bundles), per-document gates, surrogate engine, post-surrogate scans, Ed25519 receipts, and the utility check against the gold date graph.

432 documents signed in 418 shards. On every signed document the utility check passes with zero tolerance: 6,635 date-graph intervals reconstructed exactly from the surrogated text, 0 relative-date failures, 17,704 gold mentions aligned to a tagger mention, and 0 gold mentions missed whose absence would change the output. Every miss that would have left an identifying string or a true date in the output was caught by a gate and the document refused.

25 refused. Fifteen for overlapping mentions, twelve of them the same shape: a bare age in a listing whose occurrence index landed on the day or month inside a date (`25` in `25-Jun-2018`). Five for an occurrence index one too high. Two for a scan hit: a repeated date the model listed once, and initials `W.G.` it missed. One for a phone number given the site's id. Two bundle partners of refused documents. The refusal rate is 5.5% overall and 2.6% of that is the listing-age class, which a tokenization rule treating hyphen- or slash-joined digit runs as one token would remove; that is a v4 contract change and a retrain, deferred.

On gold labels the same pipeline signs 4,999 of the 5,000 documents with 86,275 intervals exact; the one refusal is a site number that also appears inside an unlabelled study identifier, which the scan cannot distinguish from a leak and does not try to.

### Independent residual scan

A second opinion from a tool that shares no code, policy or vocabulary with the engine: Microsoft Presidio (presidio-analyzer 2.2.364, spaCy `en_core_web_lg` 3.8.0, threshold 0.4) run on a Colab CPU runtime over the 432 signed output documents and, for misses, over the 432 originals. The check is not "Presidio finds nothing": every surrogate is a finding, and 11,337 of its 12,752 output-side findings lie inside the pipeline's labelled spans as expected. The check is whether a PII-type finding survives outside every labelled span that no stated pattern explains; such a finding fails its document, and the scanner labels rather than drops everything else.

The first scan, committed as run in `independent-scan-1/`, failed 92 of 432 documents on 201 unadjudicated strings. Every one was inspected: 134 were the ICH E3 listing number at the head of every listing (`16.2.7.1`, read as an IP address), 39 were study identifiers (`Study KLB-1519-361` and its numeric tail), 24 were capitalised words from the generator's own templates (`Rechallenge`, `Enrolment`, `Concomitant`) and 4 were lab-unit fragments after a value (`98 U/L.`). None was a person, a place or a date. Those four shapes were added to the scanner as stated categories and the scan re-run with the same Presidio build: `independent-scan-2/` passes 432 of 432 with 0 unadjudicated, on raw findings identical to the first scan finding for finding (the same 2,834 outside-span findings by document, side, type and offsets; only their labels changed). On the input side, 11,389 of 12,808 findings are covered by tagger spans and the rest are the same structural strings: Presidio found no mention the model missed. The pilot rerun's 27 signed documents went through the same two scans (FAIL 2 of 27 on 4 strings, then PASS 27 of 27).

Each attestation is signed with a scanner key separate from the pipeline's, verifies against the public key in its folder, and names the run receipt (`f2041932…`), the output file and the corpus by hash. The `template_vocabulary` category depends on the generator's word lists and applies to synthetic text only; on real text those words would stay unadjudicated and fail their document, which is the behaviour wanted there.

## 4. Attestable training artifact, demonstrated on our own loss event

The night before the pilot rerun, a Colab runtime was reclaimed before its artifacts were downloaded; the run's report survives only as a transcript and is labelled as such in the repo. Every run since has been checked the same way before its numbers were used: the predictions file hashes to the value in the run's own manifest (`942907f2…` here); the adapter weights hash to the value the evaluation report printed (`6fcce716…`); the manifest's resolved base-model commit equals the pinned one (`aa8e7253…`); the train, val and corpus hashes in the manifest equal the committed data (`3f30bb13…`, `2e3ac170…`, `d820ec36…`); the determinism digest is the same in both passes; the pip freeze matches the pinned requirements; and re-running the evaluator locally on the predictions reproduces the runtime's report line for line. A number that fails any link in that chain is not reported. That chain, not the F1, is what makes the adapter an artifact rather than a claim.

## 5. Where this leaves the build plan

Corpus at 5k with the weak slices rebalanced: done, gates green, split confirmed. Retrain and commit the record: done. Per-slice eval: above. End-to-end on the new adapter's predictions: done, committed, verifies. Independent Presidio scan of the pipeline output: done, both scans committed, the second passes with the first kept as the record of what it took. This adapter is the demo model.

Open items, in the order I would take them: the v4 tokenization rule for bare numbers next to dates if the listing refusal rate matters; a Block 1 hardening so a site number never coincides with a study id; a name-matching cross-document linker to replace exact-string linking on free text; and a broader relative-date detector than the corpus grammar for the prose residual scan.
