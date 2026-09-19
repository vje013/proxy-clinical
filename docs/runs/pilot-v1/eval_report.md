# Proxy Clinical tagger evaluation

Adapter `/content/runs/pilot-qwen2.5-3b-20260918/adapter` (sha256 `82ecf4a0ce918bca...`), predictions `/content/runs/pilot-qwen2.5-3b-20260918/predictions.jsonl`, corpus `data/pilot/corpus.jsonl`.
Decoding: greedy, max_new_tokens 4096, batch 8; GPU NVIDIA A100-SXM4-80GB; torch 2.14.0+cu130, transformers 5.17.0, peft 0.21.0, trl 1.13.0.

Span F1 is exact-match on (start, end, type). Entity consistency counts a gold entity as consistent only when every one of its mention spans was predicted, all under a single predicted id, and that id is not used for any other gold entity's mentions. A malformed output counts every gold mention of the sample as missed and every gold entity as incomplete.

## Per slice

| slice | samples | malformed | unfinished | span P | span R | span F1 | entity acc | consistent | split | merged | incomplete |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 52 | 3 (6%) | 2 | 0.028 | 0.027 | 0.027 | 0.000 | 0/769 | 0 | 0 | 769 |
| single | 34 | 1 (3%) | 0 | 0.045 | 0.049 | 0.047 | 0.000 | 0/323 | 0 | 0 | 323 |
| multi | 11 | 0 (0%) | 0 | 0.021 | 0.023 | 0.022 | 0.000 | 0/158 | 0 | 0 | 158 |
| listing | 7 | 2 (29%) | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0/288 | 0 | 0 | 288 |
| hard_case | 8 | 1 (12%) | 0 | 0.044 | 0.044 | 0.044 | 0.000 | 0/88 | 0 | 0 | 88 |

## Per type (all samples)

| type | tp | fp | fn | P | R | F1 |
|---|---|---|---|---|---|---|
| AGE | 16 | 138 | 167 | 0.104 | 0.087 | 0.095 |
| DATE | 2 | 640 | 654 | 0.003 | 0.003 | 0.003 |
| ID | 26 | 119 | 202 | 0.179 | 0.114 | 0.139 |
| INVESTIGATOR | 0 | 174 | 182 | 0.000 | 0.000 | 0.000 |
| LOCATION | 0 | 48 | 49 | 0.000 | 0.000 | 0.000 |
| PATIENT | 13 | 742 | 646 | 0.017 | 0.020 | 0.018 |
| SITE | 0 | 127 | 156 | 0.000 | 0.000 | 0.000 |

## Entity consistency by hard-case kind

| kind | entities | consistent | split | merged | incomplete | acc |
|---|---|---|---|---|---|---|
| initials_far | 19 | 0 | 0 | 0 | 19 | 0.000 |
| misspelled_name | 29 | 0 | 0 | 0 | 29 | 0.000 |
| nickname_split | 10 | 0 | 0 | 0 | 10 | 0.000 |
| prose_relative_date | 58 | 0 | 0 | 0 | 58 | 0.000 |
| same_surname | 30 | 0 | 0 | 0 | 30 | 0.000 |

## Malformed output reasons (top 5)

- invalid JSON: 2
- mention 33: 1

