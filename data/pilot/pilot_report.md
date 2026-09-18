# Proxy Clinical synthetic pilot report

Generator `synthgen 0.1.0`, master seed `cafebabecafebabecafebabecafebabecafebabecafebabecafebabecafebabe`, instruction version `v1`.

## Gates

| Gate | Result | Failures |
|---|---|---|
| 1 offset integrity | PASS | 0 |
| 2 entity consistency | PASS | 0 |
| 3 coverage | PASS | 0 |
| 4 distractor purity | PASS | 0 |
| 7 cross-reference pattern | PASS | 0 |
| 6 determinism | PASS | |

## Composition

- Samples: 500 (listing: 100, narrative: 400)
- Slices: bundle_listing: 20, bundle_narrative: 20, listing: 80, multi: 120, single: 260
- Bundles: 20 (narrative+listing pairs sharing entity ids)
- Multi-patient narratives: 125 of 400 (31.2%)
- Hard-case flagged samples: 50
- Locales: en_CA: 103, en_US: 397
- Total words: 146,532; narrative words: min 206, median 287, max 378
- Duplicate texts: 0
- Cast resamples (Faker name collided with vocab, template words or another cast member): 201 across 500 casts

## Hard-case kinds

| kind | count |
|---|---|
| id_only_reference | 15 |
| initials_far | 9 |
| misspelled_name | 6 |
| namelike_distractor | 13 |
| nickname_split | 8 |
| prose_relative_date | 13 |
| same_surname | 13 |

## Mentions per type

| type | count |
|---|---|
| AGE | 2060 |
| DATE | 7019 |
| ID | 2774 |
| INVESTIGATOR | 1421 |
| LOCATION | 442 |
| PATIENT | 6200 |
| SITE | 1696 |

## Distinct surface forms per patient (narratives)

| distinct forms | count |
|---|---|
| 3 | 2 |
| 4 | 6 |
| 5 | 35 |
| 6 | 39 |
| 7 | 73 |
| 8 | 339 |
| 9 | 17 |
| 10 | 31 |
| 11 | 2 |
| 12 | 1 |

## Name-like mentions per patient (narratives, capped at 12)

| mentions | count |
|---|---|
| 5 | 20 |
| 6 | 1 |
| 7 | 9 |
| 8 | 21 |
| 9 | 38 |
| 10 | 82 |
| 11 | 104 |
| 12 | 270 |

## Narrative length histogram (words)

| words | count |
|---|---|
| 200-249 | 52 |
| 250-299 | 219 |
| 300-349 | 120 |
| 350-399 | 9 |

## Templates

| template | count |
|---|---|
| listing-fixed-01 | 17 |
| listing-fixed-02 | 17 |
| listing-pipe-01 | 15 |
| listing-pipe-02 | 33 |
| listing-pipe-03 | 18 |
| narr-skel-01 | 35 |
| narr-skel-02 | 35 |
| narr-skel-03 | 35 |
| narr-skel-04 | 35 |
| narr-skel-05 | 34 |
| narr-skel-06 | 34 |
| narr-skel-07 | 34 |
| narr-skel-08 | 33 |
| narr-skel-09 | 27 |
| narr-skel-10 | 26 |
| narr-skel-11 | 26 |
| narr-skel-12 | 26 |
| narr-skel-13 | 20 |

