# ENGINEERED-WORLD V0.2 — TERMINAL EQUIVALENCE CHARACTERIZATION

## Scientific role

V0 is CLOSED with `MATCH FAILURE`: 80 models per history were executed, 0 terminal cross-history pairs were admitted, and the intervention-response target was UNTESTED.

V0.2 is a forensic prerequisite-characterization assay. It localizes where the frozen v0 terminal equivalence gate failed. It does not alter or rescue v0 and it contains no intervention, post-terminal update, transfer, control, crossover, mechanism, or hidden-state analysis.

## Provenance

- Repository: `bjoern-janson/intervention-response`
- V0 execution parent state: `158135c60b0b2d10b4bea6d0f3b49de0689504ff`
- V0 GitHub Actions run: `34534073971`
- V0 `results.json` SHA-256: `3b635ec2cc326bb46feb1cb589788b163778333188645d86b6ca45f61ee24938`
- V0 result artifact ZIP SHA-256: `5c061bb9ba6dcc114fd293e9a05dd3e2f9ff30085b992b838f7a5bf7de0249db`

The following v0 runtime source files are immutable inputs to v0.2:

- `world.py`: `17f7fa7172a1e0927ab9b93df1abcc7fdb3fc23b1838cf480fc1afa5509308cb`
- `data.py`: `2e10e6671f97da8b66ad8bfb16c3a52bb78c9f20d87d396566f655d25f6c8b8b`
- `model.py`: `2e3447b192c0c88e15012a8dfbfdc44eff5ba789302e090ce2fedcaefe67d674`
- `experiment.py`: `0519121e3b1ae218d7105b14cb51355f60aa7babdbf0989488644a09a0d9937f`

## Population and training

V0.2 regenerates the exact v0 terminal population from the frozen code, runtime, seed schedule, architecture, histories, and common terminal phase:

- 80 History-A models
- 80 History-B models
- `BASE_SEED = 1729`
- A seed `1729 + 2i`, B seed `1730 + 2i`, for `i=0,...,79`
- same history tasks, optimization steps, optimizer, learning rates, common terminal training, model architecture, and CPU device as v0

No model receives any intervention after terminal training.

## Frozen terminal strata

The original equivalence surface is partitioned exactly into three disjoint strata:

### `T_common`

`MANIFEST.terminal_train`: complete singleton truth tables for `T1,T2,T4,T5`.

Count: 32 items.

### `T_A_seen`

The subset of `EQUIVALENCE_EXTRA` with operator sequences:

- `T1 ∘ T1`
- `T2 ∘ T2`

Count: 16 items.

These 16 items are directly present in History A and absent from History B.

### `T_novel`

The subset of `EQUIVALENCE_EXTRA` with operator sequences:

- `T4 ∘ T4`
- `T5 ∘ T5`

Count: 16 items.

These pairwise compositions are absent from both history-training families.

The validator must prove:

`T_common ∪ T_A_seen ∪ T_novel = MANIFEST.equivalence`

with pairwise-disjoint item sets and counts `32 + 16 + 16 = 64`.

## Measurements

For each of all 160 terminal models record, before any post-terminal update:

- family and seed;
- accuracy and error on `T_common`;
- accuracy and error on `T_A_seen`;
- accuracy and error on `T_novel`;
- accuracy and error on the full v0 equivalence surface;
- exact 64-position argmax signature on the full equivalence surface;
- SHA-256 of that signature;
- the set of strata on which at least one error occurs.

Across histories record:

- exact per-stratum perfect-model counts;
- full-surface perfect-model counts;
- mean/min/max accuracy by stratum and history;
- full 80×80 cross-history signature-distance matrix `d_ij`;
- nearest cross-history distance for every A model and every B model;
- exact signature frequency maps in A and B;
- set-intersection count of shared signatures;
- multiset overlap `sum_s min(count_A(s), count_B(s))`;
- error-pattern frequency maps over the three strata.

## Logical identity check

Because the surface is deterministic and labeled, any two models with zero full-surface error must have the same exact argmax signature. V0.2 must machine-check this implication for all regenerated models.

Therefore the maximum possible number of exact, correctness-qualified cross-history pairs equals:

`min(n_full_perfect_A, n_full_perfect_B)`.

This quantity is reported directly.

## V0 reproduction gate

Before localization is interpreted, v0.2 recomputes the original terminal-only deterministic matcher on the regenerated models.

Required result:

`recomputed_v0_terminal_pairs = 0`.

If this does not reproduce the frozen v0 terminal count, v0.2 returns:

`REPRODUCTION MISMATCH → STOP`

and no localization interpretation is authorized.

## Descriptive localization only

If the reproduction gate passes, v0.2 reports the measured stratum distributions and exact failure-pattern frequencies. It does not assign causal responsibility merely from exposure asymmetry.

The following phrases are descriptive labels only:

- `COMMON TERMINAL GAP`: models exhibit error on `T_common`.
- `A-SEEN STRATUM GAP`: models exhibit error on `T_A_seen`.
- `NOVEL COMPOSITION GAP`: models exhibit error on `T_novel`.
- `CONTRACT / IMPLEMENTATION ANOMALY`: sufficient full-perfect models exist in both histories yet the exact deterministic matcher cannot produce the logically implied pairs.

V0.2 must report all applicable gaps rather than force a single exclusive story.

## Original-gate sufficiency ledger

For each history, report whether at least the original minimum `20` models are perfect on each individual stratum and on the full equivalence surface. This is a diagnostic ledger, not a modified matching rule.

No tolerance, training dose, history strength, surface membership, optimizer setting, or intervention property may be changed in v0.2.

## Claim ceiling

Authorized result language is limited to terminal-state localization, for example:

> Under the frozen v0 construction, terminal-equivalence failure localized to the measured strata as reported by v0.2.

If `T_A_seen` differs across histories, v0.2 may state that the equivalence surface contained history-asymmetric training exposure and report the measured gap. It may not claim that this asymmetry caused the v0 failure without a separate causal test.

No history-conditioned transfer, intervention response, latent representation, geometry, adjacent-possible, corrigibility, or mechanism claim is authorized.
