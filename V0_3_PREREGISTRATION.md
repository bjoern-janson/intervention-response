# ENGINEERED-WORLD V0.3 — CHECKPOINT / FORMAT LOCALIZATION

## Scientific role

V0 is CLOSED with `MATCH FAILURE`; the intervention-response target remains UNTESTED.

V0.2 is CLOSED as a forensic localization result. It established, under the frozen v0 construction, universal mastery of the 32 common singleton items, essentially universal failure on the 32 composition items, abundant exact cross-history prediction-signature overlap, and zero correctness-qualified terminal pairs.

V0.3 is a NEW PROSPECTIVE ASSAY. It does not repair or reinterpret v0 or v0.2. It asks:

> Was compositional behavior ever present after the history manipulation, and if so, what happened to its expression across the unchanged common phase?

The orthogonal format question is:

> Can unchanged singleton semantics survive a pure serialization-position shift under the tested layouts?

No intervention, post-terminal update, transfer assay, crossover, mechanism search, hidden-state analysis, or latent-geometry claim is authorized.

## Provenance

Repository: `bjoern-janson/intervention-response`

Frozen v0.2 parent:

- branch: `forensic-v0.2`
- execution commit: `dfac75f8b0a96d5b100fed446b91cf85b17ee943`
- GitHub Actions run: `34535640475`
- `v0_2_results.json` SHA-256: `a23b16134fa82179bed45140257d30c240d19661b62553afe66b4573eaa7dd03`
- v0.2 custody ZIP SHA-256: `ed299d00ca1dfa7b748c596bfb8bd2f21740865d8caf22bee271caf0db40cae4`

Frozen runtime source inputs remain byte-identical to v0/v0.2:

- `world.py`: `17f7fa7172a1e0927ab9b93df1abcc7fdb3fc23b1838cf480fc1afa5509308cb`
- `data.py`: `2e10e6671f97da8b66ad8bfb16c3a52bb78c9f20d87d396566f655d25f6c8b8b`
- `model.py`: `2e3447b192c0c88e15012a8dfbfdc44eff5ba789302e090ce2fedcaefe67d674`
- `experiment.py`: `0519121e3b1ae218d7105b14cb51355f60aa7babdbf0989488644a09a0d9937f`

The exact per-seed v0.2 terminal signatures are fossilized in `V0_2_TERMINAL_SIGNATURES.json`, derived from the verified v0.2 custody result before v0.3 execution.

Replay-manifest SHA-256:

`e727adb1a2f525b6c5c871ba5ff94d004cdf19d27e8de6c18ec6e5df6a6f4e9e`

## Population and training

V0.3 uses the exact frozen population and training schedule:

- 80 History-A models;
- 80 History-B models;
- `BASE_SEED = 1729`;
- A seeds `1729 + 2i`;
- B seeds `1730 + 2i`;
- `i = 0,...,79`;
- same architecture, CPU execution, task generators, history steps, common steps, learning rates, batch size, optimizer, and data-loader seeds as v0/v0.2.

No training dose, history strength, task membership, model parameter, tolerance, or world rule is changed.

## Checkpoints

Two observational checkpoints are frozen:

- `C0`: immediately after the 80-step history phase and before common terminal training;
- `C1`: immediately after the unchanged 120-step common terminal phase.

The live model is not evaluated at `C0`. Instead, it is deep-copied after history training, with Python/NumPy/Torch RNG state machine-checked as unchanged by the copy. The original model then immediately continues through the unchanged common phase. `C0` probe evaluation occurs only on the detached snapshot after the global replay gate passes.

This construction prevents checkpoint measurement from changing the terminal training trajectory.

## Mandatory v0.2 replay gate

Before any v0.3 probe result is scientifically interpretable, every regenerated `C1` model must exactly reproduce its v0.2 64-position argmax signature on `MANIFEST.equivalence`:

`signature_v0.3_C1(family, seed) == signature_v0.2(family, seed)`

for all 160 scientific models, with identical probe ordering and class-id encoding.

Any mismatch yields:

`REPRODUCTION FAILURE — STOP`

No `C0` or `C1` probe interpretation is authorized when this gate fails.

A 2×2 smoke may verify the first two frozen seeds per history but is never a scientific result and never authorizes probe interpretation.

## Frozen probe family

### 1. `S_canonical` — canonical singleton expression

Exactly `MANIFEST.terminal_train`: complete singleton truth tables for `T1`, `T2`, `T4`, `T5`.

Count: 32.

Canonical encoding:

`[CLS, op, state, SEP, PAD]`

### 2. `S_shifted` — pure position/serialization control

Exactly the same 32 semantic singleton items and labels as `S_canonical`, encoded as:

`[CLS, op, PAD, state, SEP]`

No semantic operator is added. The existing frozen model masks token id `PAD_ID = 0` at any position via its attention padding mask. Thus the supplied transformation and target are unchanged while the state and SEP occupy the positions used by two-operator sequences.

The contrast

`Acc(S_canonical) - Acc(S_shifted)`

measures position/serialization-sensitive expression under these two tested layouts. It does not establish a general serialization mechanism.

### 3. `P_practiced` — History-A practiced composition

All eight states for these four ordered pairs:

- `T1,T2`
- `T2,T1`
- `T1,T1`
- `T2,T2`

Count: 32.

Every item is directly present in History A. No pair item is present in History B.

This surface asks whether the intended History-A manipulation ever produced expressed performance on the composition tasks it directly trained.

### 4. `P_alias` — known-result compositional generalization

All eight states for these six ordered pairs:

- `T1,T4 = T5`
- `T4,T1 = T5`
- `T1,T5 = T4`
- `T5,T1 = T4`
- `T4,T5 = T1`
- `T5,T4 = T1`

Count: 48.

The equalities are exhaustively machine-checked against the frozen finite world. None of these pair serializations appears in History A or History B. Their resulting transformations are already represented among the common singleton tasks.

`P_alias` is NOT a pure format control. It measures two-operator compositional generalization when the resulting transformation is semantically familiar from singleton training.

## Measurements

For every model at `C0` and `C1`, after replay authorization, record:

- accuracy on each of the four frozen probe surfaces;
- exact argmax signature for each surface;
- SHA-256 of each probe signature;
- `format_gap = Acc(S_canonical) - Acc(S_shifted)`.

Across each history/checkpoint/surface report:

- mean accuracy;
- minimum accuracy;
- maximum accuracy;
- perfect-model count;
- zero-accuracy count.

Also report, descriptively:

- mean History-A minus History-B accuracy for every surface at `C0` and `C1`;
- within-family `C1 - C0` accuracy changes for every surface.

No individual seed is promoted to evidence of a history-level effect.

## Interpretation rules

V0.3 is a localization assay, not a binary target-support assay. Except for the mandatory replay gate, exact measured quantities are reported without post-hoc threshold creation.

Authorized descriptive language includes:

- If `S_canonical` exceeds `S_shifted`, state that expression is position/serialization-sensitive under the tested layouts, with the measured magnitude.
- If History-A `P_practiced` performance is already low at `C0`, state that the intended composition-training phase did not yield strong expressed performance on its practiced pair surface at that checkpoint. Do not infer why.
- If History-A `P_practiced` decreases from `C0` to `C1` while the shifted singleton control remains comparatively intact, state that expressed practiced-composition performance decreased across the common phase conditional on the measured format control. Do not call this internal forgetting.
- If `P_practiced` remains stronger than `P_alias`, state that observed composition performance is narrower/more training-family-specific under the tested surfaces. Do not infer an internal operator representation.
- If A and B differ on a composition surface while the corresponding singleton/format controls are comparable, state that a history-conditioned composition difference was observed under the tested probe family. This remains bounded to the engineered-world construction and seed populations.

## Claim ceiling

The strongest generic statement v0.3 may earn is:

> Under the frozen engineered-world assay, compositional performance was measured at the post-history and post-common checkpoints, with unchanged singleton semantics separately tested under a shifted serialization layout.

Depending on the measurements, v0.3 may report whether practiced compositional performance was present or absent at `C0`, whether its expressed performance changed through the common phase, and whether those observations coexist with intact or degraded shifted-singleton expression.

V0.3 does NOT establish:

- hidden representations or latent programs;
- internal forgetting or memory erasure;
- a general serialization failure;
- a general theory of compositionality;
- history-dependent future intervention response;
- an adjacent-possible or learnability geometry;
- corrigibility or self-sealing improvement;
- any Levin/Platonic ontology;
- a mechanism causing any measured difference.

## Status boundary

`V0 = CLOSED / MATCH FAILURE / intervention target UNTESTED`

`V0.2 = CLOSED / FORENSIC TERMINAL-EQUIVALENCE CHARACTERIZATION`

`V0.3 = NEW PROSPECTIVE CHECKPOINT / FORMAT LOCALIZATION ASSAY`

Scientific v0.3 execution is prohibited until the frozen branch artifact passes static tests and smoke validation and a separate execution authorization is supplied.
