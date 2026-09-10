# ENGINEERED-WORLD V0.4 — CAUSAL PARAMETER-UPDATE LOCALIZATION

## Status

**PROSPECTIVE / FROZEN WHEN COMMITTED / UNEXECUTED.**

No v0.4 scientific result exists at freeze time. No reconstruction, tensor checkpoint custody, parameter patch, interaction search, relearning episode, or intervention-response assay is authorized by this preregistration alone. Scientific execution requires a later, separate `V0_4_EXECUTE` commit that changes no frozen assay file.

## Lineage

- **V0:** CLOSED — `MATCH FAILURE`; intervention-response target untested.
- **V0.2:** CLOSED — forensic localization; prediction-signature convergence observed while correctness convergence failed.
- **V0.3:** EXECUTED — temporal/format localization complete; exact v0.2 replay 160/160; History-A practiced-pair performance strong at C0 and sharply reduced at C1 while singleton expression improved.
- **V0.4:** this prospective assay.

Permanent provenance statement:

> **v0.3 never possessed checkpoint custody; v0.4 creates endpoint tensor custody prospectively by deterministic reconstruction after exact behavioral replay.**

Therefore:

`behavioral replay fidelity != tensor identity fidelity`.

## Scientific question

Which preregistered common-phase parameter-group changes are intervention-sensitive loci for the observed History-A C0→C1 practiced-composition behavioral transition?

The assay does **not** ask where composition is stored, what representation the model contains, whether an internal memory was forgotten, or whether any group is logically necessary or sufficient.

## Frozen source and runtime lineage

The reconstruction must use the exact v0/v0.3 runtime sources inherited from the v0.3 execution lineage:

- `world.py` SHA256 `17f7fa7172a1e0927ab9b93df1abcc7fdb3fc23b1838cf480fc1afa5509308cb`
- `data.py` SHA256 `2e10e6671f97da8b66ad8bfb16c3a52bb78c9f20d87d396566f655d25f6c8b8b`
- `model.py` SHA256 `2e3447b192c0c88e15012a8dfbfdc44eff5ba789302e090ce2fedcaefe67d674`
- `experiment.py` SHA256 `0519121e3b1ae218d7105b14cb51355f60aa7babdbf0989488644a09a0d9937f`
- `v0_3_checkpoint_localization.py` SHA256 `012e3285b42131df246fa12e1c553a0805e6752a658f4204e4ec43cee0235afd`
- v0.3 scientific execution commit `214142fea6557c6637926bacff17d90d99484077`
- v0.3 frozen parent `6aca5d204f865809a2566dadf13e1c4418e953a9`
- v0.3 Actions run `34538200968`
- v0.3 artifact SHA256 `b8967dc837f7c59f01b2d624611eede8e39929b6937c2053c870a0444868475a`
- v0.3 `v0_3_results.json` SHA256 `eee8fb85b3fa16b99ffeb3d2471ac6f42e3376599c78e9aa13b845226a43f070`

Runtime remains Python `3.13.5`, NumPy `2.3.5`, PyTorch base version `2.10.0`, CPU device. The exact PyTorch build string used at execution is recorded in the new checkpoint custody manifest.

## V0.3 behavioral-custody bridge

`V0_3_BEHAVIORAL_SIGNATURES.json.zlib.b64` is derived from the already executed v0.3 custody artifact. Its frozen SHA256 is:

`15c3ae236dfef19d8e46bd8c9a401db3d5eb83c61026c57ace2fdf2c77ac54a8`

It contains all 160 v0.3 seed records, both checkpoints `C0/C1`, and the four frozen probe surfaces:

- `S_canonical`
- `S_shifted`
- `P_practiced`
- `P_alias`

The v0.4 entry gate therefore contains exactly:

`160 seeds × 2 checkpoints × 4 probes = 1280 exact prediction-signature comparisons`.

## Gate 1 — deterministic endpoint reconstruction

For every v0.3 seed and both histories, reconstruct the original trajectory using the exact frozen seed schedule, model, history data, optimizer construction, history steps, common steps, learning rates, batch size, and CPU execution path.

The live post-history model must **not** be evaluated before common training. C0 is captured by `deepcopy`; the original live model continues through the common phase exactly as in v0.3. Only after C1 exists may the C0/C1 snapshots be evaluated for replay.

Required identity:

`signature_reconstructed(family, seed, checkpoint, probe) == signature_v0.3(...)`

for all 1280 comparisons.

Any missing key or any prediction mismatch yields:

`RECONSTRUCTION FAILURE → STOP`.

If this gate fails, **no scientific checkpoint custody and no parameter-patch result may be created.**

Passing this gate establishes trajectory replay fidelity only. It does not establish equality to historical tensors that were never saved in v0.3.

## Checkpoint custody creation

Only after Gate 1 passes may v0.4 create C0/C1 tensor custody for all 160 models: 320 endpoint state dicts.

Each checkpoint record must contain:

- family
- seed
- checkpoint (`C0` or `C1`)
- every state-dict tensor name
- tensor dtype
- tensor shape
- tensor scalar count
- total scalar count
- canonical tensor hash
- checkpoint identity hash
- checkpoint container SHA256
- runtime manifest
- v0.3 source provenance
- v0.4 execution commit and frozen-parent identity

### Canonical tensor identity

The canonical tensor hash is SHA256 over a deterministic sorted sequence of:

`(parameter_name, dtype, shape, contiguous CPU tensor bytes)`.

Names and metadata are length-delimited. Mapping order and PyTorch `.pt` container metadata do not define tensor identity.

The `.pt` container SHA256 is recorded separately for artifact custody and is not treated as the canonical tensor identity.

The checkpoint identity hash additionally binds checkpoint metadata and provenance to the canonical tensor hash.

## Frozen parameter intervention partition

The v0.4 intervention ontology is exactly the model's named trainable parameter structure. Q/K/V are **not** split because PyTorch stores them inside the named `in_proj_*` tensors.

The ten groups are exact, disjoint, and exhaustive:

| Group | Exact named parameters | Scalars |
|---|---|---:|
| `token_embedding` | `token.weight` | 1,024 |
| `position_embedding` | `pos` | 160 |
| `layer0_attention` | `layers.0.self_attn.in_proj_weight`, `layers.0.self_attn.in_proj_bias`, `layers.0.self_attn.out_proj.weight`, `layers.0.self_attn.out_proj.bias` | 4,224 |
| `layer0_ffn` | `layers.0.linear1.weight`, `layers.0.linear1.bias`, `layers.0.linear2.weight`, `layers.0.linear2.bias` | 4,192 |
| `layer0_norms` | `layers.0.norm1.weight`, `layers.0.norm1.bias`, `layers.0.norm2.weight`, `layers.0.norm2.bias` | 128 |
| `layer1_attention` | `layers.1.self_attn.in_proj_weight`, `layers.1.self_attn.in_proj_bias`, `layers.1.self_attn.out_proj.weight`, `layers.1.self_attn.out_proj.bias` | 4,224 |
| `layer1_ffn` | `layers.1.linear1.weight`, `layers.1.linear1.bias`, `layers.1.linear2.weight`, `layers.1.linear2.bias` | 4,192 |
| `layer1_norms` | `layers.1.norm1.weight`, `layers.1.norm1.bias`, `layers.1.norm2.weight`, `layers.1.norm2.bias` | 128 |
| `final_norm` | `norm.weight`, `norm.bias` | 64 |
| `output_head` | `head.weight`, `head.bias` | 264 |

Required structural identities:

- every named trainable parameter appears in exactly one group;
- no group overlaps another group;
- no unknown parameter appears;
- no trainable parameter is omitted;
- total trainable scalar count is exactly `18,600`;
- group scalar counts sum exactly to `18,600`.

Any violation yields `PATCH CONTRACT FAILURE → STOP`.

No effect is normalized by group scalar count.

## Intervention population

All 160 models are reconstructed and tensor-custodied because Gate 1 is a full replay of v0.3.

The v0.4 **single-group patch experiment is restricted prospectively to the 80 History-A seeds**. That is the population exhibiting the preregistered large C0→C1 practiced-composition transition. History-B checkpoints remain in tensor custody as provenance/reference objects but receive no single-group patch search in v0.4.

Any History-B patch experiment requires a new prospective assay.

## Gate 2 — patch-contract controls

Before any single-group patch result is computed, every selected History-A seed must pass all controls.

### Loaded-endpoint replay

C0/C1 checkpoints loaded from custody must reproduce the exact corresponding v0.3 signatures on all four frozen surfaces.

### C1→C1 no-op

Replacing every C1 tensor with the corresponding C1 tensor must produce:

- exact tensor equality to C1;
- exact canonical tensor hash equality to C1;
- exact C1 prediction-signature equality on all four probe surfaces.

### C1→C0 full restoration

Replacing every C1 tensor with the corresponding C0 tensor must produce:

- exact tensor equality to C0;
- exact canonical tensor hash equality to C0;
- exact C0 prediction-signature equality on all four probe surfaces.

Any control failure yields:

`PATCH CONTRACT FAILURE → STOP`.

No single-group patch result is computed before all 80 History-A seeds pass this global gate.

## Frozen single-group interventions

For each History-A seed and each frozen group `g`:

### Restoration patch

Start from C1 and replace exactly the tensors in `g` by their C0 values:

`theta_1^(g←0)`.

### Forward patch

Start from C0 and replace exactly the tensors in `g` by their C1 values:

`theta_0^(g←1)`.

No other tensor changes. Inputs C0/C1 are immutable. Every patch is built from cloned CPU state dictionaries and loaded into a fresh exact model instance.

There is:

- no optimizer;
- no gradient;
- no training;
- no new task data;
- no new probe;
- no interaction patch;
- no adaptive selection of groups.

## Frozen behavioral readouts

Every patch is evaluated on exactly the four v0.3 surfaces:

`{S_canonical, S_shifted, P_practiced, P_alias}`.

Raw accuracies and exact prediction signatures are retained for the patched model.

For each surface `q`, the raw restoration effect is:

`R_(g,s,q) = Acc_q(theta_1^(g←0)) - Acc_q(theta_1)`.

The raw forward change is:

`F_(g,s,q) = Acc_q(theta_0^(g←1)) - Acc_q(theta_0)`.

For practiced composition, define the observed per-seed v0.3 transition:

`D_s = Acc_Ppracticed(theta_0) - Acc_Ppracticed(theta_1)`.

Because v0.3 observed decline on all 80 A seeds, v0.4 requires `D_s > 0` for every patched seed after exact replay. Otherwise `PATCH CONTRACT FAILURE → STOP`.

The descriptive normalized coordinates are:

`eta_restore_(g,s) = [Acc_Ppracticed(theta_1^(g←0)) - Acc_Ppracticed(theta_1)] / D_s`

and

`eta_forward_(g,s) = [Acc_Ppracticed(theta_0) - Acc_Ppracticed(theta_0^(g←1))] / D_s`.

Interpretation of these coordinates is descriptive:

- `0`: no recovery / no induced collapse on the practiced probe for that seed;
- `1`: full observed endpoint practiced transition matched in magnitude;
- values outside `[0,1]`: nonlinear/co-adapted patch behavior, **not** stronger causal status.

No binary threshold for `eta`, restoration, selectivity, necessity, sufficiency, or significance is preregistered in v0.4.

## Reporting

For each group, report raw per-seed patch readouts and descriptive summaries of raw practiced effects and `eta` coordinates. All four frozen surfaces remain visible so practiced-pair changes can be distinguished from broad behavioral changes.

The primary v0.4 output is a continuous map of intervention-sensitive parameter groups under the frozen patch operation.

## Claim ceiling

v0.4 may earn statements of the form:

> Replacing the C1 values of preregistered parameter group `g` with their reconstructed C0 values changed the frozen behavioral readouts by the measured amount; replacing the C0 values with C1 values produced the measured forward-patch change.

It may identify **intervention-sensitive parameter loci under the tested transplant**.

v0.4 does **not** establish:

- where composition is stored;
- representational content;
- an internal operator concept;
- biological memory;
- internal forgetting;
- logical necessity or sufficiency;
- causal irrelevance from a null single-group effect;
- independence of parameter groups;
- mechanism beyond the tested parameter intervention;
- arbitrary-model generalization;
- intervention-response differences;
- corrigibility or safe self-improvement.

Permanent prohibitions:

`single-group null != group causally uninvolved`

`restoration effect != necessity`

`forward-patch effect != sufficiency`

`parameter-intervention locus != storage location`

`causal contribution under patch != representational content`.

## Interaction boundary

v0.4 contains **single-group patches only**. It does not automatically open pairwise or higher-order combinations after any result. Interaction tests require a new prospective preregistration after v0.4 is closed.

## Stop state at freeze

At repository freeze:

- v0.4 design exists;
- v0.3 behavioral custody manifest exists;
- implementation/tests/workflow exist;
- no `V0_4_EXECUTE` marker exists;
- no 160-model reconstruction has run under v0.4;
- no v0.4 checkpoint custody exists;
- no v0.4 scientific patch result exists.
