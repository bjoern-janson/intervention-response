# V0.4 Parameter-Update Localization Design

## Purpose

V0.4 converts the v0.3 behavioral transition into a counterfactual parameter-patching assay without retroactively pretending that v0.3 saved endpoint tensors. It first reconstructs C0/C1 from the frozen deterministic trajectory, verifies all 1280 custody-recorded signatures, creates content-addressed checkpoint custody, and only then performs frozen one-group patches on History A.

## Architecture

The implementation is split into three units so provenance and intervention logic remain separable.

1. `v0_4_tensor_contract.py` defines canonical state-dict identity, the exact 10-group/18,600-scalar partition, immutable patch construction, checkpoint identity records, hard replay gates, and normalized descriptive coordinates.
2. `v0_4_reconstruct.py` reconstructs all 160 v0.3 trajectories, performs the complete 1280-signature replay gate, and only after a global pass writes 320 C0/C1 checkpoint files plus canonical tensor custody.
3. `v0_4_patch_localization.py` consumes that newly created custody, validates every checkpoint and all 80 History-A no-op/full-restoration controls, and only after a global patch-contract pass evaluates the 10 frozen single-group restoration/forward patches.

The scientific workflow is trigger-gated on a later `V0_4_EXECUTE` file. The prospective freeze contains the workflow but not the trigger file.

## Provenance boundaries

The source of historical behavioral truth is the executed v0.3 artifact, not regenerated summary statistics. `V0_3_BEHAVIORAL_SIGNATURES.json.zlib.b64` contains the exact v0.3 C0/C1 prediction signatures for the four frozen surfaces and is content-hashed inside the prospective assay.

A successful behavioral replay does not prove equality to historical tensors because historical tensors were not saved. Tensor custody begins only after reconstruction succeeds and the new reconstructed state dicts receive canonical content hashes.

## Canonical tensor identity

State dicts are ordered by tensor name. For each tensor the hash stream binds its name, dtype, shape, byte length, and contiguous CPU bytes. This is independent of Python mapping order and PyTorch container metadata. `.pt` SHA256 values are recorded separately as custody of the transport container.

A checkpoint identity hash binds the canonical tensor hash to family, seed, checkpoint, runtime manifest, and source provenance.

## Parameter groups

Groups match exact named parameters in `TinyTransformer`. They are disjoint and exhaustive, with 18,600 total trainable scalars. The model's fused `self_attn.in_proj_*` tensors are not sliced into conceptual Q/K/V subgroups.

The groups are token embedding, position embedding, layer-0 attention, layer-0 FFN, layer-0 norms, layer-1 attention, layer-1 FFN, layer-1 norms, final norm, and output head.

## Execution data flow

A future authorized execution follows this irreversible gate order:

`frozen source identity → runtime/tests → smoke → reconstruct 160 trajectories → 1280 exact signature replay → create 320 checkpoint custody objects → verify partition → verify loaded endpoints → verify all 80 A no-op/full-restoration controls → single-group patches → four-surface readouts → descriptive summaries`.

No downstream scientific result is produced when an upstream gate fails.

## Patch semantics

A restoration patch starts from C1 and replaces exactly one frozen group's tensors with C0 values. A forward patch starts from C0 and replaces exactly one group's tensors with C1 values. Inputs remain immutable and no optimization/training occurs.

V0.4 patches only History A. All 160 models are reconstructed and custodied because reconstruction is a fidelity bridge to v0.3, but the causal target is the History-A C0→C1 practiced-pair collapse localized by v0.3.

## Readouts and estimands

Every patch uses the unchanged v0.3 assay battery: canonical singleton, shifted singleton, practiced pair, and alias pair surfaces. Exact signatures and accuracies are retained.

Raw restoration and forward changes are primary measurements. `eta_restore` and `eta_forward` normalize the practiced-pair effect by that seed's observed C0→C1 practiced-pair decline. They remain descriptive continuous coordinates; no threshold or modal necessity/sufficiency label is attached.

## Failure handling

Any v0.3 replay mismatch yields `RECONSTRUCTION FAILURE → STOP` before checkpoint custody creation.

Any checkpoint hash mismatch, parameter-partition mismatch, loaded-endpoint behavior mismatch, no-op failure, full-restoration failure, or nonpositive History-A practiced transition yields `PATCH CONTRACT FAILURE → STOP` before any single-group result.

## Claim boundary

The assay identifies intervention-sensitive parameter groups under exact frozen transplants. It does not identify storage locations, internal representations, forgetting mechanisms, logical necessity/sufficiency, or causal irrelevance from null single-group patches. Interactions are outside v0.4 and require a new prospective experiment.
