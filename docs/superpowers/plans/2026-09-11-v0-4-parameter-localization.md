# V0.4 Parameter-Update Localization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze a prospective v0.4 assay that reconstructs v0.3 endpoints with exact behavioral replay, creates canonical tensor custody, validates exact patch controls, and localizes the History-A behavioral transition with single-group parameter transplants.

**Architecture:** Three implementation units separate tensor identity/patch semantics, endpoint reconstruction/custody creation, and scientific patch localization. A trigger-gated GitHub Actions workflow validates exact frozen file hashes and executes no science until a later `V0_4_EXECUTE` commit.

**Tech Stack:** Python 3.13.5, PyTorch 2.10.0, NumPy 2.3.5, `unittest`, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-11-v0-4-parameter-localization-design.md`

## Global Constraints

- V0.3 execution source is immutable and must retain its frozen SHA256 identities.
- The v0.3 behavioral bridge contains exactly 160 seeds × 2 checkpoints × 4 probes = 1280 exact signature comparisons.
- Any replay mismatch produces `RECONSTRUCTION FAILURE → STOP` before checkpoint custody.
- Checkpoint identity is canonical tensor content identity, not `.pt` container identity.
- Parameter groups are exact named-tensor sets, disjoint/exhaustive, and total 18,600 trainable scalars.
- V0.4 patch localization operates only on the 80 History-A seeds.
- No single-group patch result exists until every A seed passes loaded-endpoint, no-op, and full-restoration controls.
- No training, gradient update, new probe, interaction search, necessity/sufficiency threshold, or effect-per-parameter normalization is permitted in the patch stage.
- No `V0_4_EXECUTE` marker is added during the prospective freeze.

---

### Task 1: Freeze behavioral custody bridge and tensor contract

**Files:**
- Create: `V0_3_BEHAVIORAL_SIGNATURES.json.zlib.b64`
- Create: `v0_4_tensor_contract.py`
- Test: `test_v0_4_tensor_contract.py`

**Interfaces:**
- Consumes: v0.3 custody artifact and exact `TinyTransformer.named_parameters()`.
- Produces: `PARAMETER_GROUPS`, canonical state-dict hashing, checkpoint identity records, immutable patch constructors, hard replay assertion, and normalized eta functions.

- [ ] **Step 1: Write failing contract tests** for exact 18,600-scalar partition, hash order-independence/value sensitivity, non-mutating group patches, full/no-op patch identity, replay STOP behavior, checkpoint identity binding, eta normalization, and 1280-record custody shape.
- [ ] **Step 2: Run `python -m unittest -v test_v0_4_tensor_contract.py`** and verify failure because `v0_4_tensor_contract` does not exist.
- [ ] **Step 3: Implement the minimal tensor contract** with exact named groups and length-delimited canonical hashing of sorted `(name,dtype,shape,CPU-contiguous-bytes)` records.
- [ ] **Step 4: Re-run the contract tests** and require all tests pass.

### Task 2: Implement reconstruction and patch pipeline gates

**Files:**
- Create: `v0_4_reconstruct.py`
- Create: `v0_4_patch_localization.py`
- Test: `test_v0_4_pipeline.py`

**Interfaces:**
- Consumes: frozen v0.3 source, behavioral custody bridge, tensor contract.
- Produces: reconstruction gate/custody writer and controlled History-A single-group patch evaluator.

- [ ] **Step 1: Write failing pipeline tests** for frozen behavioral-manifest provenance, 1280 expected signature keys, tensor+signature patch controls, and threshold-free descriptive group summaries.
- [ ] **Step 2: Run `python -m unittest -v test_v0_4_pipeline.py`** and verify failure because reconstruction/patch modules do not exist.
- [ ] **Step 3: Implement reconstruction** so the live post-history model is deep-copied without evaluation, common training continues unchanged, all selected C0/C1 probes are evaluated only after C1 exists, and checkpoint files are written only after the global replay gate passes.
- [ ] **Step 4: Implement patch localization** so checkpoint/container/canonical identities and all 80 History-A controls pass globally before any group patch result is computed.
- [ ] **Step 5: Re-run both test modules** and require all tests pass.

### Task 3: Freeze documentation and trigger-gated workflow

**Files:**
- Create: `V0_4_PREREGISTRATION.md`
- Create: `docs/superpowers/specs/2026-09-11-v0-4-parameter-localization-design.md`
- Create: `docs/superpowers/plans/2026-09-11-v0-4-parameter-localization.md`
- Create: `.github/workflows/v0.4-parameter-localization.yml`

**Interfaces:**
- Consumes: exact SHA256s of all frozen source/new assay files.
- Produces: a prospective branch that is executable only by a later marker-only authorization commit.

- [ ] **Step 1: Record the exact v0.3 lineage, 1280 replay gate, checkpoint-custody distinction, 10 groups, patch population, controls, readouts, normalized coordinates, and claim ceiling in the preregistration.**
- [ ] **Step 2: Create a workflow that triggers only on `V0_4_EXECUTE`, verifies exact source/assay hashes and marker-only authorization, pins runtime versions, runs the full test suite, runs a 2×2 smoke, then executes reconstruction before patch localization.**
- [ ] **Step 3: Run `python -m py_compile v0_4_tensor_contract.py v0_4_reconstruct.py v0_4_patch_localization.py test_v0_4_tensor_contract.py test_v0_4_pipeline.py` and the full two-module unittest command.**
- [ ] **Step 4: Verify no `V0_4_EXECUTE` file exists in the freeze tree.**
- [ ] **Step 5: Preserve the actual append-only freeze lineage on `prospective-v0.4`: staging commit `720051b1012756f7a5fb75245c5aa29efec8ca11` adds only the compact v0.3 behavioral-custody bridge on top of v0.3 execution commit `214142fea6557c6637926bacff17d90d99484077`; connector-plumbing commit `a0224bfca61ee063afb1d8d2faf4d189ae3450d0` accidentally added only an empty `NO_SUCH_PATH` file; connector-plumbing commit `6b333de4ae862718b3d8fe90e985f7ff989d50fe` then accidentally added only an empty `__do_not_use__` file. Neither plumbing commit changed an assay object or execution marker. The final freeze commit is parented to `6b333de4ae862718b3d8fe90e985f7ff989d50fe`, uses the clean staging tree as its content base so both stray files are absent, and adds all remaining prospective files. No force rewrite is permitted.**
- [ ] **Step 6: Verify the remote branch head/tree, the append-only ancestry through both provenance commits, exact final blobs, absence of `NO_SUCH_PATH` and `V0_4_EXECUTE` in the final tree, and zero v0.4 workflow runs.**
