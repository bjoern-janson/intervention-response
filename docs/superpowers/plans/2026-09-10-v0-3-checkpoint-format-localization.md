# V0.3 Checkpoint / Format Localization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a prospective terminal-only assay that exactly replays all v0.2 terminal signatures, then measures post-history/post-common composition and format-controlled expression without changing the frozen training trajectory.

**Architecture:** Keep `world.py`, `data.py`, `model.py`, and `experiment.py` byte-identical. Add a custody-derived per-seed replay manifest plus a separate v0.3 runner that deep-copies the post-history checkpoint, continues the original model through the unchanged common phase, globally gates on exact v0.2 C1 replay, and only then evaluates four preregistered probe surfaces. The GitHub Actions workflow is trigger-gated by a future `V0_3_EXECUTE` file and is not scientifically executed as part of the freeze commit.

**Tech Stack:** Python 3.13.5, PyTorch 2.10.0, NumPy 2.3.5, `unittest`, GitHub Actions on Ubuntu 24.04.

**Spec:** `V0_3_PREREGISTRATION.md`

## Global Constraints

- V0 runtime source is immutable.
- V0.2 scientific result custody is immutable.
- Scientific population is exactly 80 A + 80 B.
- No intervention/post-terminal update function may be called.
- Replay must match all 160 v0.2 C1 signatures exactly before probe interpretation.
- `C0` observation must not alter the live model trajectory or RNG state.
- No new operator or world rule may be introduced.
- Scientific execution requires a later explicit `V0_3_EXECUTE` authorization marker.

---

### Task 1: Fossilize v0.2 replay manifest

**Files:**
- Create: `V0_2_TERMINAL_SIGNATURES.json`

**Interfaces:**
- Consumes: verified v0.2 `v0_2_results.json` with SHA-256 `a23b16134fa82179bed45140257d30c240d19661b62553afe66b4573eaa7dd03`.
- Produces: exact `(family, seed) -> 64-position argmax signature` map plus custody provenance.

- [x] Derive all 160 terminal signatures from verified custody.
- [x] Record v0.2 commit, Actions run, result hash, artifact hash, signature surface and encoding.
- [x] Freeze manifest SHA-256 as `e727adb1a2f525b6c5c871ba5ff94d004cdf19d27e8de6c18ec6e5df6a6f4e9e`.

### Task 2: Implement checkpoint and probe runner

**Files:**
- Create: `v0_3_checkpoint_localization.py`

**Interfaces:**
- Consumes: frozen v0 source plus `V0_2_TERMINAL_SIGNATURES.json`.
- Produces: `v0_3_results.json` or smoke equivalent.

- [x] Implement the exact frozen A/B seed schedule.
- [x] Deep-copy the post-history model before common training and assert Python/NumPy/Torch RNG state is unchanged.
- [x] Continue the original model through the unchanged common phase.
- [x] Compare each C1 64-position signature to the v0.2 replay manifest.
- [x] Return `REPRODUCTION FAILURE — STOP` and omit probe interpretation if any replay differs.
- [x] Define `S_canonical`, `S_shifted`, `P_practiced`, and `P_alias` exactly as preregistered.
- [x] Evaluate C0/C1 probe surfaces only after global replay success.
- [x] Record per-model signatures/accuracies and frozen population summaries/contrasts.

### Task 3: Add regression gates

**Files:**
- Create: `test_v0_3_checkpoint_localization.py`

**Interfaces:**
- Consumes: v0.3 runner and frozen v0 source.
- Produces: nine contract regression tests.

- [x] Verify all 160 replay-manifest entries and seed identities.
- [x] Verify probe counts and History-A/History-B membership facts.
- [x] Verify shifted encoding preserves labels and shifts state/SEP into pair positions.
- [x] Exhaustively verify all six alias equalities over all eight states.
- [x] Verify checkpoint-copy path yields an exact terminal state-dict match to original `terminal_train` for a frozen seed.
- [x] Verify scientific/smoke population guards.
- [x] Verify replay failure cannot authorize probes.
- [x] Verify v0.3 source contains no intervention helper references.
- [x] Run a 2×2 smoke and require exact v0.2 replay with `SMOKE ONLY` authorization state.

### Task 4: Freeze prospective GitHub execution wrapper

**Files:**
- Create: `.github/workflows/v0.3-checkpoint-localization.yml`

**Interfaces:**
- Consumes: exact frozen hashes for source, v0.3 spec, replay manifest, runner, and tests.
- Produces: future scientific custody only when `V0_3_EXECUTE` is later created/updated.

- [x] Verify all frozen file hashes before environment setup.
- [x] Pin Python 3.13.5, NumPy 2.3.5, PyTorch 2.10.0.
- [x] Compile and run all v0.3 tests.
- [x] Run the 2×2 smoke and record its SHA-256.
- [x] Run 80×80 only after separate execution-marker authorization.
- [x] Upload v0.3 results, result hash, smoke, smoke hash, preregistration, and replay manifest.

### Task 5: Freeze without executing science

- [ ] Create one atomic commit parented to `forensic-v0.2@dfac75f8b0a96d5b100fed446b91cf85b17ee943`.
- [ ] Point new branch `prospective-v0.3` at that commit.
- [ ] Do not create `V0_3_EXECUTE`.
- [ ] Re-fetch frozen files and verify GitHub bytes/hashes.
- [ ] Report v0.3 as `FROZEN / SCIENTIFIC EXECUTION NOT RUN`.
