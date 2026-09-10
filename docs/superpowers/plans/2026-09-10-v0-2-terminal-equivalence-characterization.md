# V0.2 Terminal Equivalence Characterization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a terminal-only forensic assay that localizes the closed v0 `MATCH FAILURE` without changing any frozen v0 runtime or invoking interventions.

**Architecture:** Keep `world.py`, `data.py`, `model.py`, and `experiment.py` byte-identical to the successful v0 execution. Add one independent `forensic_v0_2.py` consumer that calls only terminal-training/evaluation helpers, one regression-test module, and one branch-scoped GitHub Actions workflow. Freeze a provenance/spec record before execution and upload `v0_2_results.json` plus its SHA-256 as custody.

**Tech Stack:** Python 3.13.5, NumPy 2.3.5, PyTorch 2.10.0, `unittest`, GitHub Actions Ubuntu 24.04.

**Spec:** `V0_2_PREREGISTRATION.md`

## Global Constraints

- V0 remains CLOSED at `MATCH FAILURE`; no v0 source modification.
- V0.2 regenerates exactly 80 A + 80 B terminal models from the frozen v0 source and seeds.
- No call to `apply_intervention`, `frozen_pair_metrics`, or any post-terminal training function is permitted.
- The three strata must be a disjoint exact partition of the 64-item v0 equivalence surface with counts 32/16/16.
- V0 terminal match count must reproduce `0`; otherwise `REPRODUCTION MISMATCH → STOP`.
- Outputs are descriptive terminal-state measurements only.

---

### Task 1: Forensic surface and metric contract

**Files:**
- Create: `test_forensic_v0_2.py`
- Create: `forensic_v0_2.py`

**Interfaces:**
- Consumes: `MANIFEST`, `EQUIVALENCE_EXTRA`, `history_A`, `history_B`, and frozen terminal helpers from `experiment.py`.
- Produces: `T_COMMON`, `T_A_SEEN`, `T_NOVEL`, `signature_frequency`, `population_summary`, and `run_forensic`.

- [ ] **Step 1: Write failing tests** that require exact 32/16/16 partitioning, History-A-only direct overlap for the 16 A-seen items, exact perfect-signature implication, correct all-pairs distance/nearest-distance calculation, signature multiset overlap, and rejection of non-80 scientific population sizes.
- [ ] **Step 2: Run** `python -m unittest -v test_forensic_v0_2.py` and verify RED because `forensic_v0_2` does not exist.
- [ ] **Step 3: Implement minimal terminal-only functions** in `forensic_v0_2.py`; import `terminal_train`, `accuracy`, `signature`, `signature_distance`, `deterministic_match`, and frozen constants only. Do not import or call intervention helpers.
- [ ] **Step 4: Run** `python -m unittest -v test_forensic_v0_2.py` and verify all tests pass.

### Task 2: No-intervention and reproduction gates

**Files:**
- Modify: `test_forensic_v0_2.py`
- Modify: `forensic_v0_2.py`

**Interfaces:**
- Consumes: Task 1 characterization functions.
- Produces: scientific result schema with `reproduction_gate`, per-model records, stratum summaries, all-pairs distances, nearest distances, signature overlap, error-pattern frequencies, and original-gate sufficiency ledger.

- [ ] **Step 1: Add failing tests** that monkeypatch `experiment.apply_intervention` to raise if called, require a terminal-only 2×2 smoke path, require all three stratum metrics in every model record, and require `REPRODUCTION MISMATCH` to suppress localization authorization.
- [ ] **Step 2: Run tests and verify RED** for missing result-schema behavior.
- [ ] **Step 3: Implement the minimal result schema and reproduction gate**; `run_forensic(80, scientific=True)` is the only scientific population and `run_forensic(2, smoke=True)` is the only smoke population.
- [ ] **Step 4: Run tests and verify GREEN**.

### Task 3: Freeze and execute on GitHub

**Files:**
- Create: `.github/workflows/v0.2-terminal-equivalence.yml`
- Create last: `V0_2_EXECUTE`

**Interfaces:**
- Consumes: exact source SHA-256 values from the spec and finalized forensic/test/spec SHA-256 values.
- Produces: one canonical branch-scoped scientific run and custody artifact.

- [ ] **Step 1: Run local verification**: `python -m unittest -v test_forensic_v0_2.py`, `python -m py_compile world.py data.py model.py experiment.py forensic_v0_2.py`, and `python forensic_v0_2.py --smoke --out v0_2_smoke.json`.
- [ ] **Step 2: Compute SHA-256** for frozen v0 inputs plus `forensic_v0_2.py`, `test_forensic_v0_2.py`, and `V0_2_PREREGISTRATION.md`.
- [ ] **Step 3: Add branch-scoped Actions workflow** that verifies those hashes, installs exact runtime versions, reruns tests and smoke, then executes `python forensic_v0_2.py --n-models 80 --out v0_2_results.json`.
- [ ] **Step 4: Add `V0_2_EXECUTE` as the final trigger commit** so no development commit launches a scientific run.
- [ ] **Step 5: Verify run completion and artifact custody**; compare the uploaded result SHA against `v0_2_results.sha256` before interpretation.
