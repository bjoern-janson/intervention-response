"""Prospective v0.4 deterministic endpoint reconstruction and tensor-custody stage.

This stage reconstructs v0.3 C0/C1 endpoints, verifies all custody-recorded
behavioral signatures, and only after a complete replay pass creates checkpoint
custody. It performs no parameter patch experiment.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
import platform
import zlib
from pathlib import Path
from typing import Any

from v0_4_tensor_contract import (
    assert_exact_signature_replay,
    checkpoint_identity_record,
    clone_state_dict,
    validate_parameter_partition,
)

STUDY = "ENGINEERED-WORLD V0.4 — ENDPOINT RECONSTRUCTION / TENSOR CUSTODY"
V0_3_BEHAVIORAL_MANIFEST = Path(__file__).with_name("V0_3_BEHAVIORAL_SIGNATURES.json.zlib.b64")
EXPECTED_V0_3_BEHAVIORAL_MANIFEST_SHA256 = "15c3ae236dfef19d8e46bd8c9a401db3d5eb83c61026c57ace2fdf2c77ac54a8"
V0_3_EXECUTION_COMMIT = "214142fea6557c6637926bacff17d90d99484077"
V0_3_FROZEN_PARENT = "6aca5d204f865809a2566dadf13e1c4418e953a9"
V0_3_ACTIONS_RUN = 34538200968
V0_3_ARTIFACT_SHA256 = "b8967dc837f7c59f01b2d624611eede8e39929b6937c2053c870a0444868475a"
V0_3_RESULTS_SHA256 = "eee8fb85b3fa16b99ffeb3d2471ac6f42e3376599c78e9aa13b845226a43f070"
PROBES = ("S_canonical", "S_shifted", "P_practiced", "P_alias")
CHECKPOINTS = ("C0", "C1")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_behavioral_manifest(path: Path = V0_3_BEHAVIORAL_MANIFEST) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_V0_3_BEHAVIORAL_MANIFEST_SHA256:
        raise AssertionError("v0.3 behavioral-custody manifest hash mismatch")
    try:
        raw = zlib.decompress(base64.b64decode(path.read_bytes(), validate=True))
        manifest = json.loads(raw)
    except Exception as exc:
        raise AssertionError("v0.3 behavioral-custody manifest decode failure") from exc
    validate_behavioral_manifest(manifest)
    return manifest


def validate_behavioral_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    source = manifest.get("source", {})
    expected_source = {
        "v0_3_execution_commit": V0_3_EXECUTION_COMMIT,
        "v0_3_frozen_parent": V0_3_FROZEN_PARENT,
        "v0_3_actions_run": V0_3_ACTIONS_RUN,
        "v0_3_artifact_sha256": V0_3_ARTIFACT_SHA256,
        "v0_3_results_sha256": V0_3_RESULTS_SHA256,
    }
    for key, expected in expected_source.items():
        if source.get(key) != expected:
            raise AssertionError(f"v0.3 behavioral custody provenance mismatch: {key}")
    if tuple(manifest.get("probes", ())) != PROBES:
        raise AssertionError("v0.3 probe order mismatch")
    if tuple(manifest.get("checkpoints", ())) != CHECKPOINTS:
        raise AssertionError("v0.3 checkpoint order mismatch")
    records = manifest.get("records", {})
    record_order = manifest.get("record_order", [])
    expected_order = [f"{checkpoint}:{probe}" for checkpoint in CHECKPOINTS for probe in PROBES]
    if record_order != expected_order:
        raise AssertionError("v0.3 compact record order mismatch")
    if not isinstance(records, dict) or len(records) != 160:
        raise AssertionError("v0.3 behavioral custody must contain 160 seed records")
    comparisons = 0
    seen: set[tuple[str, int]] = set()
    for raw_key, entries in records.items():
        try:
            family, seed_text = raw_key.split(":", 1)
            seed = int(seed_text)
        except Exception as exc:
            raise AssertionError(f"bad v0.3 compact seed key: {raw_key}") from exc
        key = (family, seed)
        if family not in ("A", "B") or key in seen:
            raise AssertionError(f"bad or duplicate v0.3 seed record: {key}")
        seen.add(key)
        if not isinstance(entries, list) or len(entries) != len(expected_order):
            raise AssertionError(f"bad compact signature vector for {raw_key}")
        for digits in entries:
            if not isinstance(digits, str) or not digits or not set(digits) <= set("01234567"):
                raise AssertionError(f"bad signature digits for {raw_key}")
            comparisons += 1
    if comparisons != 1280:
        raise AssertionError(f"expected 1280 signature comparisons, got {comparisons}")
    return {
        "records": len(records),
        "signature_comparisons": comparisons,
        **expected_source,
    }


def expected_signature_map(manifest: dict[str, Any]) -> dict[tuple[str, int, str, str], tuple[int, ...]]:
    out: dict[tuple[str, int, str, str], tuple[int, ...]] = {}
    order = [tuple(item.split(":", 1)) for item in manifest["record_order"]]
    for raw_key, entries in manifest["records"].items():
        family, seed_text = raw_key.split(":", 1)
        seed = int(seed_text)
        for (checkpoint, probe), digits in zip(order, entries):
            out[(family, seed, checkpoint, probe)] = tuple(int(ch) for ch in digits)
    return out


def _seed_schedule(n_models: int) -> dict[str, list[int]]:
    from experiment import BASE_SEED
    return {
        "A": [BASE_SEED + 2 * i for i in range(n_models)],
        "B": [BASE_SEED + 2 * i + 1 for i in range(n_models)],
    }


def _reconstruct_endpoints(seed: int, family: str):
    """Reproduce v0.3 trajectory without evaluating the live model before C1."""
    from data import HistoryConfig, history_A, history_B
    from experiment import (
        COMMON_STEPS, HISTORY_STEPS, LR_COMMON, LR_HISTORY,
        build_model, train_steps,
    )
    from world import MANIFEST

    live = build_model(seed)
    if family == "A":
        train_steps(live, history_A(HistoryConfig("A")), HISTORY_STEPS, LR_HISTORY, seed + 10)
    elif family == "B":
        train_steps(live, history_B(HistoryConfig("B")), HISTORY_STEPS, LR_HISTORY, seed + 20)
    else:
        raise ValueError(family)
    c0 = copy.deepcopy(live)
    train_steps(live, MANIFEST.terminal_train, COMMON_STEPS, LR_COMMON, seed + 30)
    c1 = copy.deepcopy(live)
    return c0, c1


def _probe_signatures(model) -> dict[str, tuple[int, ...]]:
    from experiment import signature
    from v0_3_checkpoint_localization import (
        P_ALIAS, P_PRACTICED, S_CANONICAL, shifted_singleton_signature,
    )
    return {
        "S_canonical": tuple(signature(model, S_CANONICAL)),
        "S_shifted": tuple(shifted_singleton_signature(model)),
        "P_practiced": tuple(signature(model, P_PRACTICED)),
        "P_alias": tuple(signature(model, P_ALIAS)),
    }


def _runtime_record() -> dict[str, str]:
    import numpy
    import torch
    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "device": "cpu",
    }


def _save_state(path: Path, state: dict[str, Any]) -> str:
    import torch
    torch.save(state, path)
    return sha256_file(path)


def run_reconstruction(n_models: int, out_dir: Path, *, smoke: bool = False) -> dict[str, Any]:
    from experiment import N_MODELS_PER_HISTORY

    expected_n = 2 if smoke else N_MODELS_PER_HISTORY
    if n_models != expected_n:
        raise ValueError(f"{'smoke' if smoke else 'scientific'} reconstruction requires {expected_n} models/history")

    manifest = load_behavioral_manifest()
    all_expected = expected_signature_map(manifest)
    schedule = _seed_schedule(n_models)
    selected = {
        key: sig for key, sig in all_expected.items()
        if key[0] in schedule and key[1] in set(schedule[key[0]])
    }
    expected_comparisons = n_models * 2 * 2 * 4
    if len(selected) != expected_comparisons:
        raise AssertionError("RECONSTRUCTION FAILURE: selected replay surface incomplete")

    endpoints: dict[tuple[str, int, str], dict[str, Any]] = {}
    actual: dict[tuple[str, int, str, str], tuple[int, ...]] = {}
    first_model = None
    for family in ("A", "B"):
        for seed in schedule[family]:
            c0_model, c1_model = _reconstruct_endpoints(seed, family)
            if first_model is None:
                first_model = c0_model
            for checkpoint, model in (("C0", c0_model), ("C1", c1_model)):
                sigs = _probe_signatures(model)
                for probe, sig in sigs.items():
                    actual[(family, seed, checkpoint, probe)] = sig
                endpoints[(family, seed, checkpoint)] = clone_state_dict(model)

    checked = assert_exact_signature_replay(selected, actual)
    if checked != expected_comparisons:
        raise AssertionError("RECONSTRUCTION FAILURE: comparison count mismatch")

    runtime = _runtime_record()
    source = {
        "v0_3_execution_commit": V0_3_EXECUTION_COMMIT,
        "v0_3_frozen_parent": V0_3_FROZEN_PARENT,
        "v0_3_actions_run": V0_3_ACTIONS_RUN,
        "v0_3_artifact_sha256": V0_3_ARTIFACT_SHA256,
        "v0_3_results_sha256": V0_3_RESULTS_SHA256,
        "v0_3_behavioral_manifest_sha256": EXPECTED_V0_3_BEHAVIORAL_MANIFEST_SHA256,
        "v0_4_execution_commit": os.environ.get("GITHUB_SHA", "LOCAL_SMOKE" if smoke else "LOCAL_UNAUTHORIZED"),
        "v0_4_freeze_parent": os.environ.get("V0_4_FREEZE_PARENT", "LOCAL_UNSET"),
    }
    partition = validate_parameter_partition(first_model)

    out_dir.mkdir(parents=True, exist_ok=False)
    checkpoints_dir = out_dir / "checkpoints"
    checkpoints_dir.mkdir()
    records = []
    for family in ("A", "B"):
        for seed in schedule[family]:
            for checkpoint in CHECKPOINTS:
                state = endpoints[(family, seed, checkpoint)]
                filename = f"{family}_{seed}_{checkpoint}.pt"
                path = checkpoints_dir / filename
                container_sha = _save_state(path, state)
                identity = checkpoint_identity_record(
                    state,
                    family=family,
                    seed=seed,
                    checkpoint=checkpoint,
                    runtime=runtime,
                    source=source,
                )
                identity["checkpoint_file"] = f"checkpoints/{filename}"
                identity["container_sha256"] = container_sha
                records.append(identity)

    reconstruction = {
        "study": STUDY,
        "mode": "SMOKE" if smoke else "SCIENTIFIC",
        "replay_gate": {
            "required": "exact v0.3 C0/C1 x 4-probe per-seed signature equality",
            "checked_signatures": checked,
            "mismatch_count": 0,
            "passed": True,
        },
        "checkpoint_custody_created": True,
        "runtime": runtime,
        "source": source,
    }
    custody = {
        "schema": "engineered-world/v0.4/checkpoint-custody/v1",
        "mode": reconstruction["mode"],
        "reconstruction_gate": reconstruction["replay_gate"],
        "runtime": runtime,
        "source": source,
        "parameter_partition": partition,
        "checkpoint_count": len(records),
        "records": records,
    }
    (out_dir / "v0_4_reconstruction.json").write_text(json.dumps(reconstruction, indent=2, sort_keys=True))
    (out_dir / "v0_4_checkpoint_custody.json").write_text(json.dumps(custody, indent=2, sort_keys=True))
    return reconstruction


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-models", type=int, default=80)
    ap.add_argument("--out-dir", type=Path, default=Path("v0_4_reconstructed"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_models = 2
    result = run_reconstruction(args.n_models, args.out_dir, smoke=args.smoke)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
