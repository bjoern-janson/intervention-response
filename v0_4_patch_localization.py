"""Prospective v0.4 single-group parameter-update localization stage.

Consumes only a reconstruction artifact that passed exact v0.3 behavioral
replay and created tensor custody. Performs no training and no interactions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping

import torch

from v0_4_tensor_contract import (
    PARAMETER_GROUPS,
    apply_full_patch,
    apply_group_patch,
    canonical_state_dict_hash,
    normalized_forward_eta,
    normalized_restore_eta,
    validate_parameter_partition,
)

PROBES = ("S_canonical", "S_shifted", "P_practiced", "P_alias")
PATCH_FAMILY = "A"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state_equal(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> bool:
    return set(a) == set(b) and all(torch.equal(a[name], b[name]) for name in a)


def assert_patch_controls(
    *,
    c0_state: Mapping[str, torch.Tensor],
    c1_state: Mapping[str, torch.Tensor],
    noop_state: Mapping[str, torch.Tensor],
    full_restore_state: Mapping[str, torch.Tensor],
    c0_signatures: Mapping[str, tuple[int, ...]],
    c1_signatures: Mapping[str, tuple[int, ...]],
    noop_signatures: Mapping[str, tuple[int, ...]],
    full_restore_signatures: Mapping[str, tuple[int, ...]],
) -> dict[str, Any]:
    if canonical_state_dict_hash(noop_state) != canonical_state_dict_hash(c1_state) or not _state_equal(noop_state, c1_state):
        raise AssertionError("PATCH CONTRACT FAILURE: C1->C1 no-op tensor identity failed")
    if canonical_state_dict_hash(full_restore_state) != canonical_state_dict_hash(c0_state) or not _state_equal(full_restore_state, c0_state):
        raise AssertionError("PATCH CONTRACT FAILURE: C1->C0 full-restoration tensor identity failed")
    for probe in PROBES:
        if tuple(noop_signatures[probe]) != tuple(c1_signatures[probe]):
            raise AssertionError(f"PATCH CONTRACT FAILURE: no-op signature mismatch on {probe}")
        if tuple(full_restore_signatures[probe]) != tuple(c0_signatures[probe]):
            raise AssertionError(f"PATCH CONTRACT FAILURE: full-restoration signature mismatch on {probe}")
    return {"passed": True, "tensor_gates": 2, "signature_gates": 8}


def summarize_group_rows(rows: list[dict[str, Any]], partition_groups: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group in PARAMETER_GROUPS:
        g_rows = [r for r in rows if r["group"] == group]
        if not g_rows:
            continue
        restores = [float(r["eta_restore"]) for r in g_rows]
        forwards = [float(r["eta_forward"]) for r in g_rows]
        restore_raw = [float(r["restore_effect_practiced"]) for r in g_rows]
        forward_raw = [float(r["forward_change_practiced"]) for r in g_rows]
        out[group] = {
            "scalar_count": int(partition_groups[group]["scalar_count"]),
            "n_seeds": len(g_rows),
            "eta_restore_mean": mean(restores),
            "eta_restore_median": median(restores),
            "eta_restore_min": min(restores),
            "eta_restore_max": max(restores),
            "eta_forward_mean": mean(forwards),
            "eta_forward_median": median(forwards),
            "eta_forward_min": min(forwards),
            "eta_forward_max": max(forwards),
            "restore_effect_practiced_mean": mean(restore_raw),
            "forward_change_practiced_mean": mean(forward_raw),
        }
    return out


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)


def _load_custody(custody_dir: Path) -> dict[str, Any]:
    path = custody_dir / "v0_4_checkpoint_custody.json"
    custody = json.loads(path.read_text())
    if not custody.get("reconstruction_gate", {}).get("passed"):
        raise AssertionError("PATCH CONTRACT FAILURE: reconstruction gate not passed")
    return custody


def _expected_behavioral_map() -> dict[tuple[str, int, str, str], tuple[int, ...]]:
    from v0_4_reconstruct import expected_signature_map, load_behavioral_manifest
    return expected_signature_map(load_behavioral_manifest())


def _model_from_state(seed: int, state: Mapping[str, torch.Tensor]):
    from experiment import build_model
    model = build_model(seed)
    model.load_state_dict(state, strict=True)
    return model


def _probe_record(model) -> dict[str, dict[str, Any]]:
    from experiment import signature
    from v0_3_checkpoint_localization import (
        P_ALIAS, P_PRACTICED, S_CANONICAL,
        shifted_singleton_signature,
    )
    surfaces = {
        "S_canonical": tuple(signature(model, S_CANONICAL)),
        "S_shifted": tuple(shifted_singleton_signature(model)),
        "P_practiced": tuple(signature(model, P_PRACTICED)),
        "P_alias": tuple(signature(model, P_ALIAS)),
    }
    labels = {
        "S_canonical": tuple(i.target_id for i in S_CANONICAL),
        "S_shifted": tuple(i.target_id for i in S_CANONICAL),
        "P_practiced": tuple(i.target_id for i in P_PRACTICED),
        "P_alias": tuple(i.target_id for i in P_ALIAS),
    }
    return {
        probe: {
            "signature": list(sig),
            "accuracy": sum(p == y for p, y in zip(sig, labels[probe])) / len(sig),
        }
        for probe, sig in surfaces.items()
    }


def _sig_map(record: Mapping[str, Mapping[str, Any]]) -> dict[str, tuple[int, ...]]:
    return {probe: tuple(record[probe]["signature"]) for probe in PROBES}


def _custody_index(custody: dict[str, Any]) -> dict[tuple[str, int, str], dict[str, Any]]:
    out = {}
    for rec in custody["records"]:
        key = (rec["family"], int(rec["seed"]), rec["checkpoint"])
        if key in out:
            raise AssertionError(f"PATCH CONTRACT FAILURE: duplicate checkpoint custody {key}")
        out[key] = rec
    return out


def _verified_state(custody_dir: Path, rec: dict[str, Any]) -> dict[str, torch.Tensor]:
    path = custody_dir / rec["checkpoint_file"]
    if sha256_file(path) != rec["container_sha256"]:
        raise AssertionError("PATCH CONTRACT FAILURE: checkpoint container hash mismatch")
    state = _load_state(path)
    if canonical_state_dict_hash(state) != rec["canonical_tensor_hash"]:
        raise AssertionError("PATCH CONTRACT FAILURE: canonical checkpoint tensor hash mismatch")
    return state


def run_patch_localization(custody_dir: Path, out_path: Path, *, smoke: bool = False) -> dict[str, Any]:
    custody = _load_custody(custody_dir)
    index = _custody_index(custody)
    expected_behavior = _expected_behavioral_map()
    partition_groups = custody["parameter_partition"]["groups"]

    # The intervention target is the History-A collapse localized by v0.3.
    seeds = sorted(seed for family, seed, checkpoint in index if family == PATCH_FAMILY and checkpoint == "C0")
    if smoke:
        seeds = seeds[:2]
    elif len(seeds) != 80:
        raise AssertionError(f"PATCH CONTRACT FAILURE: expected 80 History-A seeds, got {len(seeds)}")

    # First verify checkpoint custody and all no-op/full-restoration controls for
    # every selected seed. No single-group result is computed before this passes.
    endpoint_states: dict[int, tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]] = {}
    endpoint_records: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    control_records = []
    for seed in seeds:
        c0 = _verified_state(custody_dir, index[(PATCH_FAMILY, seed, "C0")])
        c1 = _verified_state(custody_dir, index[(PATCH_FAMILY, seed, "C1")])
        if seed == seeds[0]:
            validate_parameter_partition(_model_from_state(seed, c0))
        c0_rec = _probe_record(_model_from_state(seed, c0))
        c1_rec = _probe_record(_model_from_state(seed, c1))
        for checkpoint, rec in (("C0", c0_rec), ("C1", c1_rec)):
            for probe in PROBES:
                if tuple(rec[probe]["signature"]) != expected_behavior[(PATCH_FAMILY, seed, checkpoint, probe)]:
                    raise AssertionError(f"PATCH CONTRACT FAILURE: loaded endpoint behavior mismatch {(seed, checkpoint, probe)}")

        noop = apply_full_patch(c1, c1)
        full_restore = apply_full_patch(c1, c0)
        noop_rec = _probe_record(_model_from_state(seed, noop))
        full_rec = _probe_record(_model_from_state(seed, full_restore))
        gate = assert_patch_controls(
            c0_state=c0,
            c1_state=c1,
            noop_state=noop,
            full_restore_state=full_restore,
            c0_signatures=_sig_map(c0_rec),
            c1_signatures=_sig_map(c1_rec),
            noop_signatures=_sig_map(noop_rec),
            full_restore_signatures=_sig_map(full_rec),
        )
        control_records.append({"seed": seed, **gate})
        endpoint_states[seed] = (c0, c1)
        endpoint_records[seed] = (c0_rec, c1_rec)

    rows = []
    for seed in seeds:
        c0, c1 = endpoint_states[seed]
        c0_rec, c1_rec = endpoint_records[seed]
        c0_p = float(c0_rec["P_practiced"]["accuracy"])
        c1_p = float(c1_rec["P_practiced"]["accuracy"])
        if c0_p <= c1_p:
            raise AssertionError(f"PATCH CONTRACT FAILURE: expected positive v0.3 practiced transition for A seed {seed}")
        for group in PARAMETER_GROUPS:
            restore_state = apply_group_patch(c1, c0, group)
            forward_state = apply_group_patch(c0, c1, group)
            restore_rec = _probe_record(_model_from_state(seed, restore_state))
            forward_rec = _probe_record(_model_from_state(seed, forward_state))
            restore_effects = {
                probe: float(restore_rec[probe]["accuracy"]) - float(c1_rec[probe]["accuracy"])
                for probe in PROBES
            }
            forward_changes = {
                probe: float(forward_rec[probe]["accuracy"]) - float(c0_rec[probe]["accuracy"])
                for probe in PROBES
            }
            rows.append({
                "family": PATCH_FAMILY,
                "seed": seed,
                "group": group,
                "scalar_count": int(partition_groups[group]["scalar_count"]),
                "baseline_C0": {p: c0_rec[p]["accuracy"] for p in PROBES},
                "baseline_C1": {p: c1_rec[p]["accuracy"] for p in PROBES},
                "restore": restore_rec,
                "forward_patch": forward_rec,
                "restore_effects": restore_effects,
                "forward_changes": forward_changes,
                "restore_effect_practiced": restore_effects["P_practiced"],
                "forward_change_practiced": forward_changes["P_practiced"],
                "eta_restore": normalized_restore_eta(c0_p, c1_p, restore_rec["P_practiced"]["accuracy"]),
                "eta_forward": normalized_forward_eta(c0_p, c1_p, forward_rec["P_practiced"]["accuracy"]),
            })

    result = {
        "study": "ENGINEERED-WORLD V0.4 — CAUSAL PARAMETER-UPDATE LOCALIZATION",
        "mode": "SMOKE" if smoke else "SCIENTIFIC",
        "checkpoint_custody_verified": True,
        "patch_contract_gate": {
            "passed": True,
            "seeds_checked": len(seeds),
            "no_op_tensor_gates": len(seeds),
            "full_restore_tensor_gates": len(seeds),
            "signature_gates": len(seeds) * 8,
        },
        "patch_family": PATCH_FAMILY,
        "parameter_groups": custody["parameter_partition"],
        "rows": rows,
        "summary": summarize_group_rows(rows, partition_groups),
        "decision": "SMOKE ONLY — NOT SCIENTIFIC" if smoke else "V0.4 SINGLE-GROUP LOCALIZATION COMPLETE",
        "claim_ceiling": "intervention-sensitive parameter localization only; no storage-location, representation-content, internal-forgetting, necessity, or sufficiency claim",
    }
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--custody-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("v0_4_results.json"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    result = run_patch_localization(args.custody_dir, args.out, smoke=args.smoke)
    print(json.dumps({"decision": result["decision"], "patch_contract_gate": result["patch_contract_gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
