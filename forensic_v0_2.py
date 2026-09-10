"""V0.2 terminal-equivalence characterization.

This module is terminal-only. It regenerates the frozen v0 terminal population
and measures where exact equivalence fails. It does not import or call any
intervention, post-terminal update, transfer, control, or crossover helper.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Sequence

from data import HistoryConfig, history_A, history_B
from experiment import (
    BASE_SEED,
    MIN_VALID_PAIRS,
    N_MODELS_PER_HISTORY,
    accuracy,
    deterministic_match,
    signature,
    signature_distance,
    terminal_train,
)
from world import EQUIVALENCE_EXTRA, MANIFEST, Item

V0_PARENT_COMMIT = "158135c60b0b2d10b4bea6d0f3b49de0689504ff"
V0_RUN_ID = 34534073971
V0_RESULTS_SHA256 = "3b635ec2cc326bb46feb1cb589788b163778333188645d86b6ca45f61ee24938"
V0_ARTIFACT_SHA256 = "5c061bb9ba6dcc114fd293e9a05dd3e2f9ff30085b992b838f7a5bf7de0249db"
V0_EXPECTED_TERMINAL_PAIRS = 0

FROZEN_SOURCE_SHA256 = {
    "world.py": "17f7fa7172a1e0927ab9b93df1abcc7fdb3fc23b1838cf480fc1afa5509308cb",
    "data.py": "2e10e6671f97da8b66ad8bfb16c3a52bb78c9f20d87d396566f655d25f6c8b8b",
    "model.py": "2e3447b192c0c88e15012a8dfbfdc44eff5ba789302e090ce2fedcaefe67d674",
    "experiment.py": "0519121e3b1ae218d7105b14cb51355f60aa7babdbf0989488644a09a0d9937f",
}

T_COMMON: tuple[Item, ...] = tuple(MANIFEST.terminal_train)
T_A_SEEN: tuple[Item, ...] = tuple(
    item for item in EQUIVALENCE_EXTRA
    if item.ops in {("T1", "T1"), ("T2", "T2")}
)
T_NOVEL: tuple[Item, ...] = tuple(
    item for item in EQUIVALENCE_EXTRA
    if item.ops in {("T4", "T4"), ("T5", "T5")}
)
T_FULL: tuple[Item, ...] = tuple(MANIFEST.equivalence)

SURFACE_FIELDS = {
    "T_common": "common_accuracy",
    "T_A_seen": "A_seen_accuracy",
    "T_novel": "novel_accuracy",
    "T_full": "full_accuracy",
}


def validate_surfaces() -> None:
    surfaces = (set(T_COMMON), set(T_A_SEEN), set(T_NOVEL))
    if (len(T_COMMON), len(T_A_SEEN), len(T_NOVEL)) != (32, 16, 16):
        raise AssertionError("unexpected v0.2 stratum counts")
    if surfaces[0] & surfaces[1] or surfaces[0] & surfaces[2] or surfaces[1] & surfaces[2]:
        raise AssertionError("v0.2 strata are not disjoint")
    if surfaces[0] | surfaces[1] | surfaces[2] != set(T_FULL):
        raise AssertionError("v0.2 strata do not exactly partition the v0 equivalence surface")


def history_overlap_counts() -> dict[str, int]:
    unique_a = set(history_A(HistoryConfig("A")))
    unique_b = set(history_B(HistoryConfig("B")))
    a_seen = set(T_A_SEEN)
    return {
        "A_seen_unique_items_in_history_A": len(a_seen & unique_a),
        "A_seen_unique_items_in_history_B": len(a_seen & unique_b),
    }


def validate_execution_request(seed_count: int, smoke: bool) -> None:
    if smoke:
        if seed_count != 2:
            raise ValueError("v0.2 smoke execution is frozen to 2 models per history")
        return
    if seed_count != N_MODELS_PER_HISTORY:
        raise ValueError(
            f"v0.2 scientific execution is frozen to {N_MODELS_PER_HISTORY} models per history; got {seed_count}"
        )


def signature_sha256(sig: Sequence[int]) -> str:
    blob = json.dumps(list(sig), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def distance_matrix(signatures_a: Sequence[Sequence[int]], signatures_b: Sequence[Sequence[int]]) -> list[list[float]]:
    return [
        [float(signature_distance(tuple(a), tuple(b))) for b in signatures_b]
        for a in signatures_a
    ]


def nearest_distances(matrix: Sequence[Sequence[float]]) -> tuple[list[float], list[float]]:
    if not matrix:
        return [], []
    if not matrix[0]:
        return [float("nan") for _ in matrix], []
    nearest_a = [float(min(row)) for row in matrix]
    n_cols = len(matrix[0])
    nearest_b = [float(min(matrix[i][j] for i in range(len(matrix)))) for j in range(n_cols)]
    return nearest_a, nearest_b


def signature_overlap(signatures_a: Sequence[Sequence[int]], signatures_b: Sequence[Sequence[int]]) -> dict:
    ca = Counter(tuple(s) for s in signatures_a)
    cb = Counter(tuple(s) for s in signatures_b)
    shared = set(ca) & set(cb)
    return {
        "shared_signature_count": len(shared),
        "multiset_overlap": int(sum(min(ca[s], cb[s]) for s in shared)),
        "frequency_A": {signature_sha256(s): int(n) for s, n in sorted(ca.items())},
        "frequency_B": {signature_sha256(s): int(n) for s, n in sorted(cb.items())},
    }


def error_pattern_counts(records: Sequence[dict]) -> dict[str, int]:
    out: Counter[str] = Counter()
    for rec in records:
        failing = list(rec["failing_strata"])
        key = "+".join(failing) if failing else "PERFECT"
        out[key] += 1
    return dict(sorted(out.items()))


def assert_perfect_signature_implication(records: Sequence[dict]) -> None:
    labels = tuple(item.target_id for item in T_FULL)
    perfect_signatures: set[tuple[int, ...]] = set()
    for rec in records:
        if rec["full_error"] == 0.0:
            sig = tuple(rec["signature"])
            if sig != labels:
                raise AssertionError(
                    f"perfect-labeled model has non-label signature: family={rec['family']} seed={rec['seed']}"
                )
            perfect_signatures.add(sig)
    if len(perfect_signatures) > 1:
        raise AssertionError("full-perfect models have more than one signature")


def _model_record(family: str, seed: int, model) -> dict:
    common = accuracy(model, T_COMMON)
    a_seen = accuracy(model, T_A_SEEN)
    novel = accuracy(model, T_NOVEL)
    full = accuracy(model, T_FULL)
    sig = tuple(signature(model, T_FULL))
    failing: list[str] = []
    if common < 1.0:
        failing.append("T_common")
    if a_seen < 1.0:
        failing.append("T_A_seen")
    if novel < 1.0:
        failing.append("T_novel")
    return {
        "family": family,
        "seed": seed,
        "common_accuracy": common,
        "common_error": 1.0 - common,
        "A_seen_accuracy": a_seen,
        "A_seen_error": 1.0 - a_seen,
        "novel_accuracy": novel,
        "novel_error": 1.0 - novel,
        "full_accuracy": full,
        "full_error": 1.0 - full,
        "signature": list(sig),
        "signature_sha256": signature_sha256(sig),
        "failing_strata": failing,
    }


def _accuracy_summary(records: Sequence[dict], field: str) -> dict[str, float]:
    values = [float(r[field]) for r in records]
    return {
        "mean": float(mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _perfect_counts(records: Sequence[dict]) -> dict[str, int]:
    return {
        surface: sum(1 for r in records if r[field] == 1.0)
        for surface, field in SURFACE_FIELDS.items()
    }


def population_summary(records_a: Sequence[dict], records_b: Sequence[dict]) -> dict:
    perfect_a = _perfect_counts(records_a)
    perfect_b = _perfect_counts(records_b)
    return {
        "perfect_counts": {"A": perfect_a, "B": perfect_b},
        "accuracy": {
            "A": {surface: _accuracy_summary(records_a, field) for surface, field in SURFACE_FIELDS.items()},
            "B": {surface: _accuracy_summary(records_b, field) for surface, field in SURFACE_FIELDS.items()},
        },
        "error_patterns": {
            "A": error_pattern_counts(records_a),
            "B": error_pattern_counts(records_b),
        },
        "original_gate_sufficiency": {
            "minimum": MIN_VALID_PAIRS,
            "A": {surface: count >= MIN_VALID_PAIRS for surface, count in perfect_a.items()},
            "B": {surface: count >= MIN_VALID_PAIRS for surface, count in perfect_b.items()},
        },
        "max_exact_correctness_qualified_pair_capacity": min(
            perfect_a["T_full"], perfect_b["T_full"]
        ),
    }


def reproduction_gate(recomputed_pairs: int, scientific: bool) -> dict:
    if not scientific:
        return {
            "outcome": "SMOKE ONLY",
            "expected_v0_terminal_pairs": V0_EXPECTED_TERMINAL_PAIRS,
            "recomputed_v0_terminal_pairs": recomputed_pairs,
            "localization_authorized": False,
        }
    if recomputed_pairs != V0_EXPECTED_TERMINAL_PAIRS:
        return {
            "outcome": "REPRODUCTION MISMATCH",
            "expected_v0_terminal_pairs": V0_EXPECTED_TERMINAL_PAIRS,
            "recomputed_v0_terminal_pairs": recomputed_pairs,
            "localization_authorized": False,
        }
    return {
        "outcome": "V0 TERMINAL COUNT REPRODUCED",
        "expected_v0_terminal_pairs": V0_EXPECTED_TERMINAL_PAIRS,
        "recomputed_v0_terminal_pairs": recomputed_pairs,
        "localization_authorized": True,
    }


def _nearest_partner_records(
    seeds_a: Sequence[int], seeds_b: Sequence[int], matrix: Sequence[Sequence[float]]
) -> tuple[list[dict], list[dict]]:
    if not matrix or not seeds_a or not seeds_b:
        return [], []
    out_a = []
    for i, seed_a in enumerate(seeds_a):
        j = min(range(len(seeds_b)), key=lambda k: (matrix[i][k], seeds_b[k]))
        out_a.append({"seed": seed_a, "nearest_seed": seeds_b[j], "distance": float(matrix[i][j])})
    out_b = []
    for j, seed_b in enumerate(seeds_b):
        i = min(range(len(seeds_a)), key=lambda k: (matrix[k][j], seeds_a[k]))
        out_b.append({"seed": seed_b, "nearest_seed": seeds_a[i], "distance": float(matrix[i][j])})
    return out_a, out_b


def run_forensic(seed_count: int, out_path: Path, *, smoke: bool = False) -> dict:
    validate_execution_request(seed_count, smoke)
    validate_surfaces()

    models_a = [
        (BASE_SEED + 2 * i, terminal_train(BASE_SEED + 2 * i, "A"))
        for i in range(seed_count)
    ]
    models_b = [
        (BASE_SEED + 2 * i + 1, terminal_train(BASE_SEED + 2 * i + 1, "B"))
        for i in range(seed_count)
    ]

    records_a = [_model_record("A", seed, model) for seed, model in models_a]
    records_b = [_model_record("B", seed, model) for seed, model in models_b]
    all_records = records_a + records_b
    assert_perfect_signature_implication(all_records)

    sigs_a = [tuple(r["signature"]) for r in records_a]
    sigs_b = [tuple(r["signature"]) for r in records_b]
    matrix = distance_matrix(sigs_a, sigs_b)
    nearest_a, nearest_b = nearest_distances(matrix)
    nearest_records_a, nearest_records_b = _nearest_partner_records(
        [s for s, _ in models_a], [s for s, _ in models_b], matrix
    )

    recomputed_pairs = len(deterministic_match(models_a, models_b))
    gate = reproduction_gate(recomputed_pairs, scientific=not smoke)
    summary = population_summary(records_a, records_b)

    result = {
        "mode": "SMOKE" if smoke else "SCIENTIFIC",
        "study": "ENGINEERED-WORLD V0.2 — TERMINAL EQUIVALENCE CHARACTERIZATION",
        "provenance": {
            "v0_parent_commit": V0_PARENT_COMMIT,
            "v0_actions_run": V0_RUN_ID,
            "v0_results_sha256": V0_RESULTS_SHA256,
            "v0_artifact_sha256": V0_ARTIFACT_SHA256,
            "frozen_source_sha256": FROZEN_SOURCE_SHA256,
        },
        "execution": {
            "models_per_history": seed_count,
            "post_terminal_interventions": 0,
        },
        "surfaces": {
            "counts": {
                "T_common": len(T_COMMON),
                "T_A_seen": len(T_A_SEEN),
                "T_novel": len(T_NOVEL),
                "T_full": len(T_FULL),
            },
            "history_overlap": history_overlap_counts(),
        },
        "reproduction_gate": gate,
        "summary": summary,
        "cross_history": {
            "distance_matrix": matrix,
            "nearest_distances_A": nearest_a,
            "nearest_distances_B": nearest_b,
            "nearest_partners_A": nearest_records_a,
            "nearest_partners_B": nearest_records_b,
            "signature_overlap": signature_overlap(sigs_a, sigs_b),
        },
        "models": {"A": records_a, "B": records_b},
    }

    if not smoke:
        result["decision"] = (
            {"outcome": "V0.2 CHARACTERIZATION COMPLETE", "localization_authorized": True}
            if gate["localization_authorized"]
            else {"outcome": "REPRODUCTION MISMATCH", "localization_authorized": False}
        )
    else:
        result["decision"] = {"outcome": "SMOKE ONLY", "localization_authorized": False}

    out_path.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-models", type=int, default=N_MODELS_PER_HISTORY)
    ap.add_argument("--out", type=Path, default=Path("v0_2_results.json"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_models = 2
    result = run_forensic(args.n_models, args.out, smoke=args.smoke)
    print(json.dumps({
        "mode": result["mode"],
        "reproduction_gate": result["reproduction_gate"],
        "perfect_counts": result["summary"]["perfect_counts"],
        "max_pair_capacity": result["summary"]["max_exact_correctness_qualified_pair_capacity"],
    }, indent=2))


if __name__ == "__main__":
    main()
