"""Prospective v0.3 checkpoint/format localization assay.

Scientific role:
- replay the exact v0.2 terminal signatures per seed;
- only after global replay success, evaluate observationally pure C0/C1 probes;
- never apply an intervention or post-terminal update.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence

import numpy as np
import torch

from data import HistoryConfig, history_A, history_B
from experiment import (
    BASE_SEED,
    BATCH_SIZE,
    COMMON_STEPS,
    DEVICE,
    HISTORY_STEPS,
    LR_COMMON,
    LR_HISTORY,
    N_MODELS_PER_HISTORY,
    build_model,
    signature,
    train_steps,
)
from world import CLS_ID, MANIFEST, PAD_ID, SEP_ID, STATE_OFFSET, Item, make_item, STATES

STUDY = "ENGINEERED-WORLD V0.3 — CHECKPOINT / FORMAT LOCALIZATION"
V0_2_EXECUTION_COMMIT = "dfac75f8b0a96d5b100fed446b91cf85b17ee943"
V0_2_ACTIONS_RUN = 34535640475
V0_2_RESULTS_SHA256 = "a23b16134fa82179bed45140257d30c240d19661b62553afe66b4573eaa7dd03"
V0_2_ARTIFACT_SHA256 = "ed299d00ca1dfa7b748c596bfb8bd2f21740865d8caf22bee271caf0db40cae4"
REPLAY_MANIFEST_PATH = Path(__file__).with_name("V0_2_TERMINAL_SIGNATURES.json")
EXPECTED_REPLAY_MANIFEST_SHA256 = "e727adb1a2f525b6c5c871ba5ff94d004cdf19d27e8de6c18ec6e5df6a6f4e9e"

PRACTICED_PAIRS: tuple[tuple[str, str], ...] = (
    ("T1", "T2"),
    ("T2", "T1"),
    ("T1", "T1"),
    ("T2", "T2"),
)

# Pair semantics equal terminally mastered singleton semantics under the frozen world.
ALIAS_PAIR_TO_SINGLETON: dict[tuple[str, str], str] = {
    ("T1", "T4"): "T5",
    ("T4", "T1"): "T5",
    ("T1", "T5"): "T4",
    ("T5", "T1"): "T4",
    ("T4", "T5"): "T1",
    ("T5", "T4"): "T1",
}

S_CANONICAL: tuple[Item, ...] = tuple(MANIFEST.terminal_train)
P_PRACTICED: tuple[Item, ...] = tuple(
    make_item(pair, state) for pair in PRACTICED_PAIRS for state in STATES
)
P_ALIAS: tuple[Item, ...] = tuple(
    make_item(pair, state) for pair in ALIAS_PAIR_TO_SINGLETON for state in STATES
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_replay_manifest(path: Path = REPLAY_MANIFEST_PATH) -> dict:
    if sha256_file(path) != EXPECTED_REPLAY_MANIFEST_SHA256:
        raise AssertionError("v0.2 terminal-signature replay manifest hash mismatch")
    manifest = json.loads(path.read_text())
    if manifest["source"]["v0_2_execution_commit"] != V0_2_EXECUTION_COMMIT:
        raise AssertionError("replay manifest v0.2 commit mismatch")
    if manifest["source"]["v0_2_actions_run"] != V0_2_ACTIONS_RUN:
        raise AssertionError("replay manifest v0.2 run mismatch")
    if manifest["source"]["v0_2_results_sha256"] != V0_2_RESULTS_SHA256:
        raise AssertionError("replay manifest v0.2 results hash mismatch")
    if manifest["source"]["v0_2_artifact_sha256"] != V0_2_ARTIFACT_SHA256:
        raise AssertionError("replay manifest v0.2 artifact hash mismatch")
    return manifest


def expected_signature_map(manifest: dict) -> dict[tuple[str, int], tuple[int, ...]]:
    dominant = tuple(int(ch) for ch in manifest["dominant"])
    if len(dominant) != len(MANIFEST.equivalence):
        raise AssertionError("bad dominant signature length")
    out: dict[tuple[str, int], tuple[int, ...]] = {}
    schedule = seed_schedule(N_MODELS_PER_HISTORY)
    for family in ("A", "B"):
        exceptions = manifest[f"{family}_exceptions"]
        allowed = {str(seed) for seed in schedule[family]}
        if not set(exceptions).issubset(allowed):
            raise AssertionError(f"replay exceptions outside frozen {family} seed schedule")
        for seed in schedule[family]:
            raw = exceptions.get(str(seed), manifest["dominant"])
            sig = tuple(int(ch) for ch in raw)
            if len(sig) != len(MANIFEST.equivalence):
                raise AssertionError(f"bad signature length for {(family, seed)}")
            out[(family, seed)] = sig
    return out


def shifted_singleton_encoding(item: Item) -> tuple[int, ...]:
    """Pure layout control: [CLS, op, PAD, state, SEP].

    Semantics/label are unchanged from the canonical singleton item. The PAD
    position is attention-masked by the frozen TinyTransformer implementation.
    """
    if len(item.ops) != 1:
        raise ValueError("shifted singleton control accepts singleton items only")
    from world import OP_IDS
    return (
        CLS_ID,
        OP_IDS[item.ops[0]],
        PAD_ID,
        STATE_OFFSET + item.state_id,
        SEP_ID,
    )


def _encoded_logits(model, encoded: Sequence[Sequence[int]]) -> torch.Tensor:
    model.eval()
    x = torch.tensor(encoded, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        return model(x)


def shifted_singleton_signature(model) -> tuple[int, ...]:
    enc = [shifted_singleton_encoding(i) for i in S_CANONICAL]
    return tuple(_encoded_logits(model, enc).argmax(dim=-1).tolist())


def shifted_singleton_accuracy(model) -> float:
    pred = shifted_singleton_signature(model)
    labels = tuple(i.target_id for i in S_CANONICAL)
    return sum(p == y for p, y in zip(pred, labels)) / len(labels)


def item_accuracy(model, items: Sequence[Item]) -> float:
    items = tuple(items)
    labels = tuple(i.target_id for i in items)
    pred = signature(model, items)
    return sum(p == y for p, y in zip(pred, labels)) / len(labels)


def validate_probe_surfaces() -> None:
    if len(S_CANONICAL) != 32:
        raise AssertionError("S_canonical must contain 32 terminal singletons")
    if len(P_PRACTICED) != 32:
        raise AssertionError("P_practiced must contain 32 pair items")
    if len(P_ALIAS) != 48:
        raise AssertionError("P_alias must contain 48 pair items")

    hist_a = set(history_A(HistoryConfig("A")))
    hist_b = set(history_B(HistoryConfig("B")))
    if not set(P_PRACTICED).issubset(hist_a):
        raise AssertionError("P_practiced is not fully practiced in History A")
    if set(P_PRACTICED) & hist_b:
        raise AssertionError("History B unexpectedly contains P_practiced pair items")
    if set(P_ALIAS) & hist_a or set(P_ALIAS) & hist_b:
        raise AssertionError("P_alias must be unpracticed as pair serialization in both histories")

    for pair, singleton in ALIAS_PAIR_TO_SINGLETON.items():
        for state in STATES:
            pair_item = make_item(pair, state)
            singleton_item = make_item((singleton,), state)
            if pair_item.target_id != singleton_item.target_id:
                raise AssertionError(f"alias equivalence failed: {pair} != {singleton}")

    # Shifted control must preserve labels while moving state/SEP into pair positions.
    for item in S_CANONICAL:
        enc = shifted_singleton_encoding(item)
        if enc[2] != PAD_ID or enc[3] != STATE_OFFSET + item.state_id or enc[4] != SEP_ID:
            raise AssertionError("shifted singleton layout contract violated")


def _rng_snapshot() -> tuple[object, tuple, torch.Tensor]:
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def _rng_equal(a: tuple[object, tuple, torch.Tensor], b: tuple[object, tuple, torch.Tensor]) -> bool:
    if a[0] != b[0]:
        return False
    na, nb = a[1], b[1]
    if na[0] != nb[0] or not np.array_equal(na[1], nb[1]) or na[2:] != nb[2:]:
        return False
    return torch.equal(a[2], b[2])


def train_with_observational_checkpoint(seed: int, family: str):
    """Return (C0, C1) while preserving the frozen terminal trajectory.

    The C0 model is deep-copied after history training. No evaluation occurs on
    the live model before the unchanged common phase. The copy operation is
    machine-checked not to consume Python/NumPy/Torch RNG state.
    """
    model = build_model(seed)
    if family == "A":
        train_steps(model, history_A(HistoryConfig("A")), HISTORY_STEPS, LR_HISTORY, seed + 10)
    elif family == "B":
        train_steps(model, history_B(HistoryConfig("B")), HISTORY_STEPS, LR_HISTORY, seed + 20)
    else:
        raise ValueError(family)

    before = _rng_snapshot()
    c0 = copy.deepcopy(model)
    after = _rng_snapshot()
    if not _rng_equal(before, after):
        raise AssertionError("checkpoint copy consumed RNG state")

    train_steps(model, MANIFEST.terminal_train, COMMON_STEPS, LR_COMMON, seed + 30)
    c1 = model
    return c0, c1


def seed_schedule(seed_count: int) -> dict[str, list[int]]:
    return {
        "A": [BASE_SEED + 2 * i for i in range(seed_count)],
        "B": [BASE_SEED + 2 * i + 1 for i in range(seed_count)],
    }


def validate_execution_request(seed_count: int, smoke: bool) -> None:
    if smoke:
        if seed_count != 2:
            raise ValueError("v0.3 smoke is frozen to 2 models per history")
    elif seed_count != N_MODELS_PER_HISTORY:
        raise ValueError(f"v0.3 scientific execution is frozen to {N_MODELS_PER_HISTORY} models per history")


def replay_terminal_population(seed_count: int, smoke: bool):
    manifest = load_replay_manifest()
    expected = expected_signature_map(manifest)
    schedule = seed_schedule(seed_count)
    snapshots: dict[tuple[str, int], tuple[object, object]] = {}
    mismatches: list[dict] = []

    for family in ("A", "B"):
        for seed in schedule[family]:
            c0, c1 = train_with_observational_checkpoint(seed, family)
            actual = tuple(signature(c1, MANIFEST.equivalence))
            exp = expected[(family, seed)]
            if actual != exp:
                mismatches.append({
                    "family": family,
                    "seed": seed,
                    "expected_sha256": hashlib.sha256(json.dumps(list(exp), separators=(",", ":")).encode()).hexdigest(),
                    "actual_sha256": hashlib.sha256(json.dumps(list(actual), separators=(",", ":")).encode()).hexdigest(),
                    "distance": sum(a != b for a, b in zip(actual, exp)) / len(exp),
                })
            snapshots[(family, seed)] = (c0, c1)

    # Smoke validates machinery against the first two frozen seeds per family;
    # scientific execution requires all 160 signatures to match exactly.
    expected_total = 4 if smoke else 2 * N_MODELS_PER_HISTORY
    if len(snapshots) != expected_total:
        raise AssertionError("unexpected replay snapshot count")
    return snapshots, mismatches


def _surface_record(model) -> dict:
    canonical_sig = tuple(signature(model, S_CANONICAL))
    shifted_sig = shifted_singleton_signature(model)
    practiced_sig = tuple(signature(model, P_PRACTICED))
    alias_sig = tuple(signature(model, P_ALIAS))

    def pack(sig: tuple[int, ...], labels: tuple[int, ...]) -> dict:
        acc = sum(p == y for p, y in zip(sig, labels)) / len(labels)
        blob = json.dumps(list(sig), separators=(",", ":")).encode()
        return {"accuracy": acc, "signature": list(sig), "signature_sha256": hashlib.sha256(blob).hexdigest()}

    rec = {
        "S_canonical": pack(canonical_sig, tuple(i.target_id for i in S_CANONICAL)),
        "S_shifted": pack(shifted_sig, tuple(i.target_id for i in S_CANONICAL)),
        "P_practiced": pack(practiced_sig, tuple(i.target_id for i in P_PRACTICED)),
        "P_alias": pack(alias_sig, tuple(i.target_id for i in P_ALIAS)),
    }
    rec["format_gap"] = rec["S_canonical"]["accuracy"] - rec["S_shifted"]["accuracy"]
    return rec


def _summary(records: Sequence[dict], checkpoint: str, surface: str) -> dict:
    vals = [float(r[checkpoint][surface]["accuracy"]) for r in records]
    return {
        "mean": float(mean(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
        "perfect_count": int(sum(v == 1.0 for v in vals)),
        "zero_count": int(sum(v == 0.0 for v in vals)),
    }


def _population_summary(records: dict[str, list[dict]]) -> dict:
    out = {"A": {}, "B": {}}
    for family in ("A", "B"):
        for checkpoint in ("C0", "C1"):
            out[family][checkpoint] = {
                surface: _summary(records[family], checkpoint, surface)
                for surface in ("S_canonical", "S_shifted", "P_practiced", "P_alias")
            }
            gaps = [float(r[checkpoint]["format_gap"]) for r in records[family]]
            out[family][checkpoint]["format_gap"] = {
                "mean": float(mean(gaps)),
                "min": float(min(gaps)),
                "max": float(max(gaps)),
            }
    return out


def _contrasts(records: dict[str, list[dict]]) -> dict:
    out: dict[str, dict] = {"history_A_minus_B": {}, "C1_minus_C0": {"A": {}, "B": {}}}
    for checkpoint in ("C0", "C1"):
        out["history_A_minus_B"][checkpoint] = {}
        for surface in ("S_canonical", "S_shifted", "P_practiced", "P_alias"):
            a = mean(r[checkpoint][surface]["accuracy"] for r in records["A"])
            b = mean(r[checkpoint][surface]["accuracy"] for r in records["B"])
            out["history_A_minus_B"][checkpoint][surface] = float(a - b)
    for family in ("A", "B"):
        for surface in ("S_canonical", "S_shifted", "P_practiced", "P_alias"):
            diffs = [r["C1"][surface]["accuracy"] - r["C0"][surface]["accuracy"] for r in records[family]]
            out["C1_minus_C0"][family][surface] = {
                "mean": float(mean(diffs)),
                "min": float(min(diffs)),
                "max": float(max(diffs)),
            }
    return out


def run(seed_count: int, out_path: Path, *, smoke: bool = False) -> dict:
    validate_execution_request(seed_count, smoke)
    validate_probe_surfaces()

    snapshots, mismatches = replay_terminal_population(seed_count, smoke)
    result = {
        "study": STUDY,
        "mode": "SMOKE" if smoke else "SCIENTIFIC",
        "provenance": {
            "v0_2_execution_commit": V0_2_EXECUTION_COMMIT,
            "v0_2_actions_run": V0_2_ACTIONS_RUN,
            "v0_2_results_sha256": V0_2_RESULTS_SHA256,
            "v0_2_artifact_sha256": V0_2_ARTIFACT_SHA256,
            "replay_manifest_sha256": EXPECTED_REPLAY_MANIFEST_SHA256,
        },
        "execution": {
            "models_per_history": seed_count,
            "checkpoints": ["C0_post_history", "C1_post_common"],
            "post_terminal_interventions": 0,
        },
        "replay_gate": {
            "required": "exact per-seed C1 signature equality to v0.2",
            "checked_models": 2 * seed_count,
            "mismatch_count": len(mismatches),
            "mismatches": mismatches,
            "passed": len(mismatches) == 0,
        },
    }

    if mismatches:
        result["decision"] = "REPRODUCTION FAILURE — STOP"
        result["probe_results_authorized"] = False
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        return result

    records: dict[str, list[dict]] = {"A": [], "B": []}
    for family in ("A", "B"):
        for seed in seed_schedule(seed_count)[family]:
            c0, c1 = snapshots[(family, seed)]
            records[family].append({
                "family": family,
                "seed": seed,
                "C0": _surface_record(c0),
                "C1": _surface_record(c1),
            })

    result["probe_results_authorized"] = not smoke
    result["models"] = records
    result["summary"] = _population_summary(records)
    result["contrasts"] = _contrasts(records)
    result["decision"] = "SMOKE ONLY" if smoke else "V0.3 LOCALIZATION COMPLETE"
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-models", type=int, default=N_MODELS_PER_HISTORY)
    ap.add_argument("--out", type=Path, default=Path("v0_3_results.json"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_models = 2
    result = run(args.n_models, args.out, smoke=args.smoke)
    print(json.dumps({
        "mode": result["mode"],
        "replay_gate": result["replay_gate"],
        "decision": result["decision"],
        "probe_results_authorized": result["probe_results_authorized"],
    }, indent=2))


if __name__ == "__main__":
    main()
