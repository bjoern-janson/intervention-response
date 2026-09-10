"""Executable repaired v0.1 assay contract.

No v1 correction analysis, mechanism search, hidden-state analysis, or
post-hoc intervention selection is implemented here.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data import HistoryConfig, history_A, history_B, loader
from model import TinyTransformer
from world import MANIFEST, Item, encode_item

# ----------------------------- frozen constants -----------------------------
MODEL_D_MODEL = 32
MODEL_NHEAD = 4
MODEL_LAYERS = 2
MODEL_FF = 64
MODEL_MAX_LEN = 5
MODEL_VOCAB = 32
N_CLASSES = 8

HISTORY_STEPS = 80
COMMON_STEPS = 120
INTERVENTION_STEPS = 1
BATCH_SIZE = 64
LR_HISTORY = 2e-3
LR_COMMON = 2e-3
LR_INTERVENTION = 2e-3
WEIGHT_DECAY = 0.0

EPS_EQ = 0.0
EPS_PAIR = 0.0
TAU_A = 0.0
TAU_0 = 0.05
TAU_T = 0.10
MIN_VALID_PAIRS = 20
N_MODELS_PER_HISTORY = 80
BASE_SEED = 1729
DIAGNOSTIC_INTERVENTION_SEED = 2284
CONTROL_INTERVENTION_SEED = 2506
EXPECTED_DIAG_SIGN = +1
DEVICE = torch.device("cpu")


def validate_execution_request(seed_count: int, smoke: bool) -> None:
    if smoke:
        if seed_count != 2:
            raise ValueError("smoke execution is frozen to 2 models per history")
        return
    if seed_count != N_MODELS_PER_HISTORY:
        raise ValueError(
            f"scientific execution is frozen to {N_MODELS_PER_HISTORY} models per history; "
            f"got {seed_count}"
        )


def contract_record() -> dict:
    return {
        "model": {
            "d_model": MODEL_D_MODEL,
            "heads": MODEL_NHEAD,
            "layers": MODEL_LAYERS,
            "ff": MODEL_FF,
            "max_len": MODEL_MAX_LEN,
            "vocab_size": MODEL_VOCAB,
        },
        "history_steps": HISTORY_STEPS,
        "common_steps": COMMON_STEPS,
        "intervention_steps": INTERVENTION_STEPS,
        "batch_size": BATCH_SIZE,
        "learning_rates": {
            "history": LR_HISTORY,
            "common": LR_COMMON,
            "intervention": LR_INTERVENTION,
        },
        "weight_decay": WEIGHT_DECAY,
        "eps_eq": EPS_EQ,
        "eps_pair": EPS_PAIR,
        "tau_A": TAU_A,
        "tau_0": TAU_0,
        "tau_T": TAU_T,
        "min_valid_pairs": MIN_VALID_PAIRS,
        "models_per_history": N_MODELS_PER_HISTORY,
        "base_seed": BASE_SEED,
        "diagnostic_intervention_seed": DIAGNOSTIC_INTERVENTION_SEED,
        "control_intervention_seed": CONTROL_INTERVENTION_SEED,
        "expected_diag_sign": EXPECTED_DIAG_SIGN,
        "diagnostic_evidence_items": len(MANIFEST.diagnostic_evidence),
        "control_evidence_items": len(MANIFEST.control_evidence),
        "diagnostic_transfer_items": len(MANIFEST.diagnostic_transfer),
        "control_transfer_items": len(MANIFEST.control_transfer),
    }


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(seed: int) -> TinyTransformer:
    seed_all(seed)
    return TinyTransformer(
        vocab_size=MODEL_VOCAB,
        d_model=MODEL_D_MODEL,
        nhead=MODEL_NHEAD,
        num_layers=MODEL_LAYERS,
        dim_ff=MODEL_FF,
        max_len=MODEL_MAX_LEN,
        n_classes=N_CLASSES,
    ).to(DEVICE)


def train_steps(model: nn.Module, items: tuple[Item, ...], steps: int, lr: float, seed: int) -> None:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    dl = loader(items, batch_size=BATCH_SIZE, shuffle=True, seed=seed)
    it = iter(dl)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(dl)
            x, y = next(it)
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        loss_fn(model(x), y).backward()
        opt.step()


def logits(model: nn.Module, items: tuple[Item, ...]) -> torch.Tensor:
    model.eval()
    x = torch.tensor([encode_item(i) for i in items], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        return model(x)


def accuracy(model: nn.Module, items: tuple[Item, ...]) -> float:
    y = torch.tensor([i.target_id for i in items], dtype=torch.long, device=DEVICE)
    p = logits(model, items).argmax(dim=-1)
    return float((p == y).float().mean().item())


def target_probability(model: nn.Module, item: Item) -> float:
    y = torch.tensor([item.target_id], dtype=torch.long, device=DEVICE)
    p = logits(model, (item,)).softmax(dim=-1)[0, y].item()
    return float(p)


def signature(model: nn.Module, items: tuple[Item, ...]) -> tuple[int, ...]:
    return tuple(logits(model, items).argmax(dim=-1).tolist())


def terminal_train(seed: int, family: str) -> TinyTransformer:
    m = build_model(seed)
    if family == "A":
        train_steps(m, history_A(HistoryConfig("A")), HISTORY_STEPS, LR_HISTORY, seed + 10)
    elif family == "B":
        train_steps(m, history_B(HistoryConfig("B")), HISTORY_STEPS, LR_HISTORY, seed + 20)
    else:
        raise ValueError(family)
    train_steps(m, MANIFEST.terminal_train, COMMON_STEPS, LR_COMMON, seed + 30)
    return m


def signature_distance(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    if len(a) != len(b):
        raise ValueError("signature length mismatch")
    if not a:
        return 0.0
    return sum(x != y for x, y in zip(a, b)) / len(a)


def deterministic_match(models_a: list[tuple[int, TinyTransformer]], models_b: list[tuple[int, TinyTransformer]]) -> list[tuple[int, int]]:
    """Seed-ordered matching after correctness gates using frozen pair distance."""
    eligible_b: dict[int, tuple[TinyTransformer, tuple[int, ...]]] = {}
    for seed, model in sorted(models_b, key=lambda z: z[0]):
        err = 1.0 - accuracy(model, MANIFEST.equivalence)
        if err <= EPS_EQ:
            eligible_b[seed] = (model, signature(model, MANIFEST.equivalence))

    pairs: list[tuple[int, int]] = []
    for seed_a, model_a in sorted(models_a, key=lambda z: z[0]):
        err = 1.0 - accuracy(model_a, MANIFEST.equivalence)
        if err > EPS_EQ:
            continue
        sig_a = signature(model_a, MANIFEST.equivalence)
        candidates = [
            (signature_distance(sig_a, sig_b), seed_b)
            for seed_b, (_, sig_b) in eligible_b.items()
            if signature_distance(sig_a, sig_b) <= EPS_PAIR
        ]
        if candidates:
            _, seed_b = min(candidates, key=lambda z: (z[0], z[1]))
            pairs.append((seed_a, seed_b))
            del eligible_b[seed_b]
    return pairs


def apply_intervention(model: TinyTransformer, evidence: tuple[Item, ...], seed: int) -> tuple[float, float]:
    """One optimizer update on the frozen full-truth-table evidence batch."""
    before = accuracy(model, evidence)
    train_steps(model, evidence, INTERVENTION_STEPS, LR_INTERVENTION, seed)
    after = accuracy(model, evidence)
    return before, after


def frozen_pair_metrics(ma: TinyTransformer, mb: TinyTransformer, sa: int, sb: int) -> dict:
    ma_diag = copy.deepcopy(ma)
    mb_diag = copy.deepcopy(mb)
    _, acq_a = apply_intervention(ma_diag, MANIFEST.diagnostic_evidence, DIAGNOSTIC_INTERVENTION_SEED)
    _, acq_b = apply_intervention(mb_diag, MANIFEST.diagnostic_evidence, DIAGNOSTIC_INTERVENTION_SEED)
    diag_a = accuracy(ma_diag, MANIFEST.diagnostic_transfer)
    diag_b = accuracy(mb_diag, MANIFEST.diagnostic_transfer)

    ma_ctrl = copy.deepcopy(ma)
    mb_ctrl = copy.deepcopy(mb)
    _, ctrl_acq_a = apply_intervention(ma_ctrl, MANIFEST.control_evidence, CONTROL_INTERVENTION_SEED)
    _, ctrl_acq_b = apply_intervention(mb_ctrl, MANIFEST.control_evidence, CONTROL_INTERVENTION_SEED)
    ctrl_a = accuracy(ma_ctrl, MANIFEST.control_transfer)
    ctrl_b = accuracy(mb_ctrl, MANIFEST.control_transfer)

    return {
        "seed_a": sa,
        "seed_b": sb,
        "terminal_equiv": True,
        "acquisition_a": acq_a,
        "acquisition_b": acq_b,
        "acquisition_gap": abs(acq_a - acq_b),
        "control_acquisition_a": ctrl_acq_a,
        "control_acquisition_b": ctrl_acq_b,
        "control_acquisition_gap": abs(ctrl_acq_a - ctrl_acq_b),
        "diag_a": diag_a,
        "diag_b": diag_b,
        "delta_diag": diag_a - diag_b,
        "control_a": ctrl_a,
        "control_b": ctrl_b,
        "delta_control": ctrl_a - ctrl_b,
    }


def decide(rows: list[dict]) -> dict:
    if len(rows) < MIN_VALID_PAIRS:
        return {"outcome": "MATCH FAILURE", "n_terminal_pairs": len(rows), "n_valid": 0}
    valid = [
        r for r in rows
        if r["acquisition_gap"] <= TAU_A
        and r["control_acquisition_gap"] <= TAU_A
    ]
    if len(valid) < MIN_VALID_PAIRS:
        return {"outcome": "ACQUISITION CONFOUND", "n_valid": len(valid)}
    control_mean = float(np.mean([r["delta_control"] for r in valid]))
    if abs(control_mean) >= TAU_0:
        return {"outcome": "GENERAL RESPONSE DIFFERENCE", "n_valid": len(valid), "control_mean": control_mean}
    diag_mean = float(np.mean([r["delta_diag"] for r in valid]))
    sign_ok = (diag_mean > 0) if EXPECTED_DIAG_SIGN > 0 else (diag_mean < 0)
    if sign_ok and abs(diag_mean) >= TAU_T:
        return {"outcome": "V0 TARGET SUPPORTED", "n_valid": len(valid), "control_mean": control_mean, "diag_mean": diag_mean}
    return {"outcome": "NULL UNDER V0 ASSAY", "n_valid": len(valid), "control_mean": control_mean, "diag_mean": diag_mean}


def run(seed_count: int, out_path: Path, *, smoke: bool = False) -> dict:
    validate_execution_request(seed_count, smoke)
    models_a = [(BASE_SEED + 2 * i, terminal_train(BASE_SEED + 2 * i, "A")) for i in range(seed_count)]
    models_b = [(BASE_SEED + 2 * i + 1, terminal_train(BASE_SEED + 2 * i + 1, "B")) for i in range(seed_count)]
    pairs = deterministic_match(models_a, models_b)
    a = {s: m for s, m in models_a}
    b = {s: m for s, m in models_b}
    rows = [frozen_pair_metrics(a[sa], b[sb], sa, sb) for sa, sb in pairs]
    result = {
        "mode": "SMOKE" if smoke else "SCIENTIFIC",
        "contract": contract_record(),
        "execution": {"models_per_history": seed_count},
        "n_terminal_pairs": len(pairs),
        "decision": decide(rows),
        "rows": rows,
    }
    out_path.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-models", type=int, default=N_MODELS_PER_HISTORY)
    ap.add_argument("--out", type=Path, default=Path("results.json"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        # Smoke is code-path validation only; no scientific decision is interpreted.
        args.n_models = 2
    result = run(args.n_models, args.out, smoke=args.smoke)
    print(json.dumps({"n_terminal_pairs": result["n_terminal_pairs"], "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
