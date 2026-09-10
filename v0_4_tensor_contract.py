"""Frozen tensor identity and parameter-patch contract for Engineered-World v0.4."""
from __future__ import annotations

import hashlib
import json
import struct
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import torch

EXPECTED_TOTAL_TRAINABLE_SCALARS = 18_600

PARAMETER_GROUPS: "OrderedDict[str, tuple[str, ...]]" = OrderedDict([
    ("token_embedding", (
        "token.weight",
    )),
    ("position_embedding", (
        "pos",
    )),
    ("layer0_attention", (
        "layers.0.self_attn.in_proj_weight",
        "layers.0.self_attn.in_proj_bias",
        "layers.0.self_attn.out_proj.weight",
        "layers.0.self_attn.out_proj.bias",
    )),
    ("layer0_ffn", (
        "layers.0.linear1.weight",
        "layers.0.linear1.bias",
        "layers.0.linear2.weight",
        "layers.0.linear2.bias",
    )),
    ("layer0_norms", (
        "layers.0.norm1.weight",
        "layers.0.norm1.bias",
        "layers.0.norm2.weight",
        "layers.0.norm2.bias",
    )),
    ("layer1_attention", (
        "layers.1.self_attn.in_proj_weight",
        "layers.1.self_attn.in_proj_bias",
        "layers.1.self_attn.out_proj.weight",
        "layers.1.self_attn.out_proj.bias",
    )),
    ("layer1_ffn", (
        "layers.1.linear1.weight",
        "layers.1.linear1.bias",
        "layers.1.linear2.weight",
        "layers.1.linear2.bias",
    )),
    ("layer1_norms", (
        "layers.1.norm1.weight",
        "layers.1.norm1.bias",
        "layers.1.norm2.weight",
        "layers.1.norm2.bias",
    )),
    ("final_norm", (
        "norm.weight",
        "norm.bias",
    )),
    ("output_head", (
        "head.weight",
        "head.bias",
    )),
])


def clone_state_dict(model_or_state: Any) -> dict[str, torch.Tensor]:
    state = model_or_state.state_dict() if hasattr(model_or_state, "state_dict") else model_or_state
    return {name: tensor.detach().cpu().clone() for name, tensor in state.items()}


def _assert_same_schema(base: Mapping[str, torch.Tensor], donor: Mapping[str, torch.Tensor]) -> None:
    if set(base) != set(donor):
        raise AssertionError("state_dict key mismatch")
    for name in base:
        a, b = base[name], donor[name]
        if a.dtype != b.dtype:
            raise AssertionError(f"dtype mismatch for {name}")
        if tuple(a.shape) != tuple(b.shape):
            raise AssertionError(f"shape mismatch for {name}")


def tensor_metadata(state: Mapping[str, torch.Tensor]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in sorted(state):
        t = state[name].detach().cpu().contiguous()
        out.append({
            "name": name,
            "dtype": str(t.dtype),
            "shape": list(t.shape),
            "scalar_count": int(t.numel()),
        })
    return out


def canonical_state_dict_hash(state: Mapping[str, torch.Tensor]) -> str:
    """Content identity independent of mapping order and torch container metadata."""
    h = hashlib.sha256()
    h.update(b"engineered-world/v0.4/canonical-state-dict/v1\0")
    for name in sorted(state):
        t = state[name].detach().cpu().contiguous()
        name_b = name.encode("utf-8")
        dtype_b = str(t.dtype).encode("ascii")
        raw = t.numpy().tobytes(order="C")
        h.update(struct.pack(">I", len(name_b)))
        h.update(name_b)
        h.update(struct.pack(">I", len(dtype_b)))
        h.update(dtype_b)
        h.update(struct.pack(">I", t.ndim))
        for dim in t.shape:
            h.update(struct.pack(">q", int(dim)))
        h.update(struct.pack(">Q", len(raw)))
        h.update(raw)
    return h.hexdigest()


def validate_parameter_partition(model: torch.nn.Module) -> dict[str, Any]:
    named = dict(model.named_parameters())
    expected_names = set(named)
    flattened = [name for names in PARAMETER_GROUPS.values() for name in names]
    if len(flattened) != len(set(flattened)):
        raise AssertionError("parameter groups overlap")
    if set(flattened) != expected_names:
        missing = sorted(expected_names - set(flattened))
        extra = sorted(set(flattened) - expected_names)
        raise AssertionError(f"parameter partition mismatch: missing={missing}, extra={extra}")
    total = sum(int(p.numel()) for p in named.values())
    if total != EXPECTED_TOTAL_TRAINABLE_SCALARS:
        raise AssertionError(f"trainable scalar count mismatch: {total}")
    groups: dict[str, Any] = {}
    for group, names in PARAMETER_GROUPS.items():
        groups[group] = {
            "parameter_names": list(names),
            "tensor_count": len(names),
            "scalar_count": sum(int(named[name].numel()) for name in names),
        }
    if sum(v["scalar_count"] for v in groups.values()) != total:
        raise AssertionError("group scalar counts do not sum to total")
    return {
        "schema": "engineered-world/v0.4/parameter-partition/v1",
        "total_trainable_scalars": total,
        "groups": groups,
    }


def apply_group_patch(
    base: Mapping[str, torch.Tensor],
    donor: Mapping[str, torch.Tensor],
    group: str,
) -> dict[str, torch.Tensor]:
    if group not in PARAMETER_GROUPS:
        raise KeyError(group)
    _assert_same_schema(base, donor)
    names = set(PARAMETER_GROUPS[group])
    return {
        name: (donor[name] if name in names else base[name]).detach().cpu().clone()
        for name in base
    }


def apply_full_patch(
    base: Mapping[str, torch.Tensor],
    donor: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    _assert_same_schema(base, donor)
    return {name: donor[name].detach().cpu().clone() for name in base}


def assert_exact_signature_replay(
    expected: Mapping[tuple[str, int, str, str], tuple[int, ...]],
    actual: Mapping[tuple[str, int, str, str], tuple[int, ...]],
) -> int:
    if set(expected) != set(actual):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise AssertionError(f"RECONSTRUCTION FAILURE: key mismatch missing={missing[:3]} extra={extra[:3]}")
    mismatches = [key for key in expected if tuple(expected[key]) != tuple(actual[key])]
    if mismatches:
        raise AssertionError(f"RECONSTRUCTION FAILURE: {len(mismatches)} signature mismatches; first={mismatches[0]}")
    return len(expected)


def checkpoint_identity_record(
    state: Mapping[str, torch.Tensor],
    *,
    family: str,
    seed: int,
    checkpoint: str,
    runtime: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = tensor_metadata(state)
    canonical_hash = canonical_state_dict_hash(state)
    scalar_count = sum(item["scalar_count"] for item in metadata)
    body = {
        "schema": "engineered-world/v0.4/checkpoint-identity/v1",
        "family": family,
        "seed": int(seed),
        "checkpoint": checkpoint,
        "runtime": dict(runtime),
        "source": dict(source),
        "scalar_count": scalar_count,
        "tensors": metadata,
        "canonical_tensor_hash": canonical_hash,
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    body["checkpoint_identity_hash"] = hashlib.sha256(encoded).hexdigest()
    return body


def _positive_transition(c0: float, c1: float) -> float:
    d = float(c0) - float(c1)
    if d <= 0.0:
        raise ValueError("normalized eta requires a positive practiced transition")
    return d


def normalized_restore_eta(c0: float, c1: float, restored: float) -> float:
    return (float(restored) - float(c1)) / _positive_transition(c0, c1)


def normalized_forward_eta(c0: float, c1: float, forward_patched: float) -> float:
    return (float(c0) - float(forward_patched)) / _positive_transition(c0, c1)
