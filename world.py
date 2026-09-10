"""Frozen 3-bit engineered world for repaired v0.1.

T3 is a withheld diagnostic primitive whose truth table equals T1∘T2 but is
not duplicated by any terminal primitive. T6 is a distinct withheld control
primitive. Held-out transfer targets are mechanically derived from supplied
truth-table evidence plus terminal singleton relations.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence

BITS = (0, 1)
STATES = tuple(product(BITS, repeat=3))
STATE_TO_ID = {s: i for i, s in enumerate(STATES)}
ID_TO_STATE = {i: s for s, i in STATE_TO_ID.items()}


def T1(x: tuple[int, int, int]) -> tuple[int, int, int]:
    a, b, c = x
    return (1 - a, b, c)


def T2(x: tuple[int, int, int]) -> tuple[int, int, int]:
    a, b, c = x
    return (a, 1 - b, c)


def T3(x: tuple[int, int, int]) -> tuple[int, int, int]:
    return T1(T2(x))


def T4(x: tuple[int, int, int]) -> tuple[int, int, int]:
    a, b, c = x
    return (a, b, 1 - c)


def T5(x: tuple[int, int, int]) -> tuple[int, int, int]:
    a, b, c = x
    return (1 - a, b, 1 - c)


def T6(x: tuple[int, int, int]) -> tuple[int, int, int]:
    a, b, c = x
    return (a, 1 - b, 1 - c)

TRANSFORM_NAMES = ("T1", "T2", "T3", "T4", "T5", "T6")
TRANSFORMS = {"T1": T1, "T2": T2, "T3": T3, "T4": T4, "T5": T5, "T6": T6}
OP_IDS = {name: i + 2 for i, name in enumerate(TRANSFORM_NAMES)}
CLS_ID = 1
SEP_ID = 9
STATE_OFFSET = 12
PAD_ID = 0
MAX_LEN = 5


def compose(names: Sequence[str], x: tuple[int, int, int]) -> tuple[int, int, int]:
    out = x
    for name in names:
        out = TRANSFORMS[name](out)
    return out


@dataclass(frozen=True)
class Item:
    ops: tuple[str, ...]
    state_id: int
    target_id: int


def make_item(ops: Sequence[str], state: tuple[int, int, int]) -> Item:
    return Item(tuple(ops), STATE_TO_ID[state], STATE_TO_ID[compose(ops, state)])


def encode_item(item: Item) -> tuple[int, ...]:
    """Fixed-length [CLS, ops..., state, SEP, PAD...] sequence."""
    raw = [CLS_ID, *(OP_IDS[o] for o in item.ops), STATE_OFFSET + item.state_id, SEP_ID]
    if len(raw) > MAX_LEN:
        raise ValueError(f"encoded item too long: {item.ops}")
    return tuple(raw + [PAD_ID] * (MAX_LEN - len(raw)))


def derive_transfer_from_evidence(
    evidence: Sequence[Item],
    contexts: Sequence[str],
    prior_singletons: Sequence[Item],
) -> tuple[Item, ...]:
    """Mechanical closure for new-primitive -> known-context composition.

    The new primitive's complete mapping is read only from supplied evidence.
    The second-step mappings are read only from prior singleton relations.
    No hidden TRANSFORMS lookup is used to construct the held-out targets.
    """
    evidence = tuple(evidence)
    if not evidence:
        raise ValueError("evidence must be non-empty")
    if any(len(i.ops) != 1 for i in evidence):
        raise ValueError("evidence must contain singleton primitive relations")
    new_ops = {i.ops[0] for i in evidence}
    if len(new_ops) != 1:
        raise ValueError("evidence must define exactly one primitive")
    if {i.state_id for i in evidence} != set(range(len(STATES))):
        raise ValueError("evidence must provide the full state truth table")
    new_op = next(iter(new_ops))
    evidence_map = {i.state_id: i.target_id for i in evidence}

    prior_map = {
        (i.ops[0], i.state_id): i.target_id
        for i in prior_singletons
        if len(i.ops) == 1
    }
    out: list[Item] = []
    for context in contexts:
        for state_id in range(len(STATES)):
            mid_id = evidence_map[state_id]
            key = (context, mid_id)
            if key not in prior_map:
                raise ValueError(f"missing prior singleton relation for {key}")
            out.append(Item((new_op, context), state_id, prior_map[key]))
    return tuple(out)


# Terminal experience: identical for both histories. T3 is withheld.
TERMINAL_TRAIN = tuple(
    make_item((op,), s)
    for op in ("T1", "T2", "T4", "T5")
    for s in STATES
)

# Frozen equivalence surface includes terminal singletons and non-diagnostic
# compositions outside the two diagnostic families.
EQUIVALENCE_EXTRA = tuple(
    make_item(pair, s)
    for pair in (("T1", "T1"), ("T2", "T2"), ("T4", "T4"), ("T5", "T5"))
    for s in STATES
)
EQUIVALENCE_SURFACE = TERMINAL_TRAIN + EQUIVALENCE_EXTRA

# Diagnostic intervention evidence: the full singleton truth table for a
# genuinely new primitive token T3. Held-out transfer is defined separately.
DIAGNOSTIC_EVIDENCE = tuple(make_item(("T3",), s) for s in STATES)
DIAGNOSTIC_CONTEXTS = ("T1", "T2")
DIAGNOSTIC_TRANSFER = derive_transfer_from_evidence(
    DIAGNOSTIC_EVIDENCE, DIAGNOSTIC_CONTEXTS, TERMINAL_TRAIN
)

# Negative control: a second genuinely new primitive token T6. Its held-out
# composition contexts T4/T5 were not used as pairwise compositional tasks in
# History A.
CONTROL_EVIDENCE = tuple(make_item(("T6",), s) for s in STATES)
CONTROL_CONTEXTS = ("T4", "T5")
CONTROL_TRANSFER = derive_transfer_from_evidence(
    CONTROL_EVIDENCE, CONTROL_CONTEXTS, TERMINAL_TRAIN
)


@dataclass(frozen=True)
class Manifest:
    terminal_train: tuple[Item, ...]
    equivalence: tuple[Item, ...]
    diagnostic_evidence: tuple[Item, ...]
    diagnostic_transfer: tuple[Item, ...]
    control_evidence: tuple[Item, ...]
    control_transfer: tuple[Item, ...]


MANIFEST = Manifest(
    terminal_train=TERMINAL_TRAIN,
    equivalence=EQUIVALENCE_SURFACE,
    diagnostic_evidence=DIAGNOSTIC_EVIDENCE,
    diagnostic_transfer=DIAGNOSTIC_TRANSFER,
    control_evidence=CONTROL_EVIDENCE,
    control_transfer=CONTROL_TRANSFER,
)


def checksum_items(items: Iterable[Item]) -> str:
    import hashlib, json
    payload = [
        {"ops": list(i.ops), "state_id": i.state_id, "target_id": i.target_id}
        for i in items
    ]
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


if __name__ == "__main__":
    for name, items in (
        ("terminal_train", TERMINAL_TRAIN),
        ("equivalence", EQUIVALENCE_SURFACE),
        ("diagnostic_transfer", DIAGNOSTIC_TRANSFER),
        ("control_transfer", CONTROL_TRANSFER),
    ):
        print(name, len(items), checksum_items(items))
    print("diagnostic_evidence", DIAGNOSTIC_EVIDENCE)
    print("control_evidence", CONTROL_EVIDENCE)
