from __future__ import annotations

import base64
import copy
import json
import unittest
import zlib
from pathlib import Path

import torch

from model import TinyTransformer
from v0_4_tensor_contract import (
    EXPECTED_TOTAL_TRAINABLE_SCALARS,
    PARAMETER_GROUPS,
    apply_full_patch,
    apply_group_patch,
    assert_exact_signature_replay,
    canonical_state_dict_hash,
    checkpoint_identity_record,
    clone_state_dict,
    normalized_forward_eta,
    normalized_restore_eta,
    validate_parameter_partition,
)


ROOT = Path(__file__).resolve().parent


class V04TensorContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.c0 = TinyTransformer()
        torch.manual_seed(23)
        self.c1 = TinyTransformer()

    def test_partition_is_exact_disjoint_exhaustive_and_18600_scalars(self):
        record = validate_parameter_partition(self.c0)
        self.assertEqual(record["total_trainable_scalars"], 18_600)
        self.assertEqual(EXPECTED_TOTAL_TRAINABLE_SCALARS, 18_600)
        self.assertEqual(len(PARAMETER_GROUPS), 10)
        flattened = [name for names in PARAMETER_GROUPS.values() for name in names]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(flattened), {n for n, _ in self.c0.named_parameters()})
        self.assertEqual(sum(g["scalar_count"] for g in record["groups"].values()), 18_600)

    def test_canonical_hash_ignores_mapping_order_but_detects_value_change(self):
        s = clone_state_dict(self.c0)
        reversed_state = dict(reversed(list(s.items())))
        h0 = canonical_state_dict_hash(s)
        self.assertEqual(h0, canonical_state_dict_hash(reversed_state))
        changed = {k: v.clone() for k, v in s.items()}
        first = sorted(changed)[0]
        changed[first].view(-1)[0] += 1.0
        self.assertNotEqual(h0, canonical_state_dict_hash(changed))

    def test_group_patch_changes_only_frozen_group_and_does_not_mutate_endpoints(self):
        base = clone_state_dict(self.c1)
        donor = clone_state_dict(self.c0)
        before_base = {k: v.clone() for k, v in base.items()}
        before_donor = {k: v.clone() for k, v in donor.items()}
        group = "layer0_attention"
        patched = apply_group_patch(base, donor, group)
        group_names = set(PARAMETER_GROUPS[group])
        for name in base:
            expected = donor[name] if name in group_names else base[name]
            self.assertTrue(torch.equal(patched[name], expected), name)
        for name in base:
            self.assertTrue(torch.equal(base[name], before_base[name]), name)
            self.assertTrue(torch.equal(donor[name], before_donor[name]), name)

    def test_full_patch_is_exact_donor_and_noop_is_exact_base(self):
        c0 = clone_state_dict(self.c0)
        c1 = clone_state_dict(self.c1)
        restored = apply_full_patch(c1, c0)
        noop = apply_full_patch(c1, c1)
        self.assertEqual(canonical_state_dict_hash(restored), canonical_state_dict_hash(c0))
        self.assertEqual(canonical_state_dict_hash(noop), canonical_state_dict_hash(c1))
        for name in c0:
            self.assertTrue(torch.equal(restored[name], c0[name]))
            self.assertTrue(torch.equal(noop[name], c1[name]))

    def test_exact_signature_replay_has_hard_stop_on_any_mismatch(self):
        expected = {("A", 1729, "C0", "S_canonical"): (0, 1, 2)}
        actual = copy.deepcopy(expected)
        self.assertEqual(assert_exact_signature_replay(expected, actual), 1)
        actual[("A", 1729, "C0", "S_canonical")] = (0, 1, 3)
        with self.assertRaisesRegex(AssertionError, "RECONSTRUCTION FAILURE"):
            assert_exact_signature_replay(expected, actual)

    def test_checkpoint_identity_binds_metadata_to_canonical_tensor_hash(self):
        state = clone_state_dict(self.c0)
        record = checkpoint_identity_record(
            state,
            family="A",
            seed=1729,
            checkpoint="C0",
            runtime={"python": "3.13.5", "numpy": "2.3.5", "torch": "2.10.0+cpu", "device": "cpu"},
            source={"v0_3_execution_commit": "abc"},
        )
        self.assertEqual(record["scalar_count"], 18_600)
        self.assertEqual(record["canonical_tensor_hash"], canonical_state_dict_hash(state))
        self.assertEqual(len(record["checkpoint_identity_hash"]), 64)
        changed = clone_state_dict(self.c0)
        changed["head.bias"][0] += 1.0
        changed_record = checkpoint_identity_record(
            changed,
            family="A",
            seed=1729,
            checkpoint="C0",
            runtime=record["runtime"],
            source=record["source"],
        )
        self.assertNotEqual(record["checkpoint_identity_hash"], changed_record["checkpoint_identity_hash"])

    def test_eta_coordinates_use_observed_within_seed_transition(self):
        c0, c1 = 0.90, 0.10
        self.assertAlmostEqual(normalized_restore_eta(c0, c1, 0.50), 0.5)
        self.assertAlmostEqual(normalized_forward_eta(c0, c1, 0.50), 0.5)
        with self.assertRaisesRegex(ValueError, "positive practiced transition"):
            normalized_restore_eta(0.1, 0.1, 0.2)

    def test_behavioral_custody_manifest_is_160_by_2_by_4(self):
        raw = zlib.decompress(base64.b64decode((ROOT / "V0_3_BEHAVIORAL_SIGNATURES.json.zlib.b64").read_bytes(), validate=True))
        manifest = json.loads(raw)
        self.assertEqual(manifest["source"]["v0_3_execution_commit"], "214142fea6557c6637926bacff17d90d99484077")
        self.assertEqual(manifest["source"]["v0_3_actions_run"], 34538200968)
        self.assertEqual(manifest["source"]["v0_3_artifact_sha256"], "b8967dc837f7c59f01b2d624611eede8e39929b6937c2053c870a0444868475a")
        self.assertEqual(manifest["source"]["v0_3_results_sha256"], "eee8fb85b3fa16b99ffeb3d2471ac6f42e3376599c78e9aa13b845226a43f070")
        self.assertIsInstance(manifest["records"], dict)
        self.assertEqual(len(manifest["records"]), 160)
        self.assertEqual(len(manifest["record_order"]), 8)
        comparisons = 0
        for key, entries in manifest["records"].items():
            self.assertRegex(key, r"^[AB]:\d+$")
            self.assertEqual(len(entries), 8)
            for digits in entries:
                self.assertIsInstance(digits, str)
                self.assertTrue(set(digits) <= set("01234567"))
                self.assertGreater(len(digits), 0)
                comparisons += 1
        self.assertEqual(comparisons, 1280)


if __name__ == "__main__":
    unittest.main()
