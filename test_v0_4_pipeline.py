from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from model import TinyTransformer
from v0_4_tensor_contract import canonical_state_dict_hash, clone_state_dict
from v0_4_reconstruct import (
    EXPECTED_V0_3_BEHAVIORAL_MANIFEST_SHA256,
    expected_signature_map,
    load_behavioral_manifest,
    validate_behavioral_manifest,
)
from v0_4_patch_localization import (
    assert_patch_controls,
    summarize_group_rows,
)


ROOT = Path(__file__).resolve().parent


class V04PipelineTests(unittest.TestCase):
    def test_behavioral_manifest_hash_and_provenance_are_frozen(self):
        manifest = load_behavioral_manifest(ROOT / "V0_3_BEHAVIORAL_SIGNATURES.json.zlib.b64")
        self.assertEqual(EXPECTED_V0_3_BEHAVIORAL_MANIFEST_SHA256, "15c3ae236dfef19d8e46bd8c9a401db3d5eb83c61026c57ace2fdf2c77ac54a8")
        record = validate_behavioral_manifest(manifest)
        self.assertEqual(record["records"], 160)
        self.assertEqual(record["signature_comparisons"], 1280)
        self.assertEqual(record["v0_3_execution_commit"], "214142fea6557c6637926bacff17d90d99484077")

    def test_expected_signature_map_contains_all_1280_keys(self):
        manifest = load_behavioral_manifest(ROOT / "V0_3_BEHAVIORAL_SIGNATURES.json.zlib.b64")
        sigs = expected_signature_map(manifest)
        self.assertEqual(len(sigs), 1280)
        self.assertIn(("A", 1729, "C0", "S_canonical"), sigs)
        self.assertIn(("B", 1730, "C1", "P_alias"), sigs)

    def test_patch_controls_require_tensor_and_signature_identity(self):
        torch.manual_seed(1)
        c0 = clone_state_dict(TinyTransformer())
        torch.manual_seed(2)
        c1 = clone_state_dict(TinyTransformer())
        sig0 = {q: (0, 1, 2) for q in ("S_canonical", "S_shifted", "P_practiced", "P_alias")}
        sig1 = {q: (2, 1, 0) for q in sig0}
        result = assert_patch_controls(
            c0_state=c0,
            c1_state=c1,
            noop_state=clone_state_dict(c1),
            full_restore_state=clone_state_dict(c0),
            c0_signatures=sig0,
            c1_signatures=sig1,
            noop_signatures=dict(sig1),
            full_restore_signatures=dict(sig0),
        )
        self.assertTrue(result["passed"])
        bad = clone_state_dict(c0)
        bad["head.bias"][0] += 1.0
        with self.assertRaisesRegex(AssertionError, "PATCH CONTRACT FAILURE"):
            assert_patch_controls(
                c0_state=c0,
                c1_state=c1,
                noop_state=clone_state_dict(c1),
                full_restore_state=bad,
                c0_signatures=sig0,
                c1_signatures=sig1,
                noop_signatures=dict(sig1),
                full_restore_signatures=dict(sig0),
            )

    def test_group_summary_is_descriptive_and_has_no_binary_effect_label(self):
        rows = [
            {"group": "output_head", "eta_restore": 0.25, "eta_forward": 0.5,
             "restore_effect_practiced": 0.2, "forward_change_practiced": -0.4},
            {"group": "output_head", "eta_restore": 0.75, "eta_forward": 0.0,
             "restore_effect_practiced": 0.6, "forward_change_practiced": 0.0},
        ]
        summary = summarize_group_rows(rows, {"output_head": {"scalar_count": 264}})
        rec = summary["output_head"]
        self.assertAlmostEqual(rec["eta_restore_mean"], 0.5)
        self.assertAlmostEqual(rec["eta_forward_mean"], 0.25)
        self.assertEqual(rec["scalar_count"], 264)
        forbidden = {"necessary", "sufficient", "selective", "significant", "effect_label"}
        self.assertTrue(forbidden.isdisjoint(rec))


if __name__ == "__main__":
    unittest.main()
