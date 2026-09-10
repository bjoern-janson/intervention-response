"""Regression tests for terminal-only v0.2 forensic characterization."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import experiment
import world
import forensic_v0_2 as f


class SurfaceContractTests(unittest.TestCase):
    def test_strata_are_exact_disjoint_partition_of_v0_equivalence_surface(self):
        self.assertEqual(len(f.T_COMMON), 32)
        self.assertEqual(len(f.T_A_SEEN), 16)
        self.assertEqual(len(f.T_NOVEL), 16)
        self.assertEqual(set(f.T_COMMON) | set(f.T_A_SEEN) | set(f.T_NOVEL), set(world.MANIFEST.equivalence))
        self.assertFalse(set(f.T_COMMON) & set(f.T_A_SEEN))
        self.assertFalse(set(f.T_COMMON) & set(f.T_NOVEL))
        self.assertFalse(set(f.T_A_SEEN) & set(f.T_NOVEL))

    def test_a_seen_overlap_is_history_asymmetric_exactly_as_preregistered(self):
        overlap = f.history_overlap_counts()
        self.assertEqual(overlap["A_seen_unique_items_in_history_A"], 16)
        self.assertEqual(overlap["A_seen_unique_items_in_history_B"], 0)

    def test_execution_population_is_frozen(self):
        f.validate_execution_request(80, smoke=False)
        f.validate_execution_request(2, smoke=True)
        with self.assertRaises(ValueError):
            f.validate_execution_request(40, smoke=False)
        with self.assertRaises(ValueError):
            f.validate_execution_request(3, smoke=True)


class PureMetricTests(unittest.TestCase):
    def test_perfect_labeled_models_have_zero_signature_distance(self):
        labels = tuple(i.target_id for i in world.MANIFEST.equivalence)
        self.assertEqual(f.signature_distance(labels, labels), 0.0)
        f.assert_perfect_signature_implication([
            {"family": "A", "seed": 1, "full_error": 0.0, "signature": list(labels)},
            {"family": "B", "seed": 2, "full_error": 0.0, "signature": list(labels)},
        ])

    def test_distance_matrix_and_nearest_distances_are_exact(self):
        a = [(0, 0, 0, 0), (0, 1, 1, 1)]
        b = [(0, 0, 0, 1), (1, 1, 1, 1)]
        matrix = f.distance_matrix(a, b)
        self.assertEqual(matrix, [[0.25, 1.0], [0.5, 0.25]])
        nearest_a, nearest_b = f.nearest_distances(matrix)
        self.assertEqual(nearest_a, [0.25, 0.25])
        self.assertEqual(nearest_b, [0.25, 0.25])

    def test_signature_overlap_reports_set_and_multiset_overlap(self):
        s1 = (0, 1)
        s2 = (1, 0)
        s3 = (1, 1)
        out = f.signature_overlap([s1, s1, s2], [s1, s3, s3])
        self.assertEqual(out["shared_signature_count"], 1)
        self.assertEqual(out["multiset_overlap"], 1)
        self.assertEqual(sum(out["frequency_A"].values()), 3)
        self.assertEqual(sum(out["frequency_B"].values()), 3)

    def test_error_pattern_counts_preserve_multi_stratum_failures(self):
        records = [
            {"failing_strata": []},
            {"failing_strata": ["T_common"]},
            {"failing_strata": ["T_A_seen", "T_novel"]},
            {"failing_strata": ["T_A_seen", "T_novel"]},
        ]
        self.assertEqual(
            f.error_pattern_counts(records),
            {"PERFECT": 1, "T_common": 1, "T_A_seen+T_novel": 2},
        )


class OrchestrationTests(unittest.TestCase):
    def test_reproduction_mismatch_suppresses_localization_authorization(self):
        gate = f.reproduction_gate(1, scientific=True)
        self.assertEqual(gate["outcome"], "REPRODUCTION MISMATCH")
        self.assertFalse(gate["localization_authorized"])
        ok = f.reproduction_gate(0, scientific=True)
        self.assertTrue(ok["localization_authorized"])

    def test_smoke_is_terminal_only_and_writes_all_required_sections(self):
        with tempfile.TemporaryDirectory() as td, \
             patch.object(experiment, "apply_intervention", side_effect=AssertionError("intervention called")):
            path = Path(td) / "smoke.json"
            result = f.run_forensic(2, path, smoke=True)
        self.assertEqual(result["mode"], "SMOKE")
        self.assertEqual(result["execution"]["models_per_history"], 2)
        self.assertEqual(len(result["models"]["A"]), 2)
        self.assertEqual(len(result["models"]["B"]), 2)
        for family in ("A", "B"):
            for rec in result["models"][family]:
                for field in (
                    "common_accuracy", "A_seen_accuracy", "novel_accuracy",
                    "full_accuracy", "full_error", "signature", "signature_sha256",
                    "failing_strata",
                ):
                    self.assertIn(field, rec)
        self.assertEqual(len(result["cross_history"]["distance_matrix"]), 2)
        self.assertIn("signature_overlap", result["cross_history"])
        self.assertIn("perfect_counts", result["summary"])
        self.assertIn("error_patterns", result["summary"])


if __name__ == "__main__":
    unittest.main()
