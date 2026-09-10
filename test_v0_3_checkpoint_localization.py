from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import v0_3_checkpoint_localization as v03
from data import HistoryConfig, history_A, history_B
from experiment import BASE_SEED, terminal_train
from world import MANIFEST, make_item, STATES


class V03ContractTests(unittest.TestCase):
    def test_replay_manifest_exact_population_and_seed_schedule(self):
        manifest = v03.load_replay_manifest()
        mapping = v03.expected_signature_map(manifest)
        self.assertEqual(len(mapping), 160)
        for i in range(80):
            self.assertIn(("A", BASE_SEED + 2 * i), mapping)
            self.assertIn(("B", BASE_SEED + 2 * i + 1), mapping)
            self.assertEqual(len(mapping[("A", BASE_SEED + 2 * i)]), 64)
            self.assertEqual(len(mapping[("B", BASE_SEED + 2 * i + 1)]), 64)

    def test_probe_surface_counts_and_history_membership(self):
        v03.validate_probe_surfaces()
        self.assertEqual(len(v03.S_CANONICAL), 32)
        self.assertEqual(len(v03.P_PRACTICED), 32)
        self.assertEqual(len(v03.P_ALIAS), 48)
        self.assertTrue(set(v03.P_PRACTICED).issubset(set(history_A(HistoryConfig("A")))))
        self.assertFalse(set(v03.P_PRACTICED) & set(history_B(HistoryConfig("B"))))

    def test_shifted_singleton_is_semantically_identical_and_pair_positioned(self):
        for item in v03.S_CANONICAL:
            enc = v03.shifted_singleton_encoding(item)
            self.assertEqual(enc[0], 1)
            self.assertEqual(enc[2], 0)
            self.assertEqual(enc[3], 12 + item.state_id)
            self.assertEqual(enc[4], 9)
            self.assertEqual(item.target_id, make_item(item.ops, STATES[item.state_id]).target_id)

    def test_alias_pairs_equal_frozen_known_singleton_semantics(self):
        for pair, singleton in v03.ALIAS_PAIR_TO_SINGLETON.items():
            for state in STATES:
                self.assertEqual(
                    make_item(pair, state).target_id,
                    make_item((singleton,), state).target_id,
                )

    def test_checkpoint_copy_does_not_change_terminal_model(self):
        # One frozen seed is enough to test the implementation invariant.
        seed = BASE_SEED
        _, c1 = v03.train_with_observational_checkpoint(seed, "A")
        direct = terminal_train(seed, "A")
        self.assertEqual(
            tuple(v03.signature(c1, MANIFEST.equivalence)),
            tuple(v03.signature(direct, MANIFEST.equivalence)),
        )
        for a, b in zip(c1.state_dict().values(), direct.state_dict().values()):
            self.assertTrue(torch.equal(a, b))

    def test_execution_population_is_frozen(self):
        v03.validate_execution_request(2, True)
        v03.validate_execution_request(80, False)
        with self.assertRaises(ValueError):
            v03.validate_execution_request(3, True)
        with self.assertRaises(ValueError):
            v03.validate_execution_request(79, False)

    def test_replay_failure_blocks_probe_authorization(self):
        mismatch = [{"family": "A", "seed": BASE_SEED, "distance": 1.0 / 64.0}]
        with tempfile.TemporaryDirectory() as td, patch.object(
            v03, "replay_terminal_population", return_value=({}, mismatch)
        ), patch.object(v03, "_surface_record", side_effect=AssertionError("probe evaluation must not run")):
            out = Path(td) / "blocked.json"
            result = v03.run(80, out, smoke=False)
            self.assertEqual(result["decision"], "REPRODUCTION FAILURE — STOP")
            self.assertFalse(result["probe_results_authorized"])
            self.assertNotIn("models", result)
            self.assertNotIn("summary", result)

    def test_no_intervention_helper_reference_in_v03_source(self):
        src = Path(v03.__file__).read_text()
        forbidden = ("apply_intervention", "frozen_pair_metrics", "DIAGNOSTIC_EVIDENCE", "CONTROL_EVIDENCE")
        for token in forbidden:
            self.assertNotIn(token, src)

    def test_smoke_replays_first_two_seeds_without_authorizing_science(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "smoke.json"
            result = v03.run(2, out, smoke=True)
            self.assertTrue(result["replay_gate"]["passed"])
            self.assertEqual(result["replay_gate"]["mismatch_count"], 0)
            self.assertEqual(result["decision"], "SMOKE ONLY")
            self.assertFalse(result["probe_results_authorized"])
            self.assertEqual(result["execution"]["post_terminal_interventions"], 0)


if __name__ == "__main__":
    unittest.main()
