from __future__ import annotations

import argparse
import unittest

from safejudge.cli import build_parser


class CliTaxonomyDefaultTests(unittest.TestCase):
    @staticmethod
    def _evaluate_args(*extra: str) -> argparse.Namespace:
        return build_parser().parse_args(
            [
                "evaluate",
                "run-jsonl",
                "--input",
                "samples.jsonl",
                "--target-responses",
                "responses.jsonl",
                "--output",
                "evaluations.jsonl",
                "--store",
                "calls.sqlite3",
                "--checkpoint",
                "checkpoints.sqlite3",
                "--node-ledger",
                "node-ledger.sqlite3",
                "--artifact-root",
                "artifacts",
                "--jury-plan",
                "jury.toml",
                "--run-id",
                "taxonomy-default-test",
                *extra,
            ]
        )

    def test_gbt_45654_taxonomy_is_enabled_by_default(self) -> None:
        args = self._evaluate_args()

        self.assertEqual(args.taxonomy, "gb-t-45654-2025-safejudge-v1")
        self.assertIsNone(args.taxonomy_version)

    def test_taxonomy_can_be_explicitly_disabled(self) -> None:
        args = self._evaluate_args("--no-taxonomy")

        self.assertIsNone(args.taxonomy)

    def test_evaluation_cache_only_mode_is_explicit(self) -> None:
        self.assertFalse(self._evaluate_args().cache_only)
        self.assertTrue(self._evaluate_args("--cache-only").cache_only)


if __name__ == "__main__":
    unittest.main()
