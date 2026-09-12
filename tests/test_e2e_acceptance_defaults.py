from __future__ import annotations

import unittest

from scripts.run_e2e_acceptance import _acceptance_status, build_parser


class E2EAcceptanceDefaultsTests(unittest.TestCase):
    def test_taxonomy_version_defaults_to_latest_registered_version(self) -> None:
        args = build_parser().parse_args(
            [
                "--input",
                "input.jsonl",
                "--media-root",
                "media",
                "--target-profile",
                "target",
                "--grounding-profile",
                "grounding",
                "--output-dir",
                "output",
            ]
        )

        self.assertEqual(args.taxonomy, "gb-t-45654-2025-safejudge-v1")
        self.assertIsNone(args.taxonomy_version)
        self.assertFalse(args.replay)
        self.assertEqual(args.acceptance_mode, "strict")

    def test_replay_mode_is_explicit(self) -> None:
        args = build_parser().parse_args(
            [
                "--input",
                "input.jsonl",
                "--media-root",
                "media",
                "--target-profile",
                "target",
                "--grounding-profile",
                "grounding",
                "--output-dir",
                "output",
                "--replay",
            ]
        )

        self.assertTrue(args.replay)

    def test_batch_mode_reports_completed_with_review(self) -> None:
        self.assertEqual(
            _acceptance_status(review_required_count=4, acceptance_mode="batch"),
            "completed_with_review",
        )
        self.assertEqual(
            _acceptance_status(review_required_count=4, acceptance_mode="strict"),
            "review_required",
        )
        self.assertEqual(
            _acceptance_status(review_required_count=0, acceptance_mode="batch"),
            "passed",
        )


if __name__ == "__main__":
    unittest.main()
