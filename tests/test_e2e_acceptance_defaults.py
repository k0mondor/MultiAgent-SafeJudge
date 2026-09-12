from __future__ import annotations

import unittest

from scripts.run_e2e_acceptance import build_parser


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


if __name__ == "__main__":
    unittest.main()
