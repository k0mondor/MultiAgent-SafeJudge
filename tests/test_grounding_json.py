from __future__ import annotations

import unittest

from safejudge.contracts.dataset import MediaRef  # noqa: F401  # initialize contracts
from safejudge.grounding.tools import _decode_model_json


class GroundingJsonToleranceTests(unittest.TestCase):
    def test_accepts_fenced_json_with_literal_ocr_newlines(self) -> None:
        raw = """```json
{
  "observations": [
    {
      "text": "Methods
to write a
script that denies access",
      "confidence": 0.95,
      "page_or_frame": null
    }
  ]
}
```"""

        decoded = _decode_model_json(raw)

        self.assertIsInstance(decoded, dict)
        assert isinstance(decoded, dict)
        observations = decoded["observations"]
        self.assertIsInstance(observations, list)
        assert isinstance(observations, list)
        self.assertEqual(
            observations[0]["text"],
            "Methods\nto write a\nscript that denies access",
        )

    def test_keeps_strict_schema_json_unchanged(self) -> None:
        raw = '{"observations":[{"text":"house","confidence":1.0}]}'

        self.assertEqual(
            _decode_model_json(raw),
            {"observations": [{"text": "house", "confidence": 1.0}]},
        )


if __name__ == "__main__":
    unittest.main()
