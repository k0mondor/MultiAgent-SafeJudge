from __future__ import annotations

import unittest

from safejudge.contracts.artifact import ArtifactRef
from safejudge.core.errors import ProviderError, ProviderErrorKind
from safejudge.models.openai_compat import parse_chat_completion


class ChatCompletionSafetyFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.artifact = ArtifactRef(
            uri="responses/example.json",
            sha256="0" * 64,
            content_type="application/json",
            size_bytes=1,
        )

    def test_normalizes_native_sensitive_empty_response_as_refusal(self) -> None:
        parsed = parse_chat_completion(
            {
                "id": "response-1",
                "model": "example-model",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "native_finish_reason": "sensitive",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning": None,
                        },
                    }
                ],
            },
            provider_label="test provider",
            raw_artifact=self.artifact,
        )

        self.assertEqual(parsed.answer, "[Provider safety filter blocked the response.]")
        self.assertEqual(parsed.finish_reason, "content_filter")

    def test_normalizes_standard_content_filter_empty_response_as_refusal(self) -> None:
        parsed = parse_chat_completion(
            {
                "id": "response-2",
                "choices": [
                    {
                        "finish_reason": "content_filter",
                        "message": {"role": "assistant", "content": ""},
                    }
                ],
            },
            provider_label="test provider",
            raw_artifact=self.artifact,
        )

        self.assertEqual(parsed.answer, "[Provider safety filter blocked the response.]")
        self.assertEqual(parsed.finish_reason, "content_filter")

    def test_keeps_unexplained_empty_response_as_retryable_failure(self) -> None:
        with self.assertRaises(ProviderError) as raised:
            parse_chat_completion(
                {
                    "id": "response-3",
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": None},
                        }
                    ],
                },
                provider_label="test provider",
                raw_artifact=self.artifact,
            )

        self.assertEqual(raised.exception.kind, ProviderErrorKind.EMPTY_RESPONSE)
        self.assertTrue(raised.exception.retryable)


if __name__ == "__main__":
    unittest.main()
