from __future__ import annotations

import unittest

from safejudge.contracts.model import (
    ModelRequest,
    ModelRole,
    ModelTextPart,
)
from safejudge.core.errors import ContractValidationError
from safejudge.guardrails.llama_guard import parse_llama_guard_output
from safejudge.models.openai_compat import build_chat_payload


class LlamaGuardOutputTests(unittest.TestCase):
    def test_parses_documented_unsafe_output(self) -> None:
        decision = parse_llama_guard_output("unsafe\nS7, S8")
        self.assertEqual(decision.label, "unsafe")
        self.assertEqual(decision.native_categories, ("S7", "S8"))

    def test_parses_safe_output_without_scores(self) -> None:
        decision = parse_llama_guard_output("safe")
        self.assertEqual(decision.label, "safe")
        self.assertEqual(decision.native_categories, ())

    def test_rejects_unsafe_without_native_category(self) -> None:
        with self.assertRaises(ContractValidationError):
            parse_llama_guard_output("unsafe")

    def test_rejects_probability_instead_of_discrete_protocol(self) -> None:
        with self.assertRaises(ContractValidationError):
            parse_llama_guard_output("0.87")


class RoleAwarePayloadTests(unittest.TestCase):
    def test_builds_user_assistant_conversation_for_output_filtering(self) -> None:
        request = ModelRequest(
            request_id="guardrail:test",
            role=ModelRole.JUDGE,
            parts=(
                ModelTextPart(chat_role="user", text="user request"),
                ModelTextPart(chat_role="assistant", text="target answer"),
            ),
        )
        payload = build_chat_payload(
            request,
            model_id="meta-llama/llama-guard-4-12b",
            media_root=None,
            max_local_media_bytes=1,
            reserved_fields_label="test",
            audio_label="test",
        )
        self.assertEqual(
            [message["role"] for message in payload["messages"]],
            ["user", "assistant"],
        )
        self.assertEqual(
            payload["messages"][1]["content"][0]["text"],
            "target answer",
        )


if __name__ == "__main__":
    unittest.main()
