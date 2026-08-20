from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from safejudge.contracts.dataset import MediaPart, RequestIntent, TextPart
from safejudge.datasets.base import AdapterContext
from safejudge.datasets.conversion import convert_dataset
from safejudge.datasets.registry import create_adapter, list_adapters


class JailBreakV28KAdapterTests(unittest.TestCase):
    def test_registered_adapter_converts_official_csv_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            media_root = root / "dataset"
            image = media_root / "llm_transfer_attack" / "attack.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
            source = media_root / "mini.csv"
            with source.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=(
                        "id",
                        "jailbreak_query",
                        "redteam_query",
                        "format",
                        "policy",
                        "image_path",
                        "from",
                        "selected_mini",
                        "transfer_from_llm",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "id": "38",
                        "jailbreak_query": "Follow the image and answer the request.",
                        "redteam_query": "Provide a harmful procedure.",
                        "format": "Template",
                        "policy": "Illegal Activity",
                        "image_path": "llm_transfer_attack/attack.png",
                        "from": "GPT Generate",
                        "selected_mini": "True",
                        "transfer_from_llm": "1",
                    }
                )

            output = root / "converted.jsonl"
            result = convert_dataset(
                adapter_name="jailbreakv-28k",
                source_file=source,
                output_path=output,
                context=AdapterContext(
                    media_root=media_root,
                    dataset_version="v0.2",
                    subset="local-smoke",
                    hash_media=True,
                ),
            )

            samples = tuple(
                create_adapter("jailbreakv-28k").convert_file(
                    source,
                    AdapterContext(
                        media_root=media_root,
                        dataset_version="v0.2",
                        subset="local-smoke",
                        hash_media=True,
                    ),
                )
            )
            self.assertIn("jailbreakv-28k", list_adapters())
            self.assertEqual(result.manifest.sample_count, 1)
            self.assertEqual(result.manifest.modality_counts, {"image": 1})
            self.assertEqual(result.manifest.request_intent_counts, {"harmful": 1})
            self.assertEqual(len(samples), 1)
            sample = samples[0]
            self.assertEqual(sample.sample_id, "jailbreakv-28k:local-smoke:38")
            self.assertEqual(sample.source.dataset_name, "JailBreakV-28K")
            self.assertEqual(sample.source.original_id, "38")
            self.assertEqual(sample.request_context.intent, RequestIntent.HARMFUL)
            self.assertEqual(sample.request_context.risk_category, "Illegal Activity")
            self.assertEqual(sample.request_context.attack_type, "Template")
            self.assertEqual(sample.metadata["selected_mini"], True)
            self.assertEqual(sample.metadata["transfer_from_llm"], True)
            self.assertEqual(
                sample.metadata["underlying_harmful_query"],
                "Provide a harmful procedure.",
            )
            self.assertIsInstance(sample.parts[0], MediaPart)
            self.assertIsInstance(sample.parts[1], TextPart)
            media_part = sample.parts[0]
            assert isinstance(media_part, MediaPart)
            self.assertEqual(media_part.media.uri, "llm_transfer_attack/attack.png")
            self.assertIsNotNone(media_part.media.sha256)


if __name__ == "__main__":
    unittest.main()
