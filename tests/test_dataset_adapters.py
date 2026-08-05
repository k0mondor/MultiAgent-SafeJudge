from __future__ import annotations

from pathlib import Path

import pytest

from safejudge.contracts.dataset import MediaPart, MediaType, RequestIntent
from safejudge.core.errors import AdapterError
from safejudge.datasets import AdapterContext, create_adapter, list_adapters
from safejudge.datasets.conversion import convert_dataset, file_sha256
from safejudge.datasets.manifest import DatasetManifest

FIXTURES = Path(__file__).parent / "fixtures"


def touch_media(media_root: Path, relative_paths: tuple[str, ...]) -> None:
    for relative_path in relative_paths:
        path = media_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture-media")


def test_registry_exposes_selected_adapters() -> None:
    assert list_adapters() == ("mm-safetybench", "mossbench", "omni-safetybench")


def test_mm_safetybench_expands_three_official_image_text_variants(tmp_path: Path) -> None:
    media_root = tmp_path / "imgs"
    touch_media(
        media_root,
        (
            "01-Illegal_Activitiy/SD/0.jpg",
            "01-Illegal_Activitiy/SD_TYPO/0.jpg",
            "01-Illegal_Activitiy/TYPO/0.jpg",
        ),
    )

    samples = list(
        create_adapter("mm-safetybench").convert_file(
            FIXTURES / "mm_safetybench" / "01-Illegal_Activitiy.json",
            AdapterContext(media_root=media_root, hash_media=True),
        )
    )

    assert len(samples) == 3
    assert {sample.metadata["variant"] for sample in samples} == {"sd", "sd_typo", "typo"}
    assert all(sample.request_context.intent is RequestIntent.HARMFUL for sample in samples)
    first_media = samples[0].parts[0]
    assert isinstance(first_media, MediaPart)
    assert first_media.media.sha256 is not None


def test_mm_safetybench_can_select_a_single_variant_without_media(tmp_path: Path) -> None:
    samples = list(
        create_adapter("mm-safetybench").convert_file(
            FIXTURES / "mm_safetybench" / "01-Illegal_Activitiy.json",
            AdapterContext(
                media_root=tmp_path,
                verify_media=False,
                variants=("sd",),
            ),
        )
    )

    assert len(samples) == 1
    assert samples[0].metadata["variant"] == "sd"


def test_mossbench_marks_benign_image_text_for_oversensitivity(tmp_path: Path) -> None:
    touch_media(tmp_path, ("images/1.png",))

    [sample] = list(
        create_adapter("mossbench").convert_file(
            FIXTURES / "mossbench" / "metadata.jsonl",
            AdapterContext(media_root=tmp_path),
        )
    )

    assert sample.request_context.intent is RequestIntent.BENIGN
    assert sample.request_context.attack_type == "type 1"
    assert sample.source.original_labels["harm"] == 4
    assert sample.metadata["intended_use"] == "test-only; training prohibited by dataset terms"


def test_mossbench_accepts_official_flattened_csv_metadata(tmp_path: Path) -> None:
    touch_media(tmp_path, ("images/2.png",))

    [sample] = list(
        create_adapter("mossbench").convert_file(
            FIXTURES / "mossbench" / "metadata.csv",
            AdapterContext(media_root=tmp_path, subset="oversensitivity"),
        )
    )

    assert sample.sample_id == "mossbench:oversensitivity:2"
    assert sample.request_context.attack_type == "type 2"
    assert sample.source.original_labels["harm"] == [7]
    assert sample.source.original_labels["subset"] == "oversensitivity"


def test_omni_adapter_supports_three_simple_dual_modal_combinations(tmp_path: Path) -> None:
    touch_media(
        tmp_path,
        (
            "mm_data/key_phrase/image/diffusion/Test_10.png",
            "mm_data/key_phrase/audio/tts/Test_11.mp3",
            "mm_data/key_phrase/video/typo/Test_12.mp4",
        ),
    )

    samples = list(
        create_adapter("omni-safetybench").convert_file(
            FIXTURES / "omni_safetybench" / "dual_modal.jsonl",
            AdapterContext(media_root=tmp_path, subset="phase1"),
        )
    )

    assert len(samples) == 3
    assert {sample.metadata["modality_combination"] for sample in samples} == {
        "image-text",
        "audio-text",
        "video-text",
    }
    assert {
        part.media.media_type
        for sample in samples
        for part in sample.parts
        if isinstance(part, MediaPart)
    } == {MediaType.IMAGE, MediaType.AUDIO, MediaType.VIDEO}
    assert {sample.metadata["parallel_seed_id"] for sample in samples} == {"10", "11", "12"}


def test_media_path_traversal_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "unsafe.jsonl"
    source.write_text(
        '{"pid":"1","question":"Benign question","image":"../outside.png"}\n',
        encoding="utf-8",
    )

    with pytest.raises(AdapterError, match="unsafe"):
        list(
            create_adapter("mossbench").convert_file(
                source,
                AdapterContext(media_root=tmp_path, verify_media=False),
            )
        )


def test_conversion_writes_canonical_jsonl_and_reproducibility_manifest(
    tmp_path: Path,
) -> None:
    output = tmp_path / "canonical" / "mossbench.jsonl"

    result = convert_dataset(
        adapter_name="mossbench",
        source_file=FIXTURES / "mossbench" / "metadata.jsonl",
        output_path=output,
        context=AdapterContext(media_root=tmp_path, verify_media=False),
    )

    assert result.manifest.sample_count == 1
    assert result.manifest.modality_counts == {"image": 1}
    assert result.manifest.request_intent_counts == {"benign": 1}
    assert len(output.read_text(encoding="utf-8").splitlines()) == 1
    saved_manifest = DatasetManifest.model_validate_json(
        result.manifest_path.read_text(encoding="utf-8")
    )
    assert saved_manifest.output.sha256 == file_sha256(output)

    with pytest.raises(AdapterError, match="already exists"):
        convert_dataset(
            adapter_name="mossbench",
            source_file=FIXTURES / "mossbench" / "metadata.jsonl",
            output_path=output,
            context=AdapterContext(media_root=tmp_path, verify_media=False),
        )


def test_conversion_limit_supports_small_real_data_acceptance(tmp_path: Path) -> None:
    output = tmp_path / "limited.jsonl"

    result = convert_dataset(
        adapter_name="mm-safetybench",
        source_file=FIXTURES / "mm_safetybench" / "01-Illegal_Activitiy.json",
        output_path=output,
        context=AdapterContext(media_root=tmp_path, verify_media=False),
        limit=2,
    )

    assert result.manifest.sample_count == 2
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2

    with pytest.raises(AdapterError, match="at least 1"):
        convert_dataset(
            adapter_name="mm-safetybench",
            source_file=FIXTURES / "mm_safetybench" / "01-Illegal_Activitiy.json",
            output_path=tmp_path / "invalid.jsonl",
            context=AdapterContext(media_root=tmp_path, verify_media=False),
            limit=0,
        )
