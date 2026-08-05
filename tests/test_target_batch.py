from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path

import httpx
import pytest

from safejudge.cli import main
from safejudge.contracts.dataset import (
    CanonicalMultimodalSample,
    MediaPart,
    MediaRef,
    MediaType,
    SourceRecord,
    TextPart,
)
from safejudge.contracts.evaluation import TargetResponse
from safejudge.contracts.model import (
    InputModality,
    InvocationContext,
    ModalityCombination,
    ModelCapabilities,
)
from safejudge.core.errors import ContractValidationError
from safejudge.models.artifacts import FileArtifactStore
from safejudge.models.batch import (
    BatchFileDigest,
    TargetBatchManifest,
    TargetBatchResult,
    run_fake_target_batch,
    run_target_batch,
)
from safejudge.models.local_openai import LocalOpenAIProvider, LocalOpenAISettings


def _write_canonical_batch(
    path: Path,
    media_root: Path,
    *,
    bad_hash: bool = False,
) -> tuple[CanonicalMultimodalSample, ...]:
    samples: list[CanonicalMultimodalSample] = []
    extensions = {
        MediaType.IMAGE: "png",
        MediaType.AUDIO: "mp3",
        MediaType.VIDEO: "mp4",
    }
    mime_types = {
        MediaType.IMAGE: "image/png",
        MediaType.AUDIO: "audio/mpeg",
        MediaType.VIDEO: "video/mp4",
    }
    for media_type in MediaType:
        relative = Path("media") / f"{media_type.value}.{extensions[media_type]}"
        physical = media_root / relative
        physical.parent.mkdir(parents=True, exist_ok=True)
        content = f"real-{media_type.value}-fixture".encode()
        physical.write_bytes(content)
        digest = "a" * 64 if bad_hash else hashlib.sha256(content).hexdigest()
        samples.append(
            CanonicalMultimodalSample(
                sample_id=f"acceptance:{media_type.value}",
                parts=(
                    MediaPart(
                        media=MediaRef(
                            media_type=media_type,
                            uri=relative.as_posix(),
                            mime_type=mime_types[media_type],
                            sha256=digest,
                            size_bytes=len(content),
                        )
                    ),
                    TextPart(text=f"Evaluate the {media_type.value}."),
                ),
                source=SourceRecord(
                    dataset_name="acceptance-fixture",
                    dataset_version="1",
                    original_id=media_type.value,
                ),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{sample.model_dump_json()}\n" for sample in samples),
        encoding="utf-8",
    )
    return tuple(samples)


def test_real_canonical_batch_runs_fake_target_then_hits_persistent_cache(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "canonical.jsonl"
    media_root = tmp_path / "vendor"
    expected_samples = _write_canonical_batch(input_path, media_root)
    store_path = tmp_path / "runs" / "model-calls.sqlite3"
    artifact_root = tmp_path / "artifacts"

    first = asyncio.run(
        run_fake_target_batch(
            input_path=input_path,
            media_root=media_root,
            output_path=tmp_path / "runs" / "pass-1.jsonl",
            store_path=store_path,
            artifact_root=artifact_root,
            context=InvocationContext(experiment_id="acceptance", run_id="pass-1"),
        )
    )
    first_responses = [
        TargetResponse.model_validate_json(line)
        for line in first.output_path.read_text(encoding="utf-8").splitlines()
    ]

    assert first.manifest.sample_count == 3
    assert first.manifest.verified_media_count == 3
    assert first.manifest.modality_counts == {"audio": 1, "image": 1, "video": 1}
    assert first.manifest.cache_hit_count == 0
    assert first.manifest.cache_miss_count == 3
    assert first.manifest.billed_cost_usd == 0
    assert [response.sample_id for response in first_responses] == [
        sample.sample_id for sample in expected_samples
    ]
    assert all(response.raw_artifact is not None for response in first_responses)
    for response in first_responses:
        assert response.raw_artifact is not None
        assert (artifact_root / response.raw_artifact.uri).is_file()

    second = asyncio.run(
        run_fake_target_batch(
            input_path=input_path,
            media_root=media_root,
            output_path=tmp_path / "runs" / "pass-2.jsonl",
            store_path=store_path,
            artifact_root=artifact_root,
            context=InvocationContext(experiment_id="acceptance", run_id="pass-2"),
        )
    )

    assert second.manifest.cache_hit_count == 3
    assert second.manifest.cache_miss_count == 0
    assert second.manifest.billed_cost_usd == 0
    with sqlite3.connect(store_path) as connection:
        call_counts = connection.execute(
            "SELECT cache_hit, COUNT(*) FROM model_calls GROUP BY cache_hit ORDER BY cache_hit"
        ).fetchall()
        attempt_count = connection.execute("SELECT COUNT(*) FROM model_attempts").fetchone()
    assert call_counts == [(0, 3), (1, 3)]
    assert attempt_count == (3,)


def test_batch_rejects_changed_media_before_creating_model_store(tmp_path: Path) -> None:
    input_path = tmp_path / "canonical.jsonl"
    media_root = tmp_path / "vendor"
    _write_canonical_batch(input_path, media_root, bad_hash=True)
    output_path = tmp_path / "runs" / "responses.jsonl"
    store_path = tmp_path / "runs" / "model-calls.sqlite3"

    with pytest.raises(ContractValidationError, match="SHA-256 mismatch"):
        asyncio.run(
            run_fake_target_batch(
                input_path=input_path,
                media_root=media_root,
                output_path=output_path,
                store_path=store_path,
                artifact_root=tmp_path / "artifacts",
                context=InvocationContext(experiment_id="acceptance", run_id="bad"),
            )
        )

    assert not output_path.exists()
    assert not store_path.exists()


def test_local_provider_batch_persists_one_normalized_pilot_response(tmp_path: Path) -> None:
    input_path = tmp_path / "canonical.jsonl"
    media_root = tmp_path / "vendor"
    expected_samples = _write_canonical_batch(input_path, media_root)

    def handler(http_request: httpx.Request) -> httpx.Response:
        del http_request
        return httpx.Response(
            200,
            json={
                "id": "local-generation-1",
                "model": "local/vlm-revision",
                "choices": [
                    {"message": {"content": "local pilot answer"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    artifact_root = tmp_path / "artifacts"
    provider = LocalOpenAIProvider(
        settings=LocalOpenAISettings(_env_file=None, model_id="local/vlm"),
        capabilities=ModelCapabilities(
            input_combinations=(
                ModalityCombination(
                    modalities=frozenset({InputModality.TEXT, InputModality.IMAGE})
                ),
            )
        ),
        artifact_store=FileArtifactStore(artifact_root),
        media_root=media_root,
        client=client,
    )

    result = asyncio.run(
        run_target_batch(
            input_path=input_path,
            media_root=media_root,
            output_path=tmp_path / "runs" / "local-pilot.jsonl",
            store_path=tmp_path / "runs" / "calls.sqlite3",
            context=InvocationContext(experiment_id="local-pilot", run_id="one-image"),
            provider=provider,
            limit=1,
        )
    )
    asyncio.run(client.aclose())
    response = TargetResponse.model_validate_json(
        result.output_path.read_text(encoding="utf-8").strip()
    )

    assert result.manifest.sample_count == 1
    assert response.sample_id == expected_samples[0].sample_id
    assert response.model.provider == "local-openai"
    assert response.text == "local pilot answer"
    assert response.raw_artifact is not None
    assert (artifact_root / response.raw_artifact.uri).is_file()


def test_target_run_jsonl_cli_reports_bridge_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "canonical.jsonl"
    media_root = tmp_path / "vendor"
    _write_canonical_batch(input_path, media_root)

    exit_code = main(
        [
            "target",
            "run-jsonl",
            "--input",
            str(input_path),
            "--media-root",
            str(media_root),
            "--output",
            str(tmp_path / "runs" / "responses.jsonl"),
            "--store",
            str(tmp_path / "runs" / "model-calls.sqlite3"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--run-id",
            "cli-pass",
        ]
    )
    summary = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert summary["samples"] == 3
    assert summary["verified_media"] == 3
    assert summary["cache_misses"] == 3
    assert summary["billed_cost_usd"] == "0"


def test_target_run_jsonl_cli_builds_local_provider_from_dotenv(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".env").write_text(
        "LOCAL_MODEL_BASE_URL=http://127.0.0.1:1234/v1\n"
        "LOCAL_MODEL_ID=local/test-vlm\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    captured: dict[str, object] = {}

    async def fake_run_target_batch(**kwargs: object) -> TargetBatchResult:
        captured.update(kwargs)
        output_path = Path(str(kwargs["output_path"])).resolve()
        store_path = Path(str(kwargs["store_path"])).resolve()
        return TargetBatchResult(
            output_path=output_path,
            manifest_path=output_path.with_suffix(".jsonl.manifest.json"),
            store_path=store_path,
            manifest=TargetBatchManifest(
                input=BatchFileDigest(name="input.jsonl", sha256="a" * 64),
                output=BatchFileDigest(name="output.jsonl", sha256="b" * 64),
                provider="local-openai",
                model="local/test-vlm",
                experiment_id="local-pilot",
                run_id="pilot-1",
                sample_count=1,
                verified_media_count=1,
                modality_counts={"image": 1},
                cache_hit_count=0,
                cache_miss_count=1,
                billed_cost_usd=0,
            ),
        )

    monkeypatch.setattr("safejudge.cli.run_target_batch", fake_run_target_batch)

    exit_code = main(
        [
            "target",
            "run-jsonl",
            "--provider",
            "local",
            "--input",
            "input.jsonl",
            "--media-root",
            "media",
            "--output",
            "output.jsonl",
            "--store",
            "calls.sqlite3",
            "--artifact-root",
            "artifacts",
            "--experiment-id",
            "local-pilot",
            "--run-id",
            "pilot-1",
            "--capability",
            "text+image",
            "--limit",
            "1",
        ]
    )
    summary = json.loads(capsys.readouterr().out)
    provider = captured["provider"]

    assert exit_code == 0
    assert isinstance(provider, LocalOpenAIProvider)
    assert provider.settings.base_url == "http://127.0.0.1:1234/v1"
    assert provider.capabilities.supports(
        frozenset({InputModality.TEXT, InputModality.IMAGE})
    )
    assert captured["limit"] == 1
    assert summary["provider"] == "local-openai"
    assert summary["model"] == "local/test-vlm"


def test_target_run_jsonl_cli_builds_openrouter_provider_before_input_validation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".env").write_text(
        "OPENROUTER_API_KEY=test-secret\n"
        "OPENROUTER_MODEL_ID=vendor/test-vlm\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as caught:
        main(
            [
                "target",
                "run-jsonl",
                "--provider",
                "openrouter",
                "--input",
                "missing.jsonl",
                "--media-root",
                ".",
                "--output",
                "output.jsonl",
                "--store",
                "calls.sqlite3",
                "--artifact-root",
                "artifacts",
                "--run-id",
                "openrouter-pilot",
                "--capability",
                "text+image",
                "--limit",
                "1",
            ]
        )

    assert caught.value.code == 2
    assert "canonical input JSONL does not exist" in capsys.readouterr().err
