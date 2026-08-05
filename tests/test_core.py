from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path

import pytest
from pydantic import ValidationError

from safejudge.cli import main
from safejudge.core.config import Environment, Settings
from safejudge.core.errors import ProviderError
from safejudge.core.ids import new_run_id, new_sample_id
from safejudge.core.time import utc_now


def test_settings_have_safe_offline_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.environment is Environment.DEVELOPMENT
    assert settings.max_concurrency == 4
    assert str(settings.data_dir) == "data"


def test_settings_reject_invalid_concurrency() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, max_concurrency=0)


def test_ids_are_prefixed_and_unique() -> None:
    run_ids = {new_run_id(), new_run_id()}

    assert len(run_ids) == 2
    assert all(value.startswith("run_") for value in run_ids)
    assert new_sample_id().startswith("sample_")


def test_utc_now_is_timezone_aware() -> None:
    now = utc_now()

    assert now.tzinfo is UTC


def test_provider_error_exposes_retry_policy() -> None:
    error = ProviderError("temporary rate limit", retryable=True)

    assert error.retryable is True


def test_doctor_outputs_machine_readable_offline_report(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["doctor"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["application"] == "multiagent-safejudge"
    assert report["network_used"] is False


def test_cli_lists_dataset_adapters(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["data", "adapters"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["adapters"] == ["mm-safetybench", "mossbench", "omni-safetybench"]


def test_cli_converts_a_dataset_without_network(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = Path(__file__).parent / "fixtures" / "mossbench" / "metadata.jsonl"
    output = tmp_path / "mossbench.jsonl"

    assert (
        main(
            [
                "data",
                "convert",
                "--adapter",
                "mossbench",
                "--input",
                str(source),
                "--media-root",
                str(tmp_path),
                "--output",
                str(output),
                "--subset",
                "oversensitivity",
                "--skip-media-verification",
            ]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["samples"] == 1
    assert report["modalities"] == {"image": 1}
    assert output.is_file()
