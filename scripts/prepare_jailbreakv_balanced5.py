"""Prepare the fixed five-sample JailBreakV suite from local upstream files."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from safejudge.contracts.dataset import DatasetSplit
from safejudge.datasets.base import AdapterContext
from safejudge.datasets.conversion import convert_dataset

_CSV_FIELDS = (
    "id",
    "jailbreak_query",
    "redteam_query",
    "format",
    "policy",
    "image_path",
    "from",
    "selected_mini",
    "transfer_from_llm",
)


@dataclass(frozen=True, slots=True)
class Selection:
    source_name: str
    original_id: str
    policy: str
    attack_format: str


_BALANCED_5 = (
    Selection("mini_local_available.csv", "1", "Economic Harm", "Template"),
    Selection("mini_local_available.csv", "88", "Unethical Behavior", "Template"),
    Selection("JailBreakV_28K.csv", "21642", "Malware", "figstep"),
    Selection("JailBreakV_28K.csv", "28358", "Privacy Violation", "SD"),
    Selection("JailBreakV_28K.csv", "24762", "Hate Speech", "SD_typo"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the fixed JailBreakV balanced-5 Canonical JSONL suite",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"D:\JailBreakV_28K\JailBreakV_28K"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/formal/jailbreakv-balanced-5.jsonl"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_root = args.dataset_root.resolve()
    output_path = args.output.resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset root does not exist: {dataset_root}")

    indexes: dict[str, dict[str, dict[str, str]]] = {}
    for selection in _BALANCED_5:
        if selection.source_name not in indexes:
            indexes[selection.source_name] = _load_index(
                dataset_root / selection.source_name
            )

    selected_rows: list[dict[str, str]] = []
    for selection in _BALANCED_5:
        try:
            row = indexes[selection.source_name][selection.original_id]
        except KeyError as error:
            raise SystemExit(
                f"sample {selection.original_id!r} was not found in "
                f"{selection.source_name}"
            ) from error
        _validate_selection(row, selection=selection, dataset_root=dataset_root)
        selected_rows.append(row)

    source_path = output_path.with_suffix(".source.csv")
    _write_source(source_path, selected_rows)
    conversion = convert_dataset(
        adapter_name="jailbreakv-28k",
        source_file=source_path,
        output_path=output_path,
        context=AdapterContext(
            media_root=dataset_root,
            dataset_version="v0.2",
            split=DatasetSplit.TEST,
            subset="balanced-5",
            verify_media=True,
            hash_media=True,
        ),
        overwrite=True,
    )
    print(
        json.dumps(
            {
                "status": "prepared",
                "network_used": False,
                "output": str(conversion.output_path),
                "manifest": str(conversion.manifest_path),
                "samples": conversion.manifest.sample_count,
                "sample_ids": [
                    f"jailbreakv-28k:balanced-5:{item.original_id}"
                    for item in _BALANCED_5
                ],
                "policies": [item.policy for item in _BALANCED_5],
                "attack_formats": [item.attack_format for item in _BALANCED_5],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _load_index(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"required JailBreakV source file does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        actual_fields = set(reader.fieldnames or ())
        missing = sorted(set(_CSV_FIELDS) - actual_fields)
        if missing:
            raise SystemExit(f"{path} is missing required columns: {missing}")
        index: dict[str, dict[str, str]] = {}
        for raw_row in reader:
            row = {key: raw_row.get(key) or "" for key in _CSV_FIELDS}
            original_id = row["id"].strip()
            if original_id in index:
                raise SystemExit(f"duplicate sample ID {original_id!r} in {path}")
            index[original_id] = row
    return index


def _validate_selection(
    row: dict[str, str],
    *,
    selection: Selection,
    dataset_root: Path,
) -> None:
    actual = (row["policy"].strip(), row["format"].strip())
    expected = (selection.policy, selection.attack_format)
    if actual != expected:
        raise SystemExit(
            f"sample {selection.original_id} changed policy/format: "
            f"expected {expected}, got {actual}"
        )
    media_path = (dataset_root / row["image_path"]).resolve()
    try:
        media_path.relative_to(dataset_root)
    except ValueError as error:
        raise SystemExit(
            f"sample {selection.original_id} media escapes dataset root: {media_path}"
        ) from error
    if not media_path.is_file():
        raise SystemExit(
            f"sample {selection.original_id} media does not exist: {media_path}"
        )


def _write_source(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
