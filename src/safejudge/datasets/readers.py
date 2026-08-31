"""Small local readers used by adapters; no network or code execution."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from safejudge.core.errors import AdapterError


def read_json_object(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AdapterError(f"cannot read JSON object from {path}: {error}") from error
    if not isinstance(payload, dict):
        raise AdapterError(f"expected a JSON object in {path}")
    return payload


def iter_mapping_records(path: Path) -> Iterator[Mapping[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        try:
            with path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise AdapterError(f"expected object at {path}:{line_number}")
                    yield value
        except (OSError, json.JSONDecodeError) as error:
            raise AdapterError(f"cannot read JSONL from {path}: {error}") from error
        return

    if suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AdapterError(f"cannot read JSON from {path}: {error}") from error
        if isinstance(payload, list):
            for index, value in enumerate(payload):
                if not isinstance(value, dict):
                    raise AdapterError(f"expected object at {path} list index {index}")
                yield value
            return
        if isinstance(payload, dict):
            yield from payload.values()
            return
        raise AdapterError(f"expected a JSON list or object in {path}")

    if suffix == ".csv":
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                yield from csv.DictReader(stream)
        except OSError as error:
            raise AdapterError(f"cannot read CSV from {path}: {error}") from error
        return

    raise AdapterError(f"unsupported metadata format {suffix!r}: {path}")
