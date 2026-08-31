"""Opaque, sortable-by-prefix identifiers for persisted entities."""

from __future__ import annotations

from typing import NewType
from uuid import uuid4

RunId = NewType("RunId", str)
SampleId = NewType("SampleId", str)


def new_run_id() -> RunId:
    return RunId(f"run_{uuid4().hex}")


def new_sample_id() -> SampleId:
    return SampleId(f"sample_{uuid4().hex}")
