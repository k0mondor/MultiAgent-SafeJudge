from __future__ import annotations

import hashlib
from pathlib import Path

from safejudge.models.artifacts import FileArtifactStore


def test_file_artifact_store_preserves_exact_bytes_by_content_hash(tmp_path: Path) -> None:
    content = b'{"id":"generation_1","answer":"raw answer"}'
    store = FileArtifactStore(tmp_path)

    first = store.write_bytes(
        category="openrouter-responses",
        content=content,
        content_type="application/json",
    )
    second = store.write_bytes(
        category="openrouter-responses",
        content=content,
        content_type="application/json",
    )

    assert first == second
    assert first.sha256 == hashlib.sha256(content).hexdigest()
    assert first.size_bytes == len(content)
    assert (tmp_path / first.uri).read_bytes() == content
