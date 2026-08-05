"""Content-addressed storage for complete provider responses."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import Protocol

from safejudge.contracts.artifact import ArtifactRef
from safejudge.core.errors import ArtifactError

_SAFE_CATEGORY = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class ArtifactStore(Protocol):
    def write_bytes(
        self,
        *,
        category: str,
        content: bytes,
        content_type: str,
    ) -> ArtifactRef: ...


class FileArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def write_bytes(
        self,
        *,
        category: str,
        content: bytes,
        content_type: str,
    ) -> ArtifactRef:
        if _SAFE_CATEGORY.fullmatch(category) is None:
            raise ArtifactError(f"unsafe artifact category: {category!r}")
        digest = hashlib.sha256(content).hexdigest()
        suffix = ".json" if "json" in content_type.lower() else ".bin"
        relative = Path(category) / f"{digest}{suffix}"
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            if not destination.exists():
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix=f".{digest}.",
                    suffix=".tmp",
                    dir=destination.parent,
                    delete=False,
                ) as stream:
                    temporary = Path(stream.name)
                    stream.write(content)
                os.replace(temporary, destination)
                temporary = None
        except OSError as error:
            raise ArtifactError(f"cannot persist provider artifact: {error}") from error
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return ArtifactRef(
            uri=relative.as_posix(),
            sha256=digest,
            content_type=content_type,
            size_bytes=len(content),
        )


class InMemoryArtifactStore:
    """Non-persistent store for offline tests."""

    def __init__(self) -> None:
        self.contents: dict[str, bytes] = {}

    def write_bytes(
        self,
        *,
        category: str,
        content: bytes,
        content_type: str,
    ) -> ArtifactRef:
        digest = hashlib.sha256(content).hexdigest()
        uri = f"memory://{category}/{digest}"
        self.contents[uri] = content
        return ArtifactRef(
            uri=uri,
            sha256=digest,
            content_type=content_type,
            size_bytes=len(content),
        )
