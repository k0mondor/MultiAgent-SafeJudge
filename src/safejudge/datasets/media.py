"""Safe media path normalization shared by dataset adapters."""

from __future__ import annotations

import mimetypes
from pathlib import Path, PurePosixPath

from safejudge.contracts.dataset import MediaRef, MediaType
from safejudge.core.errors import AdapterError
from safejudge.core.files import sha256_file

_DEFAULT_MIME_TYPES = {
    MediaType.IMAGE: "image/jpeg",
    MediaType.AUDIO: "audio/mpeg",
    MediaType.VIDEO: "video/mp4",
}


def _sha256(path: Path) -> str:
    return sha256_file(path)


def build_media_ref(
    raw_uri: str,
    media_type: MediaType,
    *,
    media_root: Path,
    verify: bool,
    hash_media: bool,
) -> MediaRef:
    normalized = raw_uri.strip().replace("\\", "/")
    relative = PurePosixPath(normalized)
    if not normalized or relative.is_absolute() or ".." in relative.parts:
        raise AdapterError(f"unsafe or empty media path: {raw_uri!r}")

    root = media_root.resolve()
    physical = (root / Path(*relative.parts)).resolve()
    try:
        physical.relative_to(root)
    except ValueError as error:
        raise AdapterError(f"media path escapes media root: {raw_uri!r}") from error

    if verify and not physical.is_file():
        raise AdapterError(f"media file does not exist: {physical}")

    mime_type = mimetypes.guess_type(relative.name)[0] or _DEFAULT_MIME_TYPES[media_type]
    can_inspect = physical.is_file()
    return MediaRef(
        media_type=media_type,
        uri=relative.as_posix(),
        mime_type=mime_type,
        sha256=_sha256(physical) if hash_media and can_inspect else None,
        size_bytes=physical.stat().st_size if can_inspect else None,
    )
