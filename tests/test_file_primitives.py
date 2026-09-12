from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from safejudge.core.files import (
    atomic_write_bytes,
    atomic_write_text,
    file_digest,
    sha256_bytes,
    sha256_file,
)


class FilePrimitiveTests(unittest.TestCase):
    def test_atomic_writes_replace_content_and_share_digest_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "artifact.jsonl"
            atomic_write_text(path, "first\n")
            atomic_write_bytes(path, b"second\n")

            self.assertEqual(path.read_bytes(), b"second\n")
            self.assertEqual(sha256_file(path), sha256_bytes(b"second\n"))
            self.assertEqual(
                file_digest(path).model_dump(mode="json"),
                {"name": "artifact.jsonl", "sha256": sha256_bytes(b"second\n")},
            )


if __name__ == "__main__":
    unittest.main()
