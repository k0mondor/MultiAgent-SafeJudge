"""Load and validate versioned taxonomy TOML files."""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path

from safejudge.constitution.registry import ConstitutionRegistry
from safejudge.core.errors import ConfigurationError
from safejudge.taxonomy.contracts import TaxonomyPack


class TaxonomyRegistry:
    def __init__(self, packs: tuple[TaxonomyPack, ...], *, source: Path) -> None:
        self.source = source
        self._packs = {
            (pack.taxonomy_id, pack.taxonomy_version): pack for pack in packs
        }
        if len(self._packs) != len(packs):
            raise ConfigurationError(f"duplicate taxonomy ID/version in {source}")

    @classmethod
    def load(cls, path: Path) -> TaxonomyRegistry:
        source = path.resolve()
        if not source.exists():
            raise ConfigurationError(f"taxonomy registry does not exist: {source}")
        try:
            files = tuple(
                sorted((source if source.is_dir() else source.parent).glob("*.toml"))
            )
            packs = tuple(
                TaxonomyPack.model_validate(item)
                for file in files
                for item in tomllib.loads(file.read_text(encoding="utf-8")).get(
                    "taxonomies", []
                )
            )
        except (OSError, tomllib.TOMLDecodeError, ValueError) as error:
            raise ConfigurationError(f"invalid taxonomy registry {source}: {error}") from error
        return cls(packs, source=source)

    def get(self, taxonomy_id: str, *, version: str | None = None) -> TaxonomyPack:
        matches = tuple(
            pack
            for (candidate_id, candidate_version), pack in self._packs.items()
            if candidate_id == taxonomy_id
            and (version is None or candidate_version == version)
        )
        if not matches:
            suffix = f" version {version!r}" if version is not None else ""
            raise ConfigurationError(f"unknown taxonomy {taxonomy_id!r}{suffix}")
        if len(matches) > 1:
            versions = sorted(pack.taxonomy_version for pack in matches)
            raise ConfigurationError(
                f"taxonomy {taxonomy_id!r} is ambiguous; choose one of {versions}"
            )
        return matches[0]

    def validate_constitutions(self, registry: ConstitutionRegistry) -> None:
        """Fail when a category points to an unknown Constitution pack."""

        for pack in self._packs.values():
            for category in pack.categories:
                for constitution_id in category.constitution_ids:
                    try:
                        registry.get(constitution_id)
                    except ConfigurationError as error:
                        raise ConfigurationError(
                            f"taxonomy {pack.taxonomy_id!r} category "
                            f"{category.category_id!r} references unknown constitution "
                            f"{constitution_id!r}"
                        ) from error

    def verify_source_artifacts(self, repository_root: Path) -> tuple[Path, ...]:
        """Verify committed source artifacts without downloading external material."""

        root = repository_root.resolve()
        verified: list[Path] = []
        for pack in self._packs.values():
            for source in pack.sources:
                if source.repository_path is None:
                    continue
                artifact = (root / source.repository_path).resolve()
                try:
                    artifact.relative_to(root)
                except ValueError as error:
                    raise ConfigurationError(
                        f"standard source escapes repository root: {source.repository_path}"
                    ) from error
                if not artifact.is_file():
                    raise ConfigurationError(
                        f"standard source artifact does not exist: {source.repository_path}"
                    )
                actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
                if actual != source.document_sha256:
                    raise ConfigurationError(
                        f"standard source hash mismatch: {source.repository_path}"
                    )
                verified.append(artifact)
        return tuple(verified)
