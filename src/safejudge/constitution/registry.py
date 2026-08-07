"""Load ConstitutionPack files from a versioned TOML registry."""

from __future__ import annotations

import tomllib
from pathlib import Path

from safejudge.constitution.contracts import ConstitutionPack
from safejudge.core.errors import ConfigurationError


class ConstitutionRegistry:
    def __init__(self, packs: tuple[ConstitutionPack, ...], *, source: Path) -> None:
        self.source = source
        self._packs = {pack.constitution_id: pack for pack in packs}
        if len(self._packs) != len(packs):
            raise ConfigurationError(f"duplicate constitution_id in {source}")

    @classmethod
    def load(cls, path: Path) -> ConstitutionRegistry:
        source = path.resolve()
        if not source.exists():
            raise ConfigurationError(f"constitution registry does not exist: {source}")
        try:
            # A file remains a valid entry point, but sibling packs are loaded so
            # versioned `extends` references can be resolved without caller changes.
            files = tuple(
                sorted((source if source.is_dir() else source.parent).glob("*.toml"))
            )
            packs = tuple(
                ConstitutionPack.model_validate(item)
                for file in files
                for item in tomllib.loads(file.read_text(encoding="utf-8")).get("packs", [])
            )
        except (OSError, tomllib.TOMLDecodeError, ValueError) as error:
            raise ConfigurationError(f"invalid constitution registry {source}: {error}") from error
        return cls(packs, source=source)

    def get(self, constitution_id: str) -> ConstitutionPack:
        try:
            return self._resolve(constitution_id, stack=())
        except KeyError as error:
            raise ConfigurationError(f"unknown constitution: {constitution_id!r}") from error

    def _resolve(
        self,
        constitution_id: str,
        *,
        stack: tuple[str, ...],
    ) -> ConstitutionPack:
        if constitution_id in stack:
            raise ConfigurationError(
                f"constitution inheritance cycle: {' -> '.join((*stack, constitution_id))}"
            )
        pack = self._packs[constitution_id]
        if not pack.extends:
            return pack
        bases = tuple(
            self._resolve(base_id, stack=(*stack, constitution_id))
            for base_id in pack.extends
        )
        return ConstitutionPack.model_validate(
            {
                **pack.model_dump(mode="python"),
                "extends": (),
                "core_rules": (
                    *(rule for base in bases for rule in base.rules),
                    *pack.core_rules,
                ),
            }
        )
