from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from safejudge.constitution import ConstitutionRegistry
from safejudge.core.errors import ConfigurationError
from safejudge.taxonomy import TaxonomyRegistry

_VALID_TAXONOMY = """
[[taxonomies]]
taxonomy_id = "formal-risk-v1"
taxonomy_version = "1.0"
status = "draft"
standard_id = "STANDARD-001"
standard_version = "2026"
standard_title = "Example standard"
jurisdiction = "CN"

[[taxonomies.sources]]
source_id = "official"
title = "Example standard"
publisher = "Example publisher"
official_url = "https://example.invalid/standard"
accessed_on = 2026-08-14
license_status = "metadata_only"

[[taxonomies.categories]]
category_id = "A"
category_name = "Parent"
standard_clause = "Clause A"
source_id = "official"
source_locator = "Section A"
constitution_ids = ["illegal-enablement-v1"]

[[taxonomies.categories]]
category_id = "A.1"
category_name = "Child"
parent_id = "A"
standard_clause = "Clause A.1"
source_id = "official"
source_locator = "Section A.1"
constitution_ids = ["illegal-enablement-v1"]
"""


class TaxonomyRegistryTests(unittest.TestCase):
    def test_repository_taxonomy_has_confirmed_standard_identity(self) -> None:
        registry = TaxonomyRegistry.load(Path("config/taxonomies"))
        pack = registry.get("gb-t-45654-2025-safejudge-v1", version="1.1")

        self.assertEqual(pack.identity.standard_id, "GB/T 45654-2025")
        self.assertEqual(pack.status.value, "active")
        self.assertEqual(len(pack.categories), 36)
        self.assertEqual(len(pack.selectable_categories), 31)
        self.assertEqual(len(pack.routed_categories), 31)
        registry.validate_constitutions(ConstitutionRegistry.load(Path("config/constitutions")))
        child_counts = {
            parent_id: sum(
                category.parent_id == parent_id for category in pack.selectable_categories
            )
            for parent_id in ("A.1", "A.2", "A.3", "A.4", "A.5")
        }
        self.assertEqual(
            child_counts,
            {"A.1": 8, "A.2": 9, "A.3": 5, "A.4": 7, "A.5": 2},
        )
        by_id = {category.category_id: category for category in pack.categories}
        self.assertEqual(by_id["A.3.c"].category_name, "泄露他人商业秘密")
        self.assertEqual(by_id["A.4.e"].standard_clause, "附录 A.4 e)")
        self.assertIsNotNone(by_id["A.3.c"].operational_definition)
        self.assertTrue(by_id["A.3.c"].inclusion_anchors)
        self.assertTrue(by_id["A.3.c"].exclusion_anchors)

    def test_load_identity_and_constitution_links(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir)
            (source / "taxonomy.toml").write_text(_VALID_TAXONOMY, encoding="utf-8")
            registry = TaxonomyRegistry.load(source)
            pack = registry.get("formal-risk-v1", version="1.0")

            self.assertEqual(pack.identity.standard_id, "STANDARD-001")
            self.assertEqual(len(pack.identity.taxonomy_hash), 64)
            registry.validate_constitutions(ConstitutionRegistry.load(Path("config/constitutions")))

    def test_unknown_source_reference_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir)
            prefix, separator, suffix = _VALID_TAXONOMY.rpartition('source_id = "official"')
            self.assertTrue(separator)
            invalid = f'{prefix}source_id = "missing"{suffix}'
            (source / "taxonomy.toml").write_text(invalid, encoding="utf-8")

            with self.assertRaisesRegex(ConfigurationError, "unknown source"):
                TaxonomyRegistry.load(source)

    def test_active_taxonomy_requires_categories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir)
            taxonomy = """
[[taxonomies]]
taxonomy_id = "empty-active"
taxonomy_version = "1.0"
status = "active"
standard_id = "STANDARD-001"
standard_version = "2026"
standard_title = "Example standard"

[[taxonomies.sources]]
source_id = "official"
title = "Example standard"
publisher = "Example publisher"
official_url = "https://example.invalid/standard"
accessed_on = 2026-08-14
license_status = "metadata_only"
"""
            (source / "taxonomy.toml").write_text(taxonomy, encoding="utf-8")

            with self.assertRaisesRegex(ConfigurationError, "requires at least one"):
                TaxonomyRegistry.load(source)

    def test_routed_category_requires_constitution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir)
            invalid = _VALID_TAXONOMY.replace(
                'constitution_ids = ["illegal-enablement-v1"]\n\n[[taxonomies.categories]]',
                "routing_enabled = true\n\n[[taxonomies.categories]]",
                1,
            )
            (source / "taxonomy.toml").write_text(invalid, encoding="utf-8")

            with self.assertRaisesRegex(ConfigurationError, "requires at least one"):
                TaxonomyRegistry.load(source)

    def test_committed_source_hash_is_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact = root / "docs/standards/sources/STANDARD-001/source.txt"
            artifact.parent.mkdir(parents=True)
            artifact.write_text("licensed source excerpt", encoding="utf-8")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            taxonomy = _VALID_TAXONOMY.replace(
                'official_url = "https://example.invalid/standard"\n'
                "accessed_on = 2026-08-14\n"
                'license_status = "metadata_only"',
                'repository_path = "docs/standards/sources/STANDARD-001/source.txt"\n'
                f'document_sha256 = "{digest}"\n'
                'license_status = "permitted_excerpt"',
            )
            config_dir = root / "config/taxonomies"
            config_dir.mkdir(parents=True)
            (config_dir / "taxonomy.toml").write_text(taxonomy, encoding="utf-8")
            registry = TaxonomyRegistry.load(config_dir)

            self.assertEqual(registry.verify_source_artifacts(root), (artifact.resolve(),))


if __name__ == "__main__":
    unittest.main()
