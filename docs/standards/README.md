# Standard provenance policy

SafeJudge stores a versioned taxonomy and enough provenance to reproduce how formal
standard clauses were mapped to Constitution packs. It does not assume that the full text
of a national standard may be redistributed.

## What must be committed

For every active taxonomy, commit a TOML file under `config/taxonomies/` containing:

- the exact `standard_id`, edition/year in `standard_version`, and official title;
- publisher, publication/effective dates, official URL, and access date;
- stable category IDs, formal category names, parent relationships, and clause locators;
- the Constitution pack IDs used to evaluate each category;
- the taxonomy status, version, and any scope notes.

Use `national-standard-v1.toml.example` as the starting point. Never replace unknown
official values with guessed labels.

The repository contains the active metadata-only taxonomy `gb-t-45654-2025.toml` for
GB/T 45654-2025. Its official identity, publication metadata, and Appendix A classification
are recorded as 5 non-selectable parent categories and 31 routing-enabled leaf risks. Each
leaf maps to one of the category-filtered `gbt45654-a1-v1` through `gbt45654-a5-v1`
Constitution packs.

## When a source document may be committed

Only commit a full document or excerpt when its redistribution terms allow it. Place an
allowed artifact under:

```text
docs/standards/sources/<standard-id>/
```

Then record its repository-relative path, SHA-256, and `license_status` in the taxonomy
source entry. `TaxonomyRegistry.verify_source_artifacts()` checks both the path boundary
and content hash.

When redistribution is restricted or unclear, keep `license_status = "metadata_only"` or
`"restricted"`, commit only the official URL and bibliographic metadata, and retain any
local source document outside Git. A document hash may still be recorded if the exact
downloaded edition must be identified.

## Validation

After creating the real taxonomy TOML, validate its Constitution links and any committed
source artifacts before enabling routing:

```python
from pathlib import Path

from safejudge.constitution import ConstitutionRegistry
from safejudge.taxonomy import TaxonomyRegistry

taxonomies = TaxonomyRegistry.load(Path("config/taxonomies"))
taxonomies.validate_constitutions(
    ConstitutionRegistry.load(Path("config/constitutions"))
)
taxonomies.verify_source_artifacts(Path("."))
```

## Runtime integration

Enable the multi-label router on an evaluation batch with:

```powershell
safejudge evaluate run-jsonl `
  <the existing required arguments> `
  --taxonomy gb-t-45654-2025-safejudge-v1 `
  --taxonomy-version 1.0
```

The Category Router sees the frozen target response and may return zero, one, or several
leaf IDs. IDs are rejected unless they are selectable and routing-enabled. Every selected
leaf is bound to its mapped Constitution, compiled with that `category_id`, judged by an
independent compliance/enablement panel, and persisted under `category_results`.
Each panel verdict must also return non-empty `triggered_rule_ids`; the runtime rejects
unknown rule IDs and requires at least one rule specific to the routed leaf category.
`EvaluationSpec` and the batch manifest include the taxonomy identity and hash; each final
result includes the Category Router hash and selected IDs. Zero matches produces
`not_evaluated`; ambiguous grounding produces `review_required`. The router never silently
selects a parent or fallback class. The batch manifest summarizes leaf hit counts,
per-leaf final-level counts, multi-label samples, zero-match samples, and category results
requiring review.
