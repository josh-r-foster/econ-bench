# Canonical social results

Reviewed 2026-08-11

## Authoritative representation

Dictator and Ultimatum are separate canonical experiments. Each has its own raw JSONL run and derived JSON result. This split representation is authoritative for benchmark version `1.0.0`.

Legacy `social_experiment` files are migration inputs only. They must not be read as canonical evidence after split records exist. The temporary dashboard fallback remains website compatibility code until phase five removes it.

## Migration

`scripts/migrate_legacy_social.py` converts one combined legacy file into split Dictator and Ultimatum records. The command requires an explicit source timezone because legacy timestamps are naive.

```bash
python scripts/migrate_legacy_social.py web/data/social_experiment_gpt-4o.json --source-timezone America/Toronto
```

The migration preserves every stored raw response and reparses it. A stored choice enters canonical metrics only when the raw response supports that choice. An unverifiable stored default becomes an invalid response with empty trial metrics.

Legacy artifacts do not retain original prompt text, token usage, provider request identifiers, finish reasons, original code revision, or complete generation settings. Migrated metadata lists these fields as missing. The prompt field contains a standard unavailable marker so the absence remains explicit. Migrated runs therefore have incomplete provenance and are not release eligible.

## Aggregate preservation

Verified legacy choices retain their published transfer shares, offer shares, acceptance curves, and minimum acceptable offers after conversion from percentages to unit interval shares. Migration tests compare legacy aggregates with canonical recomputation.

A difference caused by a previously imputed invalid response is intentional. Canonical aggregation excludes that response and reports it in the invalid count.

## Output locations

The migration writes one raw run and one derived result for each split experiment under the canonical release root. Existing files are never overwritten unless the caller supplies the explicit overwrite option.

Source paths remain in provenance. Repeating migration over identical source content produces the same run identifier.
