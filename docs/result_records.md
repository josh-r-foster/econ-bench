# Canonical result records

Reviewed 2026-08-11

## Scope

The schema at `schemas/result-record.schema.json` composes canonical metadata with either one trial or one aggregate metric object. It is the outer contract for raw and derived results in schema version `1.0.0`.

Every record contains `record_type`, `metadata`, `trial`, and `aggregate_metrics`. The record type selects exactly one substantive payload. A trial record requires a trial and a null aggregate value. An aggregate record requires aggregate metrics and a null trial value.

## Version placement

Every result carries `benchmark_version` and `schema_version` inside its metadata object. Both values are required by the referenced metadata schema and equal `1.0.0` for this result schema.

Version identity is part of each independently parsed record. It does not depend on a file name, directory name, JSONL header, or neighboring record. A reader can therefore reject an incompatible record before it interprets its payload.

Paths use the model key defined in `docs/model_identifiers.md`. Record metadata retains the semantic model identifier.

The result schema references the versioned identifiers of the metadata and trial schemas. A new incompatible component schema requires a new result schema identifier.

## Raw serialization

The canonical raw path contains JSONL. Each line is one complete result record with `record_type` equal to `trial`. The metadata object repeats on every line.

Repeated metadata increases file size modestly but preserves line independence. A partial file remains interpretable after interruption. A line copied into an error report retains its model, protocol, run, provenance, and version identity.

The raw file has no special header record. This keeps one schema valid for every line and avoids an ordering rule that ordinary JSONL tools cannot enforce.

## Derived serialization

The canonical derived path contains one JSON object with `record_type` equal to `aggregate`. Its metadata identifies the same released run as the raw trial records. Its aggregate metrics reproduce from valid trials in that run.

The result envelope confirms that aggregate metrics are present. The experiment metric schema validates their experiment specific content after the application supplies `metadata.experiment.id` as validation context.

## Composition rules

The application validator performs the following checks in order.

1. Validate the outer result record
2. Read the experiment identifier from metadata
3. Validate substantive metrics against the matching experiment contract
4. Verify relationships that span metadata, trials, and aggregate metrics

A valid trial passes the experiment metric contract. An invalid response, provider error, or interrupted trial has empty trial metrics under the common trial schema and skips substantive metric validation.

The application validator must also confirm that all records in one raw file share the same metadata and that the derived record matches those raw records. It must reject mixed benchmark versions and incompatible schema versions.

The implementation and offline commands are documented in `docs/canonical_pipeline.md`.

## Legacy inputs

Existing prototype files under `data/results` and `web/data` do not become canonical merely because a migration process reads them. They remain unversioned legacy inputs until migration creates result records that pass this schema and names every unavailable provenance field.

The benchmark and schema versions must never be added to a legacy file as an unsupported certification. Migration records the source paths and any incomplete provenance in canonical metadata.

## Examples

The files `schemas/examples/result-record.trial.json` and `schemas/examples/result-record.aggregate.json` are non release fixtures. They reuse the canonical Dictator metadata, trial, and aggregate metric examples.
