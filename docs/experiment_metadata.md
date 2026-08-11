# Canonical experiment metadata

Reviewed 2026-08-11

## Scope

The schema at `schemas/experiment-metadata.schema.json` defines the metadata object shared by canonical raw and derived results. The result envelope at `schemas/result-record.schema.json` pairs this object with a trial or aggregate metric payload.

The metadata object has six required groups in addition to its two version fields.

| Group | Purpose |
| --- | --- |
| `experiment` | Experiment identity, manifest version, and frozen settings |
| `model` | Benchmark model identity, provider endpoint, and generation settings |
| `protocol` | Shared collection, retry, parser, and invalid response rules |
| `run` | Run identity, lifecycle state, attempt number, and UTC timing |
| `provenance` | Code revision, runner, environment, source method, and completeness |

## Version identity

`benchmark_version` and `schema_version` are required and equal `1.0.0` in this schema. Every result record carries this metadata object. A later schema version receives a new schema file. Readers follow the compatibility policy in `docs/versioning.md`.

## Experiment identity

`experiment.id` uses the stable identifier from `config/experiments.json`. `experiment.family` records the elicitation or strategic game family. `experiment.manifest_version` identifies the manifest used for the run.

`experiment.parameters` is an exact snapshot of the experiment `settings` object. This snapshot prevents a later manifest edit from changing the interpretation of an existing result.

## Model identity

`model.id`, `model.provider`, and `model.api_model_id` match one entry in `config/models.json`. The schema validates their representation. Manifest validation must verify that the three values belong to the same model entry.

`model.id` is the semantic identifier and remains unchanged in result contents. Filesystem and URL paths use the derived key defined in `docs/model_identifiers.md`.

Every model parameter is present even when its value is unavailable. Requested and effective temperature are separate. A null effective temperature means that the provider did not expose or report the setting. It does not mean zero.

The same requested and effective distinction applies to reasoning mode. The seed is null when no provider seed is requested. `provider_options` retains provider specific settings that do not belong in the common fields.

## Protocol snapshot

The protocol object copies the relevant shared settings from `config/experiments.json`. It records condition order, the local random seed, every response parser, the retry policy, and the invalid response policy.

Parser names use an array because some experiments have more than one role specific parser. The retry and invalid response objects reject unknown fields so a misspelled policy cannot pass unnoticed.

## Run lifecycle

`run.id` is stable across the raw and derived artifacts produced by one execution. A repeated experiment uses a new identifier and increments `run.attempt` when it belongs to the same collection effort.

Timestamps use UTC and end in `Z`. They contain six fractional second digits. A running record has a null completion time. Completed, failed, and interrupted records require a completion time.

## Provenance completeness

Native runs and migrated legacy records use the same metadata shape. `provenance.capture_method` distinguishes them.

A complete record requires a full forty character Git revision, repository state, runner path, Python version, platform, and UTC start time. Its `missing_fields` array is empty.

An incomplete record names every unavailable provenance field in `missing_fields`. Null values remain null and are never replaced with guessed defaults. A migrated record also lists at least one source path.

Schema validity does not make an incomplete record eligible for release. The release validator will enforce provenance completeness for required cells after migration support is implemented.

## Extension rules

Fixed metadata groups reject unknown fields. Experiment settings and provider options remain open objects because their keys vary by experiment and provider. Adding a common metadata field requires a schema version change under `docs/versioning.md`.

## Example

The non release fixture at `schemas/examples/experiment-metadata.json` shows complete metadata for a Dictator run. Its experiment, model, and protocol values match the versioned manifests.
