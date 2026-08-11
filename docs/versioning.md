# EconBench version policy

## Benchmark versions

EconBench benchmark versions use semantic versioning.

A major version changes the meaning or comparability of published measures. Examples include changes to prompts, payoff structures, parsing rules, invalid response treatment, temperature policy, or required repetitions.

A minor version adds an experiment, adds a model cohort, or adds a backward compatible measure without changing existing required cells.

A patch version corrects documentation or metadata without changing model calls or metric values.

The benchmark version is stored as `benchmark_version` in every manifest and result record.

## Schema versions

EconBench schema versions use semantic versioning independently from benchmark versions.

A major schema version removes a field, renames a field, changes a field type, or changes the interpretation of a stored value.

A minor schema version adds an optional field or a backward compatible experiment metric.

A patch schema version clarifies validation or corrects a serialization defect without changing the accepted data model.

The schema version is stored as `schema_version` in every raw and derived result record. Readers must reject unsupported major schema versions. Readers may accept newer minor or patch versions only when unknown fields can be ignored safely.

## Release discipline

Published data are immutable within a benchmark version. A correction that changes a metric requires a new benchmark version and a release note that identifies the affected records.

Generated dashboard data carry the benchmark and schema versions of their canonical inputs. A dashboard build must fail when inputs contain mixed benchmark versions or incompatible schema major versions.
