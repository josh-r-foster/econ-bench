# Canonical trial records

Reviewed 2026-08-11

## Scope

The schema at `schemas/trial-record.schema.json` defines one trial object for every active experiment. A later result envelope will pair this object with the canonical experiment metadata.

The trial object stores experimental position, complete model interaction text, parser output, validity, transport details, operational errors, and per trial metrics.

## Trial identity

`trial_id` is unique within a run. `sequence_index` is zero based and records execution order. `condition_id` is a stable machine identifier for one configured condition. `condition` stores the exact condition values used for the prompt.

`repetition` is one based. `role` is null for experiments without distinct roles. Role based games use stable values such as `proposer`, `responder`, `sender`, or `receiver`.

## Interaction integrity

Prompt text is always retained in full. A SHA256 digest permits an application validator to detect later mutation. A returned completion is also retained in full with its digest.

JSON Schema validates digest representation but cannot prove that a digest matches its text. The offline application validator must recompute both digests.

Provider request identifiers and finish reasons remain null when the provider does not report them. A provider failure has null response text because no model completion was observed.

## Parser record

The parser object stores the parser name, parser state, parsed value, and diagnostics. `parsed_value` may retain a value even when the parser rejects it for violating the feasible set. This preserves the distinction between extraction failure and protocol failure.

## Trial states

| Validity state | Response text | Parser state | Operational error | Trial metrics |
| --- | --- | --- | --- | --- |
| `valid` | Required | `parsed` | Null | Allowed |
| `invalid_response` | Required | `rejected` | Null | Empty |
| `provider_error` | Null | `not_run` | Required | Empty |
| `interrupted` | Optional | Any parser state | Required | Empty |

An invalid model response is an observation and is never retried. Parser diagnostics explain why it failed. Transport and provider failures are operational events and may consume up to three total attempts under the frozen retry policy.

Invalid and failed trials cannot contain substantive trial metrics. This prevents a default action from being counted as a model choice.

## Timing and usage

Start and completion timestamps use UTC and end in `Z`. Latency is measured in milliseconds. Token counts remain null when the provider does not report usage.

## Error separation

Parser errors live inside the parser object. The top level error object is reserved for transport, provider, protocol, and internal failures. It records a stable category, an optional provider code, a human readable message, retryability, and structured details.

## Experiment metrics

`trial_metrics` is open in the common trial schema because metric keys vary by experiment. `P2.4` will define the allowed object for each active experiment. A valid trial may use an empty object when an experiment has no per trial derived quantity.

## Examples

The `schemas/examples` directory contains valid, invalid response, and provider error examples. These files are fixtures and are not benchmark observations.
