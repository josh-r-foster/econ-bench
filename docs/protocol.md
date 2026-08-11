# EconBench release protocol

## Scope

This protocol governs the first EconBench release. The benchmark version is `1.0.0`. The initial result schema version is `1.0.0`. The protocol was frozen on 2026-08-11.

The release studies economic choice by language models under fixed prompts and fixed experimental conditions. It includes two elicitation tasks and nine strategic games. The model cohort, experiment settings, and release matrix are recorded in versioned JSON manifests under `config`.

## Model cohort

The release cohort contains 17 active models. Each active model is represented by a stable EconBench identifier and a pinned or stable provider endpoint. The provider endpoint is the value in `api_model_id` within `config/models.json`.

Preview endpoints are excluded. Retired provider endpoints are excluded. Duplicate identifiers that refer to the same provider snapshot are excluded. These rules prevent the release matrix from counting one model twice or relying on an endpoint that can change during data collection.

The release cohort uses hosted models from OpenAI, Anthropic, and Google. Local model wrappers remain outside the first release cohort because they were not part of the dashboard registry reviewed for this protocol.

## Experiment scope

The active experiments are shown below.

| Experiment | Family | Primary quantity |
| --- | --- | --- |
| Independence | Elicitation | Indifference curve geometry and expected utility deviations |
| Time | Elicitation | Discounting and dynamic consistency |
| Dictator | Strategic game | Proposer transfer share |
| Ultimatum | Strategic game | Proposer offer and responder acceptance threshold |
| Trust Game | Strategic game | Sender transfer and receiver return share |
| Stag Hunt | Strategic game | Payoff dominant action rate |
| Beauty Contest | Strategic game | Guess and distance from dominance benchmarks |
| Centipede Game | Strategic game | Pass and take rates by node |
| Public Goods | Strategic game | Contribution share |
| Traveller's Dilemma | Strategic game | Claim relative to the feasible interval |
| Matching Pennies | Strategic game | Choice balance and distance from the mixed equilibrium |

The standalone risk and transitivity placeholders are excluded from version `1.0.0`. The independence task already contains the risky choice structure needed for the first release. The transitivity placeholder has no validated and distinct protocol. A later benchmark version may add either task after its estimand and incremental value are established.

## Sampling policy

Elicitation tasks request temperature `0.01`. This setting limits sampling noise in bisection paths and consistency checks.

Strategic games request temperature `0.5`. Repeated observations then estimate a choice distribution rather than a single deterministic response.

Some provider endpoints do not expose temperature. The runner must omit the parameter for those endpoints and record a null effective temperature. It must not replace the requested value with another numeric value. This rule makes unsupported sampling controls explicit in the released metadata.

The benchmark uses the provider default reasoning mode and records the effective mode when the provider reports it. Tools and system prompts are disabled. Conditions are presented in manifest order. Local random selection uses seed `20260811`.

## Repetition policy

Independence uses 12 grid divisions and 10 bisection iterations. Ten percent of completed sequences are selected for consistency checks.

Time uses 10 bisection iterations. Ten percent of completed sequences are selected for consistency checks.

Each strategic game uses 10 repetitions per condition. The ultimatum responder role uses 20 repetitions per offer condition. Exact condition grids appear in `config/experiments.json` and are part of the frozen protocol.

## Response validity

A valid response must be accepted by the named experiment parser and must lie within the feasible response set. Raw provider text is always retained.

Invalid responses are not replaced by a default action. They are excluded from metric denominators and remain present in raw records with an invalid status and parser detail. A failed bisection step invalidates its entire elicitation sequence.

A model and experiment run passes when its aggregate invalid rate is no greater than five percent and every repeated condition has at least an 80 percent valid rate. A run that fails either threshold must be repeated in full under a new run identifier. The failed run remains stored and is not eligible for release metrics.

Transport retries are limited to two attempts with waits of two and four seconds. A retry is allowed only before the provider returns a completion. A returned but unparsable completion is an observed invalid response and is not retried.

## Canonical data locations

Canonical raw records are immutable and live under the following pattern.

```text
data/releases/1.0.0/raw/{model_id}/{experiment_id}/{run_id}.jsonl
```

Canonical derived records live under the following pattern.

```text
data/releases/1.0.0/derived/{model_id}/{experiment_id}.json
```

The release manifest lives at `data/releases/1.0.0/manifest.json`. Files under `web/data` are generated dashboard projections. They are not canonical evidence and must be reproducible from released raw records.

## Release matrix

Every registered model and active experiment pair has one status in `config/release_matrix.json`. Active models have required cells for all 11 experiments. Retired models have excluded cells for all 11 experiments. No optional cells are used in version `1.0.0`.

A required cell must contain a passing canonical run before the release is complete. An excluded cell must not affect completeness statistics. A future optional cell would be reported when present and would not block a release.

## Collection gate

Full release collection must not begin until the runner consumes the manifests, the result schema carries both version fields, invalid responses are no longer silently imputed, and a one model pilot passes validation. Existing prototype results remain provisional until they pass the canonical migration and validation work in later phases.

## Version policy

Benchmark and schema versions follow the policy in `docs/versioning.md`. Every canonical raw and derived record must carry both values.
