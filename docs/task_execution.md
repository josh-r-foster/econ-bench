# Canonical task execution

## Execution path

All active task commands use the shared engine in `src/tasks/engine.py`. The engine reads the model and experiment manifests before loading a provider. The stable benchmark model identifier remains in result metadata. The provider receives the pinned endpoint recorded in the model manifest.

Each task module retains its prompt builders, response parsers, and analysis helpers. The task command delegates collection to the shared engine. Strategic game helpers receive a model interface through their constructor and do not rely on mutable module state. The older adaptive elicitation analysis classes remain available for compatibility. Their command path does not run them.

## Results

The shared path resolver in `src/tasks/config.py` constructs every canonical raw and derived location. A run identifier is required by the engine and is shared by the raw trial records and the derived aggregate. Every observed trial is checkpointed before the next request begins.

Resume loads the existing raw file for the same model, experiment, and run identifier. Valid, invalid, and provider failure records remain unchanged. An interrupted record is replaced when collection resumes. A valid trial is never requested twice.

## Failures

Provider exceptions receive at most two retries after the initial request. The waits follow the frozen two and four second schedule. Exhausted calls become provider error trials with empty metric objects.

A returned completion is parsed once. An ambiguous or infeasible completion becomes an invalid response trial. It is not retried and is never replaced with a default choice. A failed bisection step prevents an estimate for its full sequence.

## Metrics

Canonical aggregation excludes every nonvalid trial from substantive denominators. Trial counts retain all validity states. Repeated strategic estimates include a standard error for the primary pooled estimand. Valid and invalid response rates include binomial standard errors.

## Offline execution

The benchmark runner provides a deterministic fixture provider. It exercises the same prompts, parsers, checkpoints, schemas, aggregation, and validation as a hosted run. It makes no network request.
