# EconBench implementation roadmap

Last reviewed 2026-08-11

## Purpose

This document defines the work required to turn EconBench from a functional research prototype into a reproducible benchmark release. It records the present baseline, the intended sequence of work, the dependencies between workstreams, the acceptance conditions for each phase, and the procedure for continuing work across sessions.

The roadmap is the durable source of planning context for the repository. Update it whenever scope, protocol, data coverage, or release criteria change.

## Status notation

- `[ ]` means that work has not started
- `[~]` means that work is in progress
- `[x]` means that work is complete and verified
- `[!]` means that work is blocked by an explicit dependency or decision

A work item should be marked complete only after its acceptance checks pass. A code change without verification remains in progress.

## Current baseline

The repository is a functional prototype with an implemented experiment runner, provider wrappers, result files, analysis utilities, and a static website. It is not yet a complete benchmark release.

The local checkout was clean when this roadmap was created. Local `main` was two no-op commits behind `origin/main`. The local and remote trees had identical content. The latest meaningful code change was dated 2026-06-19.

### Implemented components

- [x] Independence experiment
- [x] Time preference experiment
- [x] Dictator game implementation
- [x] Ultimatum game implementation
- [x] Trust game implementation
- [x] Stag Hunt implementation
- [x] Beauty Contest implementation
- [x] Centipede game implementation
- [x] Public Goods implementation
- [x] Traveller's Dilemma implementation
- [x] Matching Pennies implementation
- [x] Batch orchestration for the eleven implemented experiments
- [x] OpenAI provider wrapper
- [x] Anthropic provider wrapper
- [x] Google provider wrapper
- [x] Local Llama 3.1 wrappers
- [x] Local Qwen wrapper
- [x] JSONL trace logging
- [x] Static dashboard and model detail cards
- [x] GitHub Pages deployment workflow
- [x] Initial validation and run quality scripts

### Incomplete or placeholder components

- [ ] Standalone risk task in `src/tasks/risk.py`
- [ ] Standalone transitivity task in `src/tasks/transitivity.py`
- [ ] Benchmark runner in `scripts/run_benchmark.py`
- [ ] Leaderboard updater in `scripts/update_leaderboard.py`
- [ ] Legacy placeholder in `web/data.js`
- [ ] Complete paper in `TeX/manuscript.tex`

The standalone risk and transitivity tasks may be removed instead of implemented if their intended measures are fully covered by the independence experiment. Phase zero requires an explicit decision.

### Registered model coverage

The website registry currently contains 24 models. Coverage in committed dashboard data was measured as follows.

| Result family | Present | Target |
| --- | --- | --- |
| Independence | 12 | 24 |
| Time preferences | 12 | 24 |
| Rationality aggregates | 12 | 24 |
| Legacy combined social results | 12 | 24 |
| Split Dictator results | 0 | 24 |
| Split Ultimatum results | 0 | 24 |
| Trust Game | 1 | 24 |
| Stag Hunt | 3 | 24 |
| Beauty Contest | 19 | 24 |
| Centipede Game | 1 | 24 |
| Public Goods | 0 | 24 |
| Traveller's Dilemma | 1 | 24 |
| Matching Pennies | 0 | 24 |

The target of 24 applies only until the release cohort is reviewed. Models that are no longer available should be retired explicitly and recorded in a manifest. They should not remain as unexplained missing observations.

### Verification baseline

- [x] All 53 Python files compile without syntax errors
- [x] All 77 committed files in `web/data` parse as JSON
- [x] The focused interface suite passes with 31 tests
- [x] The default `pytest` command completes without network access
- [ ] The result validator passes for the release cohort
- [x] A continuous integration workflow runs the offline test suite

The default test command currently collects `test_models.py`. That file makes live Gemini requests during module import. The run can hang or fail during test collection. The focused command `python -m pytest tests -q` passes.

### Known inconsistencies

- The README refers to a missing `social.py` entry point and obsolete command line flags
- The README describes three tasks even though eleven tasks are implemented
- The guide states that all experiments use temperature `0.01`
- Nine game scripts use temperature `0.5`
- The validator requires split Dictator and Ultimatum files
- The dashboard falls back to legacy combined social files
- The validator ignores most cooperation experiments
- Raw results are partly tracked even though `data/results` is ignored
- `web/public/data/models.json` does not match `web/data/models.json`
- Dependencies are unpinned and no package metadata or lock file exists
- Debug utilities contain user-specific absolute paths
- The manuscript contains placeholder text in every substantive section

## Definition of completion

EconBench is complete for its first release when all of the following conditions hold.

- [ ] The release model cohort and experiment matrix are frozen in versioned manifests
- [ ] Every active experiment has a documented protocol and a versioned result schema
- [ ] Every active model has either valid results for every required experiment or a documented exclusion
- [ ] Raw responses and derived metrics can be traced to model settings, prompts, code revision, and run time
- [ ] Default tests and validation pass without network access or API credentials
- [ ] Live provider smoke tests are opt-in and clearly separated from unit tests
- [ ] The dashboard renders every active experiment and handles unavailable data explicitly
- [ ] The README reproduces installation, execution, validation, aggregation, and local website use
- [ ] Dependencies and supported Python versions are declared and reproducible
- [ ] The manuscript contains a complete method, results, and limitations discussion
- [ ] A fresh clone can reproduce derived website data from the released raw results
- [ ] A tagged release contains a data manifest, release notes, and known limitations

## Guiding principles

### Scientific validity

Protocol decisions must precede large model runs. Temperature, repetition count, prompt wording, payoff scaling, response parsing, invalid response handling, and aggregation rules must be fixed before data collection.

### Reproducibility

Every published metric must be reproducible from stored raw results. Generated website files should never be the only surviving evidence for a result.

### Schema stability

Producers, validators, aggregators, and dashboard consumers must share one versioned schema. Compatibility code should be temporary and tested.

### Offline testability

The default test suite must never call an external model provider. Provider calls belong in explicit smoke or integration commands.

### Controlled data collection

Full model runs should begin only after a one-model pilot passes all protocol, schema, and dashboard checks. This avoids expensive invalid runs.

## Phase zero

## Freeze scope and protocol decisions

Phase zero prevents later implementation and data collection from encoding unresolved scientific choices.

### Work items

- [x] `P0.1` Reconcile local `main` with `origin/main` and confirm a clean baseline
- [x] `P0.2` Define the first release model cohort
- [x] `P0.3` Mark unavailable or deprecated models as retired with written reasons
- [x] `P0.4` Define the required experiment matrix for each active model
- [x] `P0.5` Decide whether risk and transitivity remain standalone tasks
- [x] `P0.6` Define the temperature policy for each experiment family
- [x] `P0.7` Define repetition counts and invalid response thresholds
- [x] `P0.8` Define the canonical raw and derived result locations
- [x] `P0.9` Choose the benchmark and schema versioning convention
- [x] `P0.10` Record all decisions in a protocol document under `docs`

The baseline file tree was verified against `origin/main` at tree `728a5acd9b6baa91d307ee27c9c86f625ac9ae16`. The local branch was synchronized with `origin/main` before phase one began.

### Required decisions

The temperature policy requires particular care. Low temperature supports stable elicitation in independence and time tasks. Repeated strategic games may require stochastic sampling to measure choice distributions. The protocol should state the estimand and justify the setting for each family rather than impose one value across all tasks.

The risk and transitivity stubs also require a scope decision. The independence implementation already contains related checks. A duplicate task should be added only if it measures a distinct construct.

### Deliverables

- `docs/protocol.md`
- `config/models.json`
- `config/experiments.json`
- `config/release_matrix.json`
- A written schema version policy

### Acceptance checks

- [x] Every registered model has an active or retired status
- [x] Every active experiment has fixed settings
- [x] Every cell in the release matrix is required, optional, or excluded
- [x] No full data run depends on an unresolved protocol decision

## Phase one

## Establish a reliable quality gate

This phase makes ordinary development safe before schemas or result data are changed.

### Work items

- [x] `P1.1` Move live model checks out of `test_models.py`
- [x] `P1.2` Place live checks under an explicit smoke test command
- [x] `P1.3` Ensure that importing any test module has no side effects
- [x] `P1.4` Add parser tests for every experiment response parser
- [x] `P1.5` Add metric tests with small deterministic fixtures
- [x] `P1.6` Add task configuration and output path tests
- [x] `P1.7` Add provider registry routing tests for all supported prefixes
- [x] `P1.8` Add tests for logger behavior and disabled logging
- [x] `P1.9` Add a GitHub Actions test workflow
- [x] `P1.10` Preserve the Pages deployment workflow as a separate job

### Minimum offline commands

```bash
python -m pytest -q
python scripts/validate_results.py
```

The validator is expected to fail until phase two updates its schema behavior. The test command must pass at the end of phase one.

### Acceptance checks

- [x] `python -m pytest -q` passes without API credentials
- [x] The test command makes no network request
- [x] A failed parser or metric fixture produces a clear assertion
- [x] Continuous integration runs on pushes and pull requests
- [x] Website deployment does not depend on live model tests

## Phase two

## Define and migrate canonical result schemas

This phase removes the mismatch between experiment output, validation, aggregation, and website consumption.

### Work items

- [x] `P2.1` Inventory every current raw and website JSON shape
- [x] `P2.2` Define one schema for experiment metadata
- [x] `P2.3` Define one schema for trial records
- [ ] `P2.4` Define experiment-specific metric objects
- [ ] `P2.5` Include benchmark version and schema version in every result
- [ ] `P2.6` Standardize model identifier sanitization
- [ ] `P2.7` Standardize timestamp and code revision fields
- [ ] `P2.8` Preserve prompt, response, parser result, and validity status
- [ ] `P2.9` Choose one canonical social result representation
- [ ] `P2.10` Migrate legacy combined social results or formally support them
- [ ] `P2.11` Expand the validator to every active experiment
- [ ] `P2.12` Add migration tests and schema fixtures
- [ ] `P2.13` Make aggregators consume canonical raw results
- [ ] `P2.14` Generate dashboard data from canonical results

### Recommended schema fields

- Benchmark version
- Schema version
- Experiment identifier
- Model identifier
- Provider identifier
- Code revision
- Run identifier
- Run timestamp
- Model parameters
- Experiment parameters
- Prompt text or prompt hash
- Raw response
- Parsed response
- Validity flag
- Error information
- Trial metrics
- Aggregate metrics

### Social data migration

One representation must become authoritative. The least disruptive path is to preserve legacy combined files as migration inputs, generate canonical Dictator and Ultimatum outputs, and remove dashboard fallback logic after the migrated files are verified.

### Acceptance checks

- Every sample result validates against an explicit schema
- The validator covers all active experiments
- Legacy social data produces the same published aggregates after migration
- Dashboard files can be regenerated without model calls
- Invalid and partial runs are reported without silent defaults

## Phase three

## Harden experiment implementations

This phase turns the existing scripts into a coherent experimental library.

### Work items

- [ ] `P3.1` Extract shared model loading and response logging behavior
- [ ] `P3.2` Extract shared output directory and file naming behavior
- [ ] `P3.3` Centralize experiment configuration
- [ ] `P3.4` Remove module-level mutable model state where practical
- [ ] `P3.5` Make invalid response handling consistent
- [ ] `P3.6` Add bounded retries with explicit failure records
- [ ] `P3.7` Verify monetary scaling at 10, 100, and 1000 dollar levels
- [ ] `P3.8` Verify all repeated-game metrics against hand-computed fixtures
- [ ] `P3.9` Add uncertainty estimates where repeated trials support them
- [ ] `P3.10` Make every task resumable without duplicating valid trials
- [ ] `P3.11` Make every task accept a common run identifier
- [ ] `P3.12` Replace or remove the empty benchmark runner
- [ ] `P3.13` Replace or remove the empty leaderboard updater
- [ ] `P3.14` Implement or remove the risk stub
- [ ] `P3.15` Implement or remove the transitivity stub

### Per-experiment review checklist

- [ ] Prompt states the role, payoffs, and allowed response
- [ ] Parser accepts intended formats and rejects ambiguous output
- [ ] Configuration matches the protocol document
- [ ] Trial count matches the configured design
- [ ] Payoff scaling is correct
- [ ] Invalid trials remain visible in stored output
- [ ] Aggregate denominators are correct
- [ ] Metrics have clear units
- [ ] Raw and derived files share the same run identifier
- [ ] Unit tests cover edge cases

### Acceptance checks

- Every active task passes its review checklist
- Every metric has a deterministic fixture
- A simulated provider can run the complete batch offline
- Interrupted runs can resume safely
- A failed provider call cannot be mistaken for a substantive choice

## Phase four

## Complete model coverage

Data collection begins only after phases zero through three pass for a pilot model.

### Work items

- [ ] `P4.1` Run one opt-in provider smoke test for each active provider
- [ ] `P4.2` Record unavailable model identifiers before the main run
- [ ] `P4.3` Estimate calls, tokens, time, and cost for the release matrix
- [ ] `P4.4` Run the full experiment matrix for one pilot model
- [ ] `P4.5` Validate the pilot raw results
- [ ] `P4.6` Regenerate pilot dashboard data
- [ ] `P4.7` Review pilot charts and summary statistics
- [ ] `P4.8` Freeze protocols after pilot approval
- [ ] `P4.9` Run the remaining model matrix in resumable batches
- [ ] `P4.10` Check every completed run with the quality script
- [ ] `P4.11` Rerun only invalid or incomplete cells
- [ ] `P4.12` Produce a final coverage report
- [ ] `P4.13` Freeze a release data manifest with checksums

### Run order

1. Provider smoke tests
2. One inexpensive pilot model
3. Core independence and time experiments
4. Dictator and Ultimatum games
5. Trust and cooperation games
6. High-cost models after lower-cost results pass validation
7. Local models after hardware and deterministic environment checks

### Quality rules

- Never overwrite a valid run without preserving its prior run identifier
- Never merge results produced under different protocol versions
- Record rate limits and provider errors separately from invalid model responses
- Inspect unexpected perfect scores and zero variance before accepting them
- Require a written exclusion for every incomplete matrix cell

### Acceptance checks

- Every required matrix cell is valid or explicitly excluded
- Every result passes schema validation
- Every run meets the configured valid response threshold
- Derived metrics reproduce from raw trials
- The final coverage report agrees with the data manifest

## Phase five

## Complete and verify the website

The website should consume generated data and present missing or excluded results honestly.

### Work items

- [ ] `P5.1` Make the main dashboard consume the canonical model manifest
- [ ] `P5.2` Display protocol and data version information
- [ ] `P5.3` Integrate every active experiment into model cards
- [ ] `P5.4` Decide whether Matching Pennies belongs on the main leaderboard
- [ ] `P5.5` Remove temporary legacy social fallback after migration
- [ ] `P5.6` Distinguish missing, excluded, invalid, and unavailable data
- [ ] `P5.7` Verify all per-magnitude and per-condition charts
- [ ] `P5.8` Add accessible labels and non-color status indicators
- [ ] `P5.9` Add browser smoke checks for index and model card pages
- [ ] `P5.10` Remove or explain the stale `web/public` data copy
- [ ] `P5.11` Remove the empty `web/data.js` file if unused
- [ ] `P5.12` Verify the Pages artifact before deployment

### Acceptance checks

- Every active model has a navigable page
- Every valid result renders without console errors
- Missing data never appears as a zero score
- Chart labels match stored units and protocol names
- The deployed site uses the same manifest as local validation
- Browser smoke checks pass against a local static server

## Phase six

## Make the repository reproducible and maintainable

### Work items

- [ ] `P6.1` Rewrite the README around the actual eleven-task architecture
- [ ] `P6.2` Correct all command line flags and examples
- [ ] `P6.3` Add `.env.example` with placeholder variable names
- [ ] `P6.4` Add package metadata in `pyproject.toml`
- [ ] `P6.5` Declare supported Python versions
- [ ] `P6.6` Pin or constrain dependency versions
- [ ] `P6.7` Separate base, hosted provider, and local GPU dependencies
- [ ] `P6.8` Remove hard-coded user paths from debug utilities
- [ ] `P6.9` Document raw data retention and generated artifact policy
- [ ] `P6.10` Resolve the mixed tracked and ignored state of `data/results`
- [ ] `P6.11` Add a contributor workflow for models and experiments
- [ ] `P6.12` Add a release reproduction command
- [ ] `P6.13` Document estimated cost and expected run duration

### Recommended commands

The exact interface should be finalized during implementation. A complete repository should expose commands equivalent to the following operations.

```bash
python -m pytest -q
python scripts/validate_results.py
python src/tasks/run_batch.py --model gpt-4o
python src/tools/calculate_rationality_stats.py
python -m http.server 8000
```

### Acceptance checks

- A fresh environment installs from documented commands
- A new contributor can run an offline simulated batch
- Hosted provider dependencies do not require local GPU packages
- No tracked utility contains a user-specific absolute path
- Raw data and generated artifacts have documented ownership rules

## Phase seven

## Complete the manuscript

The manuscript should describe the released benchmark rather than an earlier or aspirational design.

### Work items

- [ ] `P7.1` Write the abstract after the final results are frozen
- [ ] `P7.2` Write the introduction and research questions
- [ ] `P7.3` Complete the literature review
- [ ] `P7.4` Document model selection and release cohort rules
- [ ] `P7.5` Document every experimental protocol
- [ ] `P7.6` Document parsing, invalid responses, and exclusions
- [ ] `P7.7` Document aggregation and uncertainty methods
- [ ] `P7.8` Generate tables and figures from released data
- [ ] `P7.9` Report coverage and missingness
- [ ] `P7.10` Write the results without relying only on leaderboard ranks
- [ ] `P7.11` Discuss construct validity and provider nondeterminism
- [ ] `P7.12` Discuss model version drift and reproducibility limits
- [ ] `P7.13` Add a data and code availability statement
- [ ] `P7.14` Verify references and citation metadata
- [ ] `P7.15` Render and inspect the final PDF

### Acceptance checks

- No placeholder text remains
- Every reported number traces to a released result file
- Tables and figures are generated rather than transcribed by hand
- The methods match the released protocol version
- Limitations address missingness, sampling, prompt sensitivity, and model drift
- The rendered PDF has no layout warnings that affect readability

## Phase eight

## Release and archive

### Work items

- [ ] `P8.1` Run all offline tests from a fresh clone
- [ ] `P8.2` Validate the complete release matrix
- [ ] `P8.3` Rebuild all derived dashboard files
- [ ] `P8.4` Rebuild manuscript tables and figures
- [ ] `P8.5` Verify the website locally
- [ ] `P8.6` Deploy and verify the Pages site
- [ ] `P8.7` Create release notes
- [ ] `P8.8` Publish the protocol, model manifest, experiment manifest, and data manifest
- [ ] `P8.9` Tag the repository release
- [ ] `P8.10` Record known limitations and deferred work

### Release gate

No release should be tagged while default tests fail, validation reports unexplained missing data, dashboard artifacts cannot be regenerated, or the manuscript reports results from a different protocol version.

## Dependency order

| Phase | Depends on | Reason |
| --- | --- | --- |
| Phase zero | None | Scope and protocol choices govern all later work |
| Phase one | Phase zero decisions where relevant | Tests must encode the intended protocol |
| Phase two | Phases zero and one | Schema migration needs fixed scope and regression protection |
| Phase three | Phases zero through two | Task hardening targets the canonical protocol and schema |
| Phase four | Phases zero through three | Data collection should use stable and validated code |
| Phase five | Phase two and pilot data from phase four | The dashboard should target canonical generated data |
| Phase six | Phases one through three | Documentation must describe the stable interface |
| Phase seven | Frozen phase four data | The paper must report final results |
| Phase eight | All prior phases | The release gate verifies the complete artifact |

## Immediate priority queue

Work should proceed in this order unless a documented blocker changes the sequence.

- [x] Decide the release cohort and experiment matrix
- [x] Decide the temperature and repetition policy
- [x] Separate live provider smoke tests from default test collection
- [x] Add offline task and parser fixtures
- [ ] Define the canonical schema
- [ ] Migrate legacy social data
- [ ] Expand validation across every active experiment
- [ ] Pilot the complete pipeline on one model
- [ ] Complete the release data matrix
- [ ] Finish dashboard integration
- [ ] Rewrite repository documentation
- [ ] Complete the manuscript
- [ ] Perform the release gate

## Deferred work

The following items should remain outside the first release unless phase zero promotes them.

- Additional model providers beyond the current registry
- New experiments without a pre-registered protocol
- A dynamic backend for the static dashboard
- User accounts or hosted model execution
- Real-time leaderboards
- Broad prompt robustness studies beyond the release design

## Session workflow

### At the beginning of a session

1. Read this roadmap and the protocol document
2. Run `git status --short --branch`
3. Review the most recent roadmap change log entry
4. Select the highest priority unblocked item
5. Mark that item in progress
6. State its acceptance checks before editing code

### During a session

1. Keep changes scoped to one work item when practical
2. Add or update tests with implementation changes
3. Do not launch paid model runs unless the relevant phase four gate is satisfied
4. Record protocol or schema decisions immediately
5. Preserve unrelated user changes in the worktree

### At the end of a session

1. Run the acceptance checks for the selected item
2. Update its status in this roadmap
3. Add a dated entry to the change log
4. Record files changed and commands run
5. Record the next recommended item
6. Record any unresolved decision or blocker

### Handoff template

```text
Date
Work item
Outcome
Files changed
Checks run
Check result
Decisions made
Open blockers
Next recommended item
```

## Roadmap change log

| Date | Work item | Outcome | Next item |
| --- | --- | --- | --- |
| 2026-08-11 | Roadmap creation | Recorded repository baseline and ordered implementation plan | Begin phase zero scope and protocol decisions |
| 2026-08-11 | Phase zero | Froze the release cohort, experiment matrix, and protocol | Establish the phase one quality gate |
| 2026-08-11 | Phase one | Added an offline test gate with 86 passing tests and an explicit live smoke command | Begin the canonical result inventory in `P2.1` |
| 2026-08-11 | `P2.1` | Inventoried 178 result files and 32 recursive shape variants | Define the canonical experiment metadata in `P2.2` |
| 2026-08-11 | `P2.2` | Defined one versioned metadata schema for raw and derived results | Define the canonical trial record in `P2.3` |
| 2026-08-11 | `P2.3` | Defined one trial schema with explicit valid and failure states | Define experiment metric objects in `P2.4` |

### 2026-08-11 phase one handoff

Work item

Phase one quality gate

Outcome

The default suite passes offline with 86 tests. Live provider checks now require the explicit `scripts/smoke_models.py` command. The Pages workflow remains separate from the offline test workflow.

Files changed

- Removed `test_models.py`
- Added `scripts/smoke_models.py`
- Added `.github/workflows/tests.yml`
- Added offline parser, metric, task configuration, registry, logger, and smoke harness tests under `tests`
- Updated `tests/test_model_interfaces.py`
- Updated `ROADMAP.md`

Checks run

- `python -m pytest -q`
- `python scripts/validate_protocol.py`
- `python -m compileall -q src scripts tests`
- Imported every `test_*.py` module without output or provider activity
- `python scripts/smoke_models.py --help`
- `python scripts/validate_results.py`

Check result

The first five checks passed. The result validator returned one because canonical schema migration and release coverage remain phase two work.

Decisions made

Provider smoke checks use one generic opt-in command and require each model identifier to be named explicitly. Offline tests deny socket connections and fail with a direct assertion if a test attempts one.

Open blockers

The existing result validator still targets provisional dashboard shapes and reports missing split social files. This is the expected phase two migration blocker.

Next recommended item

Begin `P2.1` and inventory every raw and dashboard JSON shape.

### 2026-08-11 P2.1 handoff

Work item

Result shape inventory

Outcome

The inventory covers 64 raw JSON files, 34 JSONL trace files, 77 main dashboard files, and three duplicate public files. All 178 files parse successfully and map to 32 recursive shape variants.

Files changed

- Added `docs/result_shape_inventory.md`
- Added `scripts/inventory_result_shapes.py`
- Added `tests/test_result_shape_inventory.py`
- Updated `ROADMAP.md`

Checks run

- `python scripts/inventory_result_shapes.py --summary`
- `python scripts/inventory_result_shapes.py web/data web/public/data --summary`
- `python -m pytest tests/test_result_shape_inventory.py -q`
- `python -m pytest -q`
- `python -m compileall -q src scripts tests`
- `python scripts/validate_protocol.py`
- `rg -n '[:;—]' docs/result_shape_inventory.md`

Check result

The complete scan found no invalid or unclassified files. The focused inventory suite passed with three tests and the full offline suite passed with 89 tests. Compilation and protocol validation passed. The prose check found none of the prohibited punctuation.

Decisions made

Recursive shape signatures distinguish field names and JSON value types. Shape identifiers are inventory aids and do not serve as schema versions. Ignored local raw results remain in the workspace scan while the document reports their tracked status separately.

Open blockers

Current artifacts lack the required version and provenance fields. Stored Centipede, Trust, and Traveller files predate their current producers. Split social files, Public Goods files, and Matching Pennies files are absent.

Next recommended item

Begin `P2.2` and define the canonical experiment metadata object.

### 2026-08-11 P2.2 handoff

Work item

Canonical experiment metadata

Outcome

One versioned metadata object now covers version identity, experiment identity, model identity, frozen protocol settings, run lifecycle, and provenance. It supports complete native records and explicitly incomplete legacy migrations.

Files changed

- Added `schemas/experiment-metadata.schema.json`
- Added `schemas/examples/experiment-metadata.json`
- Added `docs/experiment_metadata.md`
- Added `tests/test_experiment_metadata_schema.py`
- Updated `.github/workflows/tests.yml`
- Updated `ROADMAP.md`

Checks run

- `python -m pytest tests/test_experiment_metadata_schema.py -q`
- `python -m json.tool schemas/experiment-metadata.schema.json`
- `python -m json.tool schemas/examples/experiment-metadata.json`
- `rg -n '[:;—]' docs/experiment_metadata.md`
- `python -m pytest -q`
- `python -m compileall -q src scripts tests`
- `python scripts/validate_protocol.py`
- `python scripts/inventory_result_shapes.py --summary`

Check result

The focused metadata suite passed with eleven tests and the full offline suite passed with 100 tests. Compilation, protocol validation, and the result inventory passed. Both JSON files parse and the prose check found none of the prohibited punctuation.

Decisions made

Raw and derived results share the same metadata object. Requested and effective model settings remain separate. Fixed metadata groups reject unknown fields. Experiment settings and provider options remain open objects. Incomplete provenance is valid only when every missing field is named. It does not imply release eligibility.

Open blockers

The schema does not yet define trial records or experiment metric objects. Manifest membership and release eligibility require application validation in later work items.

Next recommended item

Begin `P2.3` and define the canonical trial record.

### 2026-08-11 P2.3 handoff

Work item

Canonical trial records

Outcome

One trial object now represents valid choices, invalid model responses, provider failures, and interrupted work. It preserves full interaction text, integrity hashes, parser output, validity, transport usage, operational errors, condition identity, and per trial metrics.

Files changed

- Added `schemas/trial-record.schema.json`
- Added three trial state examples under `schemas/examples`
- Added `docs/trial_records.md`
- Added `tests/test_trial_record_schema.py`
- Updated `ROADMAP.md`

Checks run

- `python -m pytest tests/test_trial_record_schema.py -q`
- `python -m json.tool schemas/trial-record.schema.json`
- `python -m json.tool schemas/examples/trial-record.valid.json`
- `python -m json.tool schemas/examples/trial-record.invalid-response.json`
- `python -m json.tool schemas/examples/trial-record.provider-error.json`
- `rg -n '[:;—]' docs/trial_records.md`
- `python -m pytest -q`
- `python -m compileall -q src scripts tests`
- `python scripts/validate_protocol.py`
- `python scripts/inventory_result_shapes.py --summary`

Check result

The focused trial suite passed with sixteen tests and the full offline suite passed with 116 tests. Compilation, protocol validation, and the result inventory passed. The schema and examples parse as JSON. The prose check found none of the prohibited punctuation.

Decisions made

Sequence indices are zero based and repetitions are one based. Prompt and response text are stored in full with SHA256 digests. Parser diagnostics remain separate from operational errors. Invalid and failed trials must have empty metric objects. Provider retries may produce at most three total attempts under the frozen protocol.

Open blockers

The common trial schema leaves condition contents and valid trial metric contents open. Experiment specific metric contracts remain `P2.4` work. Application validation must recompute text digests and verify timestamp order, unique trial identifiers, and manifest membership.

Next recommended item

Begin `P2.4` and define experiment specific metric objects.
