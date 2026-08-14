# Independent audit request for EconBench 1.0.0

Requested 2026-08-12

## Mandate

Conduct an independent and adversarial audit of the experimental protocols that will be used for the EconBench 1.0.0 pilot. Determine whether the canonical runner presents each experiment as intended, records the model response faithfully, computes the stated estimands correctly, and preserves comparability across models and monetary treatments.

This request now governs the repeat audit after the failed review dated 2026-08-12. Use `docs/protocol_audit_remediation.md` as an index of claimed corrections. Verify every correction independently and do not treat the remediation record as evidence of passage.

Treat roadmap completion marks and existing test results as claims to verify. Do not assume that prior review established scientific validity. Seek counterexamples and identify hidden assumptions.

Do not modify repository files during the audit. Do not make provider requests or incur model costs. The requested output is an evidence based audit report.

## Pilot decision rule

The pilot may proceed only when every audit account and every experiment receives `PASS`.

Use `FAIL` when a defect could alter a model choice, misrecord a response, change a payoff or information set, confound a treatment comparison, invalidate an estimand, or prevent faithful reproduction.

Use `PASS` only when the implementation and documentation agree and the supporting evidence is sufficient. Nonblocking observations may accompany a pass. Any unresolved issue that requires a scientific or design decision must receive `FAIL` until the decision is made and implemented.

The audit applies only to the exact working tree reviewed. Record the Git revision and dirty status. Any later change to prompts, parsers, condition generation, aggregation, manifests, or schemas invalidates the approval and requires a focused repeat audit.

## Canonical scope

Audit the code path reached through `scripts/run_benchmark.py` without `--fixture`. Follow imported functions to determine the operative implementation. Do not substitute behavior from a legacy standalone class unless the canonical runner invokes it.

At minimum inspect the following files and directories.

- `config/experiments.json`
- `config/models.json`
- `config/release_matrix.json`
- `docs/protocol.md`
- `docs/canonical_pipeline.md`
- `docs/experiment_metrics.md`
- `scripts/run_benchmark.py`
- `scripts/validate_results.py`
- `scripts/estimate_release.py`
- `src/tasks/engine.py`
- `src/tasks/specs.py`
- `src/tasks/response_formats.py`
- Every active task module under `src/tasks`
- `src/results/aggregation.py`
- `src/results/validation.py`
- Schemas and examples under `schemas`
- Tests that claim protocol, parser, metric, and runner coverage

Review uncommitted and untracked files when they are part of the operative working tree.

## Independent reference standard

Compare each experiment with a primary experimental source or an authoritative game theoretic definition. Cite the source used. A deliberate variant need not fail merely because it differs from a canonical version. It must be stated clearly, preserve a coherent game, and support the claimed estimand.

Distinguish actual interactive play from a stated choice or strategy method elicitation. Confirm that the protocol and proposed interpretation use the correct description.

## Audit accounts

### A1 Canonical execution path

Trace every active experiment from the benchmark runner to condition generation, prompt construction, response parsing, trial storage, aggregation, and validation. Confirm that documented settings govern the code that will actually run.

### A2 Complete game specification

Confirm that each prompt states the roles, feasible actions, timing, observability, payoff consequences, tie rules, terminal outcomes, and information available to every relevant player. Confirm that omitted information is irrelevant under the intended game.

### A3 Neutral presentation

Check for behavioral priming, moral labels, normative cues, fixed numerical anchors, leading examples, unsupported predictions about opponents, and provider specific wording effects. Determine whether naming a familiar game creates a material demand characteristic for the claimed estimand.

### A4 Response contract and parsing

Confirm that every prompt teaches one unambiguous response expression and that its parser accepts every feasible expression permitted by the prompt. Confirm that the parser rejects infeasible, contradictory, ambiguous, and unlabeled explanatory responses rather than guessing.

Construct adversarial responses that mention endowments, both actions, percentages, payoff amounts, and reasoning before the final choice. Verify that no contextual number or initial letter becomes the recorded action.

### A5 Treatment comparability

Confirm that the 10 dollar, 100 dollar, and 1000 dollar treatments preserve the intended normalized game. Identify every parameter that changes with monetary level. Verify payoff ratios, action grids, bonuses, penalties, multipliers, and normalizations.

### A6 Condition coverage and call counts

Independently derive the expected conditions, repetitions, validation trials, and diagnostic trials. Reconcile those counts with the generated plans and the stated total of 8,060 calls per model. Check for duplicate or observationally identical conditions.

### A7 Order and label effects

Inspect condition order, action labels, role order, and option order. Determine whether fixed labels or ordering are confounded with the estimand. Evaluate whether existing bidirectional checks or counterbalancing are sufficient.

### A8 Invalid responses and provider failures

Verify feasibility checks, invalid response handling, bounded retries, interruption records, and resume behavior. Confirm that no failure becomes a substantive choice and that invalid observations remain visible without entering metric denominators.

Mock each provider transport and confirm that HTTP attempts equal the attempts recorded by the canonical runtime. Confirm that all provider SDK retry layers are disabled.

### A9 Aggregation and estimands

Recompute representative metrics by hand. Confirm that every aggregate uses the correct roles, conditions, units, denominators, and uncertainty calculation. Verify that names such as cooperation, reciprocity, backward induction consistency, and distance from equilibrium do not overstate what the elicitation identifies.

### A10 Data provenance and reproducibility

Confirm that prompts, raw responses, parser outputs, validity states, model settings, versions, timestamps, run identifiers, code revision, and dirty status survive in canonical records. Confirm that derived data and dashboard projections reproduce from raw records.

Run two fixture experiments in an isolated clean Git repository using the default output paths. Confirm that checkpoint writes do not make the repository dirty. Confirm that resume rejects a changed code revision, a dirty native record, changed experiment parameters, and changed provider SDK versions before a provider request.

### A11 Provider comparability

Confirm that requested and effective temperature, reasoning controls, output limits, system prompts, tools, and unsupported parameters are handled and recorded consistently. Identify provider differences that require an exclusion or limitation.

### A12 Statistical adequacy

Assess whether repetitions and condition grids can support the reported quantities. Pay particular attention to the three response majority at each bisection midpoint, isotonic acceptance thresholds, zero variance, condition level validity thresholds, and comparisons across monetary treatments.

### A13 Documentation fidelity

Confirm that the protocol, manifests, roadmap, commands, and intended manuscript language describe the operative implementation. Identify any claim that treats a hypothetical vignette as realized play or a conditional strategy as an observed interaction.

## Experiment specific review

Report a separate result for every experiment below.

### Independence

Verify lottery probabilities, outcome ordering, Marschak Machina triangle coordinates, axis construction, bisection direction, validation retests, monotonicity checks, transitivity checks, and bidirectional presentation.

### Time preference

Verify payment amounts, front end delays, delay conversion, bisection direction, magnitude comparisons, present bias calculations, consistency retests, and bidirectional presentation.

### Dictator

Verify that the transfer determines both players' payments, the full feasible interval is parsable, and no response example anchors a monetary treatment.

### Ultimatum

Verify proposer and responder payoffs, neutral responder uncertainty, the offer grid, role separation, acceptance parsing, and construction of the minimum acceptable offer statistic.

### Trust Game

Verify sender and receiver final payments, multiplication, retained amounts, receiver information, conditional response design, zero transfer conditions, and reciprocity metrics.

### Stag Hunt

Verify the complete symmetric payoff matrix, simultaneous timing, strategic uncertainty, risk and payoff dominance, action labels, and interpretation of the Stag choice rate.

### Beauty Contest

Verify group size, feasible number grid, target fraction, inclusion of the model's own choice in the average, prize allocation, tie handling, and equilibrium benchmark.

### Centipede Game

Verify player alternation, payoff perspective at every node, terminal payments, the history implied at queried turns, the isolated node elicitation design, and the backward induction metric.

### Public Goods

Verify individual and group payoff formulas, simultaneous information, feasible fractional contributions, the marginal per capita return, and interpretation of the multiplier equal to one treatment.

### Traveller's Dilemma

Verify simultaneous claims, bounds, increments, reward and penalty, proportional scaling, feasible grid validation, normalized claims, and the common reporting scale.

### Matching Pennies

Verify the constant sum payoff structure, both payoff roles, balanced choice order, response balance, 100 repetitions per role and payoff cell, Wilson intervals, and the descriptive interpretation of absolute deviation from one half.

## Required offline checks

Run at least the following commands from the repository root.

```bash
python -m pytest -q
python scripts/validate_protocol.py
python scripts/estimate_release.py --json
python -m compileall -q src scripts tests
git diff --check
git status --short --branch
```

Generate and inspect representative canonical prompts for every role, monetary level, payoff treatment, and queried node. Do not audit only the template strings. Inspect the fully interpolated prompts produced by the condition planner.

## Required report

Begin with one overall decision using `PASS` or `FAIL`.

Include the reviewed Git revision, dirty status, date, reviewer identity, and all commands run. State whether the review included untracked files.

Provide one row for every audit account and every experiment using the following columns.

| Item | Result | Evidence | Reason |
| --- | --- | --- | --- |

Every `PASS` must cite concrete evidence with file paths and line numbers or generated prompt excerpts. Every `FAIL` must include a minimal counterexample, the consequence for the benchmark, and a recommended acceptance test for a correction.

Classify each finding as `BLOCKER`, `MAJOR`, or `MINOR`. Any `BLOCKER` or `MAJOR` finding forces the overall decision to `FAIL`. A `MINOR` finding may accompany `PASS` only when it cannot affect choices, metrics, treatment comparisons, validity, or reproducibility.

End the report with one of the following attestations.

`All audit accounts and all experiments pass. The reviewed snapshot is approved for the one model pilot.`

`The reviewed snapshot is not approved for the pilot. The listed failures must be corrected and re-audited.`
