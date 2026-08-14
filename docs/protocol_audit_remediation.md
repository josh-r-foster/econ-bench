# Remediation record for the independent audits

## Status

The first audit and the repeat audit returned `FAIL`. The pilot remains blocked. The corrected working tree must receive an all pass review under `docs/protocol_audit_request.md`.

## Repeat audit corrections

| Account | Correction | Evidence |
| --- | --- | --- |
| A4 | Amount parsers require valid thousands grouping and at most two decimal places. Beauty Contest requires an integer | `src/tasks/response_formats.py`, `tests/test_repeat_audit.py` |
| A7 | Validation retests preserve option order. Only bidirectional checks reverse it. Independence transitivity checks cover both axes | `src/tasks/specs.py`, `tests/test_repeat_audit.py` |
| A8 | OpenAI, Anthropic, and Google SDK retry layers are disabled. The canonical runtime owns all retries | `src/models`, `src/tasks/runtime.py`, `tests/test_repeat_audit.py` |
| A9 | Independence fits normalized quadratic utility. Time selects models by the Bayesian information criterion. Traveller dollar means remain condition specific | `src/results/aggregation.py`, `tests/test_repeat_audit.py` |
| A10 | Release outputs are ignored during collection. Native dirty records and stale experiment settings are invalid. Resume checks the revision and immutable metadata before provider access | `.gitignore`, `src/tasks/engine.py`, `src/results/validation.py`, `tests/test_repeat_audit.py` |
| A11 | Direct dependencies are pinned. Provider SDK package names, versions, and disabled retries are recorded | `requirements.txt`, `src/models/inference_controls.py`, `tests/test_provider_parameters.py` |
| A12 | Every adaptive midpoint receives three responses and advances by majority choice. Ultimatum thresholds use weighted isotonic regression | `config/experiments.json`, `src/tasks/specs.py`, `src/results/aggregation.py`, `tests/test_repeat_audit.py` |
| A13 | The public README uses only the canonical runner and states the interpretation limits. Metric and protocol documents describe the operative estimators | `README.md`, `docs/protocol.md`, `docs/experiment_metrics.md` |

## Corrected workload

When every elicitation sequence brackets and completes, the planner schedules 2,787 Independence calls, 3,029 Time calls, and 2,490 strategic game calls. This maximum is 8,306 calls per model and 132,896 calls for the 16 model cohort. Censored sequences stop early. The estimate excludes retries and replacement runs.

## Final audit corrections

| Finding | Correction | Evidence |
| --- | --- | --- |
| Release integrity | The release validator rejects fixture capture. Canonical validation reconstructs the complete trial plan and rejects missing or extra trials | `scripts/validate_results.py`, `src/results/validation.py`, `tests/test_final_audit_remediation.py` |
| Prompt binding | Every canonical prompt, condition, parser result, semantic choice, and adaptive transition reproduces from the manifest derived plan | `src/tasks/specs.py`, `src/results/validation.py`, `tests/test_final_audit_remediation.py` |
| Independence bracketing | Axis references are excluded. The chosen axis follows an observed comparison with the sure middle outcome. A second endpoint comparison must reverse before bisection | `src/tasks/specs.py`, `src/results/aggregation.py`, `tests/test_final_audit_remediation.py` |
| Time censoring | Both payment endpoints are tested before bisection. Noncrossing sequences are reported as censored and excluded from finite rate fits | `src/tasks/specs.py`, `src/results/aggregation.py`, `schemas/experiment-metrics.schema.json` |
| Display fidelity | Stored probability and payment treatments equal the values shown by the canonical prompts | `src/tasks/specs.py`, `tests/test_final_audit_remediation.py` |
| Provider retries | The outer retry policy recognizes real HTTPX transport errors and Google server errors | `src/tasks/runtime.py`, `tests/test_final_audit_remediation.py` |
| Model lifecycle | The deprecated o3 snapshot is retired and excluded from every release cell. The GPT-4o 2024-11-20 snapshot remains active because the provider deprecation ledger lists only the older 2024-05-13 snapshot | `config/models.json`, `config/release_matrix.json`, `config/model_availability.json` |
| Interpretation | The strategic estimands describe stated choices. Beauty Contest reports distance from the zero Nash benchmark | `config/experiments.json`, `docs/protocol.md` |

## Remaining gate

No pilot call is authorized by this remediation. A clean commit and an independent all pass repeat audit are required. Any later change to a reviewed protocol surface requires another focused audit.
