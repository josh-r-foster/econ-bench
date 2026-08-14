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

The planner schedules 3,045 Independence calls, 2,525 Time calls, and 2,490 strategic game calls. The total is 8,060 calls per model and 137,020 calls for the 17 model cohort. Estimated list price ranges from 46.13 to 348.33 dollars before retries and replacement runs.

## Remaining gate

No pilot call is authorized by this remediation. A clean commit and an independent all pass repeat audit are required. Any later change to a reviewed protocol surface requires another focused audit.
