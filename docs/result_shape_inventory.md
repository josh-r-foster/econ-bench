# EconBench result shape inventory

Reviewed 2026-08-11

## Purpose

This document records the result shapes that exist before the canonical schema migration. It covers raw JSON, JSONL traces, dashboard JSON, and the duplicate files under `web/public/data`.

The companion scanner at `scripts/inventory_result_shapes.py` derives a recursive signature from field names and JSON value types. Arrays retain every distinct element shape. The shape identifier is the first twelve characters of a SHA256 digest. It is an inventory aid and is not a schema version.

## Workspace snapshot

The scan covered every JSON and JSONL result file visible in the workspace. Files under `data/results` include ignored local results. A fresh clone contains only five tracked raw JSON files. All 80 website JSON files are tracked.

| Area | JSON files | JSONL files | Tracked JSON files | Shape variants |
| --- | --- | --- | --- | --- |
| Raw experiment results | 64 | 0 | 5 | 10 |
| Model call traces | 0 | 34 | 0 | 5 |
| Main dashboard data | 77 | 0 | 77 | 14 |
| Duplicate public data | 3 | 0 | 3 | 3 |
| Total | 144 | 34 | 85 | 32 |

The 34 trace files contain 859 records. All 178 files parse successfully. Every file is assigned to a known family.

## Raw result families

| Family | Files | Top level shape | Nested record fields |
| --- | --- | --- | --- |
| Independence | 12 | Array of elicitation results | `axis`, `choice_history`, `final_precision`, `indifference_value`, `n_iterations`, `reference_point`, `timestamp` |
| Time | 12 | Array of elicitation results | `annual_rate`, `choice_history`, `delay_days`, `discount_factor`, `front_end_delay`, `indifference_amount`, `larger_amount` |
| Combined social preferences | 13 | Object with three trial arrays | `dictator_proposer`, `ultimatum_proposer`, `ultimatum_responder` |
| Beauty Contest | 21 | Object with `trials` | `decision`, `prize`, `raw_response`, `timestamp`, `trial_number` |
| Stag Hunt | 3 | Object with `trials` | `decision`, `payoff`, `raw_response`, `timestamp`, `trial_number`, `x_multiplier` |
| Centipede Game | 1 | Object with `config` and `trials` | `current_turn`, `current_turn_label`, `decision`, `magnitude`, `raw_response`, `take_payoff_them`, `take_payoff_you`, `timestamp`, `trial_number` |
| Trust Game | 1 | Object with `config` and two trial arrays | Sender and receiver records use role specific amount and rate fields |
| Traveller's Dilemma | 1 | Object with `config` and `trials` | `bonus`, `decision`, `high`, `low`, `magnitude`, `raw_response`, `timestamp`, `trial_number` |
| Model call trace | 34 | One object per JSONL line | `completion_tokens`, `event`, `experiment`, `latency_ms`, `model`, `prompt_chars`, `prompt_tokens`, `response_chars`, `timestamp`, `valid` |

Independence choice records use `choice`, `iteration`, `midpoint`, and `response`. Time choice records use `amount_sooner`, `choice`, `iteration`, and `response`. Neither format stores the prompt.

Combined social trial records use `pool_amount`, `offer_amount`, `offer_percentage`, `raw_response`, `trial_number`, and `timestamp`. Ultimatum responder records also use `decision`.

Trust sender records use `endowment`, `multiplier`, `amount_sent`, `send_rate`, `raw_response`, `trial_number`, and `timestamp`. Trust receiver records add the sent and received amounts plus two return rates.

The trace token fields are optional. Every trace record has `experiment` equal to `unknown`. Traces store response length rather than response text and store prompt length rather than prompt text or a prompt hash.

## Dashboard result families

| Family | Files | Top level fields |
| --- | --- | --- |
| Model registry | 1 | Array of model identifier strings |
| Independence | 12 | `analysis_text`, `results`, `tldr_text` |
| Time | 12 | `analysis_text`, `datasets`, `labels`, `tldr_text` |
| Rationality aggregate | 12 | `metrics`, `model` |
| Legacy combined social | 12 | `analysis_text_dictator`, `analysis_text_ultimatum`, `dictator_proposer`, `model_id`, `timestamp`, `tldr_dictator`, `tldr_ultimatum`, `ultimatum_proposer`, `ultimatum_responder` |
| Social statistics | 3 | `metrics`, `model` |
| Stag Hunt | 3 | `analysis_text`, `model_id`, `timestamp`, `tldr_text`, `trials` |
| Beauty Contest | 19 | `analysis_text`, `metrics`, `model_id`, `timestamp`, `tldr_text`, `trials` |
| Centipede Game | 1 | `analysis_text`, `metrics`, `model_id`, `timestamp`, `tldr_text`, `trials` |
| Trust Game | 1 | `analysis_text`, `metrics`, `model_id`, `receiver_trials`, `sender_trials`, `timestamp`, `tldr_text` |
| Traveller's Dilemma | 1 | `analysis_text`, `metrics`, `model_id`, `timestamp`, `tldr_text`, `trials` |

Rationality metrics contain `patience`, `penalties`, and `risk`. The index page reads the discount factor, magnitude effect, and risk error rate. Social statistic metrics contain `altruism`, `fairness`, `responder`, and `pro_social_score`. No current page reads the social statistic files.

The index and card pages first request split Dictator and Ultimatum files. They fall back to the legacy combined social files when either split file is absent. No split files exist.

No dashboard files exist for Public Goods or Matching Pennies. No raw files exist for split Dictator, split Ultimatum, Public Goods, or Matching Pennies.

## Shape variant register

Every discovered file maps to exactly one row below. The scanner output provides the full path list and recursive shape for each row.

| Area | Family | Shape identifier | Files | Representative file |
| --- | --- | --- | --- | --- |
| Dashboard | Beauty Contest | `b7b78387adc5` | 19 | `web/data/beauty_contest_experiment_gpt-4o.json` |
| Dashboard | Centipede Game | `7edc69bb3b50` | 1 | `web/data/centipede_game_experiment_gpt-4o-mini.json` |
| Dashboard | Independence | `367acad077e8` | 12 | `web/data/independence_results_gpt-4o.json` |
| Dashboard | Model registry | `c484612d4574` | 1 | `web/data/models.json` |
| Dashboard | Rationality | `7fac57ca92cf` | 1 | `web/data/gpt-5.2_rationality.json` |
| Dashboard | Rationality | `df633cd67730` | 11 | `web/data/gpt-4o_rationality.json` |
| Dashboard | Legacy social | `0720a56499a0` | 2 | `web/data/social_experiment_gemini-2.5-flash.json` |
| Dashboard | Legacy social | `8d7eb57fd656` | 9 | `web/data/social_experiment_gpt-4o.json` |
| Dashboard | Legacy social | `a26ac8e29275` | 1 | `web/data/social_experiment_gemini-2.0-flash.json` |
| Dashboard | Social statistics | `34c367b09bf6` | 3 | `web/data/gpt-4o_social_stats.json` |
| Dashboard | Stag Hunt | `91558c1549c9` | 3 | `web/data/stag_hunt_experiment_gpt-4o.json` |
| Dashboard | Time | `431207ac64fc` | 12 | `web/data/time_experiment_gpt-4o.json` |
| Dashboard | Traveller's Dilemma | `6a748b0d91c7` | 1 | `web/data/travellers_dilemma_experiment_gpt-4o-mini.json` |
| Dashboard | Trust Game | `67fd6673a6e2` | 1 | `web/data/trust_game_experiment_gpt-4o-mini.json` |
| Public copy | Independence | `2ccfb91f6a7a` | 1 | `web/public/data/independence_results_o3.json` |
| Public copy | Model registry | `c484612d4574` | 1 | `web/public/data/models.json` |
| Public copy | Time | `80ffbd6575de` | 1 | `web/public/data/time_experiment_o3.json` |
| Raw | Beauty Contest | `19e6c7fc5166` | 21 | `data/results/beauty_contest/gpt-4o/results.json` |
| Raw | Centipede Game | `ad493c4ce8db` | 1 | `data/results/centipede_game/gpt-4o-mini/results.json` |
| Raw | Independence | `ab678bb42502` | 12 | `data/results/independence/gpt-4o/mm_triangle_results.json` |
| Raw | Combined social | `148c30be0735` | 2 | `data/results/social_preferences/gemini-2.5-flash/results.json` |
| Raw | Combined social | `2bc66da795af` | 10 | `data/results/social_preferences/gpt-4o/results.json` |
| Raw | Combined social | `d727acf6569b` | 1 | `data/results/social_preferences/gemini-2.0-flash/results.json` |
| Raw | Stag Hunt | `06cb1a1841a8` | 3 | `data/results/stag_hunt/gpt-4o/results.json` |
| Raw | Time | `3f1314b0f5ff` | 12 | `data/results/time/gpt-4o/discount_rate_results.json` |
| Raw | Traveller's Dilemma | `73d2ee7d4a09` | 1 | `data/results/travellers_dilemma/gpt-4o-mini/results.json` |
| Raw | Trust Game | `e0a2e81ce66e` | 1 | `data/results/trust_game/gpt-4o-mini/results.json` |
| Trace | Model calls | `480788d1d3a8` | 3 | `data/results/runs/session_20260511T140328Z.jsonl` |
| Trace | Model calls | `8dbdb13fef49` | 1 | `data/results/runs/session_20260509T134317Z.jsonl` |
| Trace | Model calls | `91855872770c` | 8 | `data/results/runs/session_20260508T181552Z.jsonl` |
| Trace | Model calls | `992c70899a6a` | 20 | `data/results/runs/session_20260508T162351Z.jsonl` |
| Trace | Model calls | `e1aa9e051856` | 2 | `data/results/runs/session_20260509T134514Z.jsonl` |

The rationality variants differ only because one magnitude effect is encoded as a JSON number while the others are integers. The social variants likewise reflect integer and number encodings for offer amounts. The trace variants reflect optional token fields and integer or number encodings for latency. These differences do not represent distinct economic records.

## Producer and consumer map

| Artifact | Producer | Consumer |
| --- | --- | --- |
| JSONL model call traces | `src/models/logger.py` | `scripts/check_run.py` |
| Independence raw JSON | `src/tasks/independence.py` | `src/tools/calculate_rationality_stats.py` and `scripts/check_run.py` |
| Time raw JSON | `src/tasks/time.py` | `src/tools/calculate_rationality_stats.py` and `scripts/check_run.py` |
| Current strategic game raw JSON | Individual task modules | No aggregate consumer |
| Rationality dashboard JSON | `src/tools/calculate_rationality_stats.py` | `web/index.html` |
| Current experiment dashboard JSON | Individual task modules | `web/index.html` and `web/card.html` |
| Legacy combined social JSON | Missing legacy producer | Fallback logic in `web/index.html` and `web/card.html` |
| Social statistic JSON | Missing legacy producer | No current consumer |
| Model registry array | Several task modules and the rationality tool | `web/index.html` and `web/card.html` |
| Public data copies | Unknown manual copy | No current page fetches these paths |

## Producer drift

The files on disk do not always match the current task code.

- Current Centipede output adds `monetary_level`, final payoff fields, expanded configuration, and per level metrics. The sole stored file predates those fields.
- Current Trust output adds per endowment metric objects. The sole stored dashboard file has only two overall metrics.
- Current Traveller output adds `monetary_level`, `relative_claim`, `claim_100_scale`, and expanded metrics. The sole stored files predate those fields.
- Current Dictator and Ultimatum modules write split raw and dashboard objects. No files from those producers exist.
- Current Public Goods and Matching Pennies modules define raw and dashboard objects. No files from those producers exist.
- Independence and Time files contain manually added `tldr_text`. Their current producers do not write that field.

## Duplicate public data

The main registry contains 24 model identifiers. The public copy contains only `o3`. The public Independence and Time records have the same numeric result content as the main `o3` files but omit `tldr_text`.

## Migration findings

No current result carries `benchmark_version` or `schema_version`. No result consistently carries provider identity, code revision, run identifier, model parameters, experiment parameters, prompt text or hash, parser output, validity status, or structured error information.

Raw and dashboard artifacts cannot be joined by a run identifier. Strategic game source code truncates raw responses to 200 characters. Several experiment runners replace invalid responses with substantive defaults. The stored data therefore cannot distinguish all model choices from parser failures.

The canonical migration must replace file family specific top level shapes with one envelope, one trial record contract, and experiment specific metric objects. Legacy combined social records should be migration inputs. Dashboard JSON should become a projection generated from canonical derived results.

## Verification

Run the complete workspace scan with the following command.

```bash
python scripts/inventory_result_shapes.py --summary
```

Run the committed dashboard scan with the following command.

```bash
python scripts/inventory_result_shapes.py web/data web/public/data --summary
```

The command returns a failure when a file contains invalid JSON or cannot be assigned to a known result family.
