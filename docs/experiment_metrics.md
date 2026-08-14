# Canonical experiment metrics

Reviewed 2026-08-11

## Scope

The schema at `schemas/experiment-metrics.schema.json` defines trial and aggregate metric objects for all eleven active experiments. The schema uses the experiment identifier and metric level to select one strict contract.

The wrapper is validation context. A canonical trial stores only the inner `metrics` value under `trial_metrics`. A derived result stores the corresponding inner value under `aggregate_metrics`. The application validator supplies the experiment identifier and metric level when it validates either object.

## Units

All rates, proportions, and shares use the closed unit interval. A value of `0.4` means forty percent. Canonical fields never store forty percent as `40`.

Monetary quantities use dollars. Counts use nonnegative integers. Beauty Contest guesses use the configured zero to one hundred scale. Traveller claims retain the dollar amount, a unit interval normalization, and the common two to one hundred reporting scale.

Null denotes a quantity that cannot be estimated. It is appropriate when no valid denominator exists or a model fit fails. Zero is a substantive value and must not represent missing data.

## Sample accounting

Every aggregate contains a sample object. It reports observed trials and the count in every validity state. It also reports the valid response rate and invalid response rate.

The aggregate sample covers the full experiment. Condition entries report their own valid denominators. This arrangement prevents invalid responses from entering a metric denominator without leaving evidence.

## Trial contracts

| Experiment | Required trial metrics |
| --- | --- |
| Independence | Semantic choice between the reference and axis lotteries |
| Time | Semantic choice between the sooner and later payments |
| Dictator | Transfer amount and transfer share |
| Ultimatum | Proposer offer amount and share or responder acceptance |
| Trust Game | Sender amount and share or receiver return amount and rates |
| Stag Hunt | Hare or stag action and payoff dominant choice indicator |
| Beauty Contest | Guess and distance from the Nash benchmark |
| Centipede Game | Pass or take action and backward induction indicator |
| Public Goods | Contribution amount and contribution share |
| Traveller's Dilemma | Claim amount, normalized claim, common scale claim, and lower bound indicator |
| Matching Pennies | Heads or tails choice |

The prompt condition holds treatment values such as pool size, delay, turn, and multiplier. Trial metrics hold choices and quantities derived directly from those choices. This separation avoids repeating the experimental design inside every metric object.

Ultimatum and Trust Game metrics repeat the trial role inside the metric object. This permits strict branch selection. The application validator must confirm that the metric role equals the canonical trial role.

Invalid responses, provider errors, and interrupted trials retain an empty `trial_metrics` object under the common trial schema. They are not passed to the substantive experiment metric schema.

## Aggregate contracts

| Experiment | Required aggregate content |
| --- | --- |
| Independence | Indifference points, parallelism test, quadratic utility fit, validation, and diagnostics |
| Time | Discount estimates, Bayesian information criterion model fits, validation, and diagnostics |
| Dictator | Overall transfer share and pool summaries |
| Ultimatum | Overall proposer offer share, pool summaries, raw and isotonic responder acceptance curves, and an isotonic majority threshold |
| Trust Game | Overall sender and receiver shares plus condition summaries |
| Stag Hunt | Overall stag rate and condition summaries |
| Beauty Contest | Overall guess statistics and prize summaries |
| Centipede Game | Overall action rates, backward induction rate, and turn summaries |
| Public Goods | Overall contribution share and condition summaries |
| Traveller's Dilemma | Overall normalized claim measures, lower bound rate, and condition specific dollar summaries |
| Matching Pennies | Choice rates, absolute deviation from one half, Wilson intervals, and payoff role summaries |

Condition summaries use arrays with explicit condition identifiers. They do not use numeric values as JSON object keys. This gives each cell a stable identity and avoids inconsistent string formatting.

## Application validation

JSON Schema enforces names, types, categories, and numerical bounds. The offline application validator also enforces the following relationships.

- Prompt and response digests reproduce from the retained text
- Run and trial timestamps are ordered
- Validity state counts sum to observed trials
- Reported rates reproduce from their named counts
- Trial metric roles match trial record roles
- Monetary choices lie within the condition bounds
- Shares reproduce from the stored amount and condition denominator
- Trust receipts reproduce from the sender transfer and multiplier
- Traveller claims lie on the scaled action grid
- Counterbalanced Stag labels reproduce the stored semantic action
- Bisection midpoints lie within their recorded bounds
- Bisection steps advance using the majority of three responses
- Aggregate metrics reproduce from valid canonical trials only

These checks are implemented by `src/results/validation.py` and `src/results/aggregation.py`. The schema examples establish the version one metric vocabulary that those tools produce.

## Migration implications

Legacy social files express most rates as percentages. Migration divides those values by one hundred and retains their valid denominators. A missing denominator cannot be invented. It must be marked as incomplete provenance or recovered from the underlying trial list.

Legacy files that imputed a default after parser failure cannot treat the imputed choice as valid canonical evidence. Migration preserves the response as invalid when the raw text permits that determination. A published aggregate derived from imputation remains provisional until a valid canonical run replaces it.

## Examples

The fixture at `schemas/examples/experiment-metrics.json` contains one trial object and one aggregate object for every active experiment. The values illustrate representation and do not report benchmark findings.
