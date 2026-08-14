# EconBench release protocol

## Scope

This protocol governs the first EconBench release. The benchmark version is `1.0.0`. The initial result schema version is `1.0.0`. The pre-pilot protocol was revised on 2026-08-12 and becomes final after pilot approval.

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
| Time | Elicitation | Discounting and front end delay sensitivity |
| Dictator | Strategic game | Proposer transfer share |
| Ultimatum | Strategic game | Stated proposer offer and conditional responder acceptance curve |
| Trust Game | Strategic game | Stated sender transfer and conditional receiver return share |
| Stag Hunt | Strategic game | Payoff dominant action rate |
| Beauty Contest | Strategic game | Guess and distance from dominance benchmarks |
| Centipede Game | Strategic game | Stated pass and take rates at isolated nodes |
| Public Goods | Strategic game | Contribution share |
| Traveller's Dilemma | Strategic game | Claim relative to the feasible interval |
| Matching Pennies | Strategic game | Sampled choice balance by payoff role with uncertainty intervals |

The standalone risk and transitivity placeholders are excluded from version `1.0.0`. The independence task already contains the risky choice structure needed for the first release. The transitivity placeholder has no validated and distinct protocol. A later benchmark version may add either task after its estimand and incremental value are established.

## Sampling policy

All active experiments request temperature `0.5`. This common setting keeps the sampling policy fixed across task families. Repeated observations estimate response variation in strategic games. The elicitation tasks retain their bisection and consistency checks under the same sampling setting.

Some provider endpoints do not expose temperature. The runner must omit the parameter for those endpoints and record a null effective temperature. It must not replace the requested value with another numeric value. This rule makes unsupported sampling controls explicit in the released metadata.

The benchmark fixes the lowest supported reasoning control for each selected endpoint. OpenAI reasoning models use a fixed effort. Claude 4 models use thinking disabled. Gemini 2.5 Flash and Flash Lite use a thinking budget of zero. Gemini 2.5 Pro uses its minimum supported budget of 128. Gemini 3.1 Flash Lite uses the minimal thinking level. Every requested control and the Google safety thresholds are stored in `provider_options`. These endpoint capabilities are not identical and comparisons must retain the recorded control as a model attribute.

Tools and system prompts are disabled. Conditions use a deterministic permutation derived from seed `20260811`, the model identifier, and the experiment identifier. Primary, validation, and bidirectional elicitation sequences share this permutation. Binary labels and presentation order are balanced within the design. The exact order seed is stored with every run.

## Repetition policy

Independence uses 12 grid divisions and 10 bisection iterations. Each midpoint receives three responses. The majority response advances the bisection. Ten percent of completed sequences are selected for consistency checks.

Time uses 10 bisection iterations and three responses at each midpoint. The majority response advances the bisection. Ten percent of completed sequences are selected for consistency checks.

Validation selections use the shared local seed and round down to a whole number of sequences. A validation retest preserves the original option order. A separately named bidirectional sequence reverses that order. Independence uses five monotonicity comparisons, ten transitivity comparisons split evenly across the two axes, and five bidirectional bisection sequences. Time uses five monotonicity comparisons and five bidirectional bisection sequences. Each supplemental bisection uses three responses at each of ten midpoints.

Each strategic game uses 10 repetitions per condition except Matching Pennies. The ultimatum responder role uses 20 repetitions per offer condition. Matching Pennies uses 100 repetitions for every payoff and payoff role cell. Exact condition grids appear in `config/experiments.json` and are part of the frozen protocol.

## Strategic game presentation

Each strategic game prompt defines the feasible actions and one labeled response expression. Monetary choices use fields such as `TRANSFER=<amount>` and `CONTRIBUTION=<amount>`. Categorical choices use fields such as `CHOICE=A`. A parser accepts only the complete expression shown in the prompt. Monetary syntax permits an ungrouped numeral or correctly grouped thousands and at most two decimal places. Beauty Contest requires an integer. Bare values, malformed grouping, explanatory prose, multiple expressions, and contradictory text are invalid.

The Stag Hunt prompt gives all four outcomes for both players and balances the safe action across labels A and B. The Trust Game receiver observes the sender's starting endowment, retained amount, transfer, and multiplied transfer. Ultimatum prompts state only the feasible outcomes and do not predict how the other player responds. Dictator and Ultimatum prompts contain no fixed numerical examples. Dictator uses a neutral allocation description and avoids a familiar game label.

Ultimatum responder choices form a strategy method elicitation over independent conditional vignettes. Trust sender and receiver choices are not paired interactions. Centipede choices are elicited separately at hypothetical nodes and do not form a realized path. The resulting quantities describe stated choices under those conditions. They do not describe observed bargaining, trust, reciprocity, or play through a game tree.

Traveller's Dilemma scales its lower bound, upper bound, reward, penalty, and claim increment in the same proportion. The ten dollar treatment ranges from 0.20 dollars to 10 dollars in increments of 0.10 dollars. The one hundred dollar treatment ranges from 2 dollars to 100 dollars in increments of 1 dollar. The one thousand dollar treatment ranges from 20 dollars to 1000 dollars in increments of 10 dollars.

Beauty Contest ties split the prize equally among the tied winners. Public Goods contributions may use fractional dollar amounts within the feasible interval. With ten players and multiplier one, the marginal per capita return is 0.1. A contribution is privately costly and leaves total group payoff unchanged.

Matching Pennies includes the matching and mismatching payoff roles. The order of Heads and Tails is balanced. Reported deviations from one half are descriptive sample quantities. Every condition reports a prespecified 95 percent Wilson interval and must not be interpreted as an estimated mixed strategy without that uncertainty.

Independence fits a normalized quadratic utility on the probability simplex. Utility of the sure low outcome is zero and utility of the sure high outcome is one. The high outcome axis is linear, which fixes the otherwise observationally equivalent nonlinear normalization. A beta norm no greater than 0.05 is classified as consistent with expected utility. Time selects among exponential, hyperbolic, and quasi hyperbolic fits using the Bayesian information criterion. The present bias diagnostic uses the prespecified minimum difference of two percent of the larger amount.

Ultimatum responder rates are fitted by weighted isotonic regression over offer shares. The reported minimum acceptable offer is the first offer with a fitted acceptance rate of at least one half. The raw monotonicity flag remains descriptive and does not determine whether a threshold is reported.

## Response validity

A valid response must be accepted by the named experiment parser and must lie within the feasible response set. Raw provider text is always retained.

Invalid responses are not replaced by a default action. They are excluded from metric denominators and remain present in raw records with an invalid status and parser detail. A failed bisection step invalidates its entire elicitation sequence.

A model and experiment run passes when its aggregate invalid rate is no greater than five percent and every repeated condition has at least an 80 percent valid rate. A run that fails either threshold must be repeated in full under a new run identifier. The failed run remains stored and is not eligible for release metrics.

Transport retries are limited to two retries with waits of two and four seconds. Only timeouts, connection failures, rate limits, and specified transient server responses are retryable. Provider SDK retries are disabled and the SDK package and version are recorded. Interface contract errors and returned but unparsable completions are not retried.

## Canonical data locations

Canonical raw records are immutable and live under the following pattern.

```text
data/releases/1.0.0/raw/{model_key}/{experiment_id}/{run_id}.jsonl
```

Canonical derived records live under the following pattern.

```text
data/releases/1.0.0/derived/{model_key}/{experiment_id}.json
```

The release manifest lives at `data/releases/1.0.0/manifest.json`. Files under `web/data` are generated dashboard projections. They are not canonical evidence and must be reproducible from released raw records.

`model_key` is the portable path component derived from the semantic model identifier. The conversion follows `docs/model_identifiers.md`. Result contents always retain the original model identifier.

## Release matrix

Every registered model and active experiment pair has one status in `config/release_matrix.json`. Active models have required cells for all 11 experiments. Retired models have excluded cells for all 11 experiments. No optional cells are used in version `1.0.0`.

A required cell must contain a passing canonical run before the release is complete. An excluded cell must not affect completeness statistics. A future optional cell would be reported when present and would not block a release.

## Collection gate

Full release collection must not begin until the runner consumes the manifests, the result schema carries both version fields, invalid responses are no longer silently imputed, and a one model pilot passes validation. Native collection requires a clean Git working tree and rejects an uncommitted snapshot. Existing prototype results remain provisional until they pass the canonical migration and validation work in later phases.

The one model pilot must not begin until the independent review defined in `docs/protocol_audit_request.md` gives every audit account and every experiment a passing result. The reviewed release plan contains 8,060 calls per model and 137,020 calls for the 17 model cohort. Any failed account blocks collection until the correction receives a focused repeat audit.

## Version policy

Benchmark and schema versions follow the policy in `docs/versioning.md`. Every canonical raw and derived record must carry both values.
