# Website presentation review

Reviewed 2026-10-21 against the live `https://econbench.ai` site and the local static source.

## Findings and implemented cleanup

| Finding | Cleanup |
| --- | --- |
| Missing Stag Hunt results appeared as 0% and contributed to “competitive” labels. | Each experiment now has its own table; only observed choices enter rates. Missing observations do not become zero. |
| Sparse summary tables were filled with N/A cells; Trust and Traveller’s Dilemma observations were overlooked. | Tables follow actual experiment coverage, including the published Trust data and normalized Traveller’s Dilemma claims. |
| Two-way and four-way prosocial averages appeared in the same ranking. Patience and risk neutrality were presented as overall rationality. | Show descriptive experiment quantities and sample sizes; methodology explains the removed composites and their limits. |
| Model cards showed empty charts, permanent status messages, and “Analysis loading…” text. | Independent sections appear only with data; absent narratives are hidden. |
| Switching models could retain charts and narratives from the previous model. | Destroy charts and clear narrative state before loading; ignore late results from an older selection. |
| The selector included a model with no result files. | A generated publication catalog excludes empty models while preserving the full registry. |
| Hundreds of speculative requests targeted nonexistent result files. | Fetch only indexed files; catalog freshness is checked before deployment. |
| Prototype data appeared as finished benchmark evidence. | Visible provisional status, coverage, sample counts, source links, and protocol interpretation limits. |
| Wide tables and side-by-side charts were awkward on phones. | Tables scroll within their containers; paired charts stack; selectors and navigation wrap. |

## Current published prototype coverage

The website registry has **24 identifiers**, of which **23** have result files. It contains **73 model–experiment pairs** across **9 of 11 experiment types**. No model covers all eleven experiments. These counts describe historical prototype observations, not completion of the canonical 16-model release cohort.

| Experiment | Models with observations |
| --- | ---: |
| Time | 12 |
| Independence | 12 |
| Dictator | 12 |
| Ultimatum | 12 |
| Trust | 1 |
| Stag Hunt | 3 |
| Beauty Contest | 19 |
| Centipede | 1 |
| Public Goods | 0 |
| Traveller’s Dilemma | 1 |
| Matching Pennies | 0 |

### Coverage by model

| Model | Published experiments |
| --- | --- |
| `gpt-4o` | Time, Independence, Dictator, Ultimatum, Stag Hunt, Beauty Contest |
| `gpt-5.2` | Time, Independence, Dictator, Ultimatum, Beauty Contest |
| `claude-sonnet-4-5` | Time, Independence, Dictator, Ultimatum |
| `o3` | Time, Independence, Dictator, Ultimatum, Beauty Contest |
| `claude-haiku-4-5` | Time, Independence, Dictator, Ultimatum |
| `claude-opus-4-5` | Time, Independence, Dictator, Ultimatum |
| `gpt-5-mini` | Time, Independence, Dictator, Ultimatum, Beauty Contest |
| `gpt-5-nano` | Time, Independence, Dictator, Ultimatum, Beauty Contest |
| `gpt-5` | Time, Independence, Dictator, Ultimatum, Beauty Contest |
| `gemini-2.0-flash` | Time, Independence, Dictator, Ultimatum |
| `gemini-2.5-flash` | Time, Independence, Dictator, Ultimatum, Beauty Contest |
| `gemini-3-flash-preview` | Time, Independence, Dictator, Ultimatum, Stag Hunt, Beauty Contest |
| `gpt-4o-mini` | Trust, Stag Hunt, Beauty Contest, Centipede, Traveller’s Dilemma |
| `gpt-5.2-pro` | Beauty Contest |
| `claude-3-5-sonnet` | No observations published; excluded from selector |
| `claude-sonnet-4-6` | Beauty Contest |
| `claude-haiku-4-5-20251001` | Beauty Contest |
| `claude-sonnet-4-5-20250929` | Beauty Contest |
| `claude-opus-4-7` | Beauty Contest |
| `gemini-2.5-pro` | Beauty Contest |
| `gemini-2.5-flash-lite` | Beauty Contest |
| `gemini-3.1-pro-preview` | Beauty Contest |
| `gemini-3.1-flash-lite` | Beauty Contest |
| `gemini-3-pro-preview` | Beauty Contest |

## Remaining work before a complete benchmark release

1. Collect and validate the canonical active release matrix using `config/models.json`, `config/experiments.json`, and `config/release_matrix.json`. Do not treat the historical 24-identifier website registry as the active collection cohort.
2. The local canonical validator currently reports **176 required cells missing** (16 active models × 11 experiments). Publish passing canonical raw and derived records, reproduce dashboard projections, and update the public status when that evidence exists. The presentation cleanup does not establish validity of legacy responses or remove past imputation from prototype files.
3. Public Goods and Matching Pennies have no published observations. Trust, Centipede, and Traveller’s Dilemma each have one model, with small sample sizes (Trust: 3 sender and 18 receiver observations; Centipede: 9 choices; Traveller’s Dilemma: 3 claims). Complete collection is required for meaningful broad comparisons.
4. Audit older model-specific commentary against the canonical observations. It is preserved as exploratory prototype commentary; some narratives infer motives or reasoning levels beyond what observed choices establish.
5. Before publishing Matching Pennies, expose condition-level payoff roles and Wilson intervals in the public visualization. The existing legacy chart is not sufficient evidence for a mixed-strategy claim.
6. Review the obsolete copies under `web/public/data` before retiring them; the current pages use only `web/data` through `web/catalog.json`. The repository also has an empty `LICENSE`; choose the intended license before advertising licensing terms.

No provider experiments were run, result values were not altered, and no changes were deployed.

## Verification

- Browser-tested every one of the 23 published models and switching between them; all available charts rendered with matching source links and no console errors.
- Checked desktop and 390px mobile layouts, direct experiment anchors, the data-less model URL, and internal page links.
- Added missing-value and prototype/canonical projection regression checks in `tests/js/test_web_data.js`, publication catalog tests in `tests/test_web_catalog.py`, and catalog/data checks in both CI and the Pages deployment workflow.

## Maintaining the site

After changing published result files, run:

```bash
python scripts/build_web_catalog.py
python scripts/build_web_catalog.py --check
node tests/js/test_web_data.js
python -m http.server 8000 --directory web
```

The catalog is generated from available result files without changing the registry or results. Numerical summaries use observed fields and never fill missing experiments with default choices. Commit the regenerated catalog with the result updates.
