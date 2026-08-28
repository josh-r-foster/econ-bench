# EconBench pilot run roadmap

Updated 2026-08-28

## Objective

Run the complete EconBench 1.0.0 experiment matrix for one model, validate all
canonical raw and derived records, regenerate the pilot dashboard projections,
and decide whether the protocol is ready for broader collection.

The designated pilot is `gemini-2.5-flash-lite`. It covers all eleven active
experiments through `scripts/run_pilot.py`, which delegates collection to the
canonical runner in `scripts/run_benchmark.py`.

The independent audit is advisory and does not block this pilot. Repository
cleanliness, canonical capture, schema validation, response validity, and
post-run review remain mandatory.

## Expected workload

- Maximum provider calls: 8,306, excluding retries and replacement runs
- Repository cost estimate: 0.08 to 0.48 US dollars
- Approximate serial runtime at 2 to 10 seconds per call: 4.6 to 23.1 hours
- Canonical output root: `data/releases/1.0.0`
- Pilot model: `gemini-2.5-flash-lite`
- Recommended run identifier: `pilot-gemini-flash-lite-001`

Actual workload can be lower when elicitation sequences are censored before
bisection. Provider latency, rate limits, retries, and invalid-run replacement
can increase time or cost.

## Completion criteria

The pilot is complete only when all of the following are true.

- The live endpoint smoke check passes.
- All eleven experiments complete under one run identifier.
- `scripts/validate_results.py` reports `PASS` for all eleven pilot cells.
- No cell contains provider failures or interrupted trials.
- The aggregate invalid-response rate is no greater than five percent.
- Every repeated condition has at least an 80 percent valid-response rate.
- Dashboard and rationality projections regenerate from canonical records.
- Pilot metrics and charts receive a manual plausibility review.
- A written decision records whether to proceed, correct and repeat, or stop.

## 1. Prepare the environment

Use Python 3.12. Install the pinned CPU-only dependency set used by continuous
integration. The larger `requirements.txt` set is needed only for local model
wrappers and is not required for this Google-hosted pilot.

```bash
python --version
python -m pip install -r requirements-test.txt
```

Create the ignored credential file and add a valid Google API key. Never commit
the populated `.env` file.

```bash
cp .env.example .env
```

```env
GOOGLE_API_KEY=...
```

Confirm that the selected account has access to `gemini-2.5-flash-lite` and
enough quota for the expected call volume. The launcher performs a fresh live
smoke request before collection.

## 2. Pass the offline gate

Run every command from the repository root before making a provider request.

```bash
python -m pytest -q
python scripts/validate_protocol.py
python scripts/estimate_release.py --json
python -m compileall -q src scripts tests
git diff --check
git status --short --branch
```

The Git working tree must be clean. Commit intentional changes before starting
the pilot. Do not begin from a dirty snapshot because the canonical runner will
reject it.

## 3. Run the nonnetwork pilot preflight

The preflight checks Python, the registered model and endpoint, the run
identifier, repository cleanliness, and `GOOGLE_API_KEY`. It does not contact
Google.

```bash
python scripts/run_pilot.py \
  --model gemini-2.5-flash-lite \
  --run-id pilot-gemini-flash-lite-001 \
  --preflight-only
```

Proceed only after the command prints `PREFLIGHT PASS`.

## 4. Launch the pilot

Use the same model and run identifier from preflight.

```bash
python scripts/run_pilot.py \
  --model gemini-2.5-flash-lite \
  --run-id pilot-gemini-flash-lite-001
```

The launcher first requests one response capped at eight output tokens. A
failed smoke check aborts before canonical collection. A successful check is
followed by the complete eleven-experiment matrix.

Each observed trial is checkpointed immediately. Raw records are stored at:

```text
data/releases/1.0.0/raw/gemini-2.5-flash-lite/<experiment>/pilot-gemini-flash-lite-001.jsonl
```

Each completed aggregate is stored at:

```text
data/releases/1.0.0/derived/gemini-2.5-flash-lite/<experiment>.json
```

Do not edit generated records by hand.

## 5. Resume an interrupted pilot

If the process is interrupted, do not change code, manifests, dependencies, or
Python environments. Resume with the exact model and run identifier.

```bash
python scripts/run_pilot.py \
  --model gemini-2.5-flash-lite \
  --run-id pilot-gemini-flash-lite-001 \
  --resume
```

Resume removes the recorded interrupted placeholder and continues from the last
immutable observed trial. It rejects stale code revisions, changed experiment
settings, changed runtime provenance, and invalid completed checkpoints.

Use `--skip-smoke` only when resuming immediately after a successful smoke
check and the same provider session is known to remain healthy.

## 6. Validate the completed pilot

Do not use `--allow-incomplete` for pilot acceptance. The command must exit
successfully and report eleven `PASS` cells.

```bash
python scripts/validate_results.py \
  --models gemini-2.5-flash-lite
```

Interpret nonpassing states as follows.

- `MISSING`: an expected raw or derived artifact is absent.
- `PARTIAL`: the run is schema-valid but fails release eligibility or quality
  thresholds.
- `INVALID`: provenance, trial structure, aggregation, or schema validation
  failed.

An interrupted run should be resumed. A completed run that fails a response or
quality threshold must remain preserved and be repeated in full under a new run
identifier. Never overwrite the failed run.

## 7. Generate pilot projections

After all eleven cells pass, regenerate dashboard and rationality projections
for only the pilot model.

```bash
python scripts/update_leaderboard.py \
  --models gemini-2.5-flash-lite
```

Review the resulting `web/data` changes and confirm that each projection names
benchmark version `1.0.0`, the pilot model, and the completed run.

## 8. Review and decide

Inspect validity rates, condition-level sample counts, raw responses, and all
reported metrics. Pay particular attention to:

- censored Independence and Time sequences;
- validation and bidirectional consistency;
- Ultimatum responder acceptance curves;
- zero variance or unexpectedly perfect strategic-game behavior;
- monetary-treatment comparisons;
- Matching Pennies action balance and uncertainty;
- provider errors, empty responses, and parser rejections.

Record one outcome.

- `PROCEED`: all acceptance checks pass and wider collection may begin.
- `REPEAT`: a provider or response-quality failure requires a complete new run.
- `CORRECT`: a protocol, parser, schema, or aggregation defect requires code
  changes, a new commit, the offline gate, and a fresh pilot run identifier.
- `STOP`: cost, endpoint access, or scientific concerns make collection
  unsuitable.

## Operator checklist

- [ ] Python 3.12 is active.
- [ ] Pinned dependencies are installed.
- [ ] `GOOGLE_API_KEY` is configured in the ignored `.env` file.
- [ ] Endpoint access and quota are confirmed.
- [ ] The offline gate passes.
- [ ] The Git working tree is clean.
- [ ] Nonnetwork preflight passes.
- [ ] Live smoke check passes.
- [ ] All eleven experiments complete under one run identifier.
- [ ] All eleven validation cells report `PASS`.
- [ ] Pilot projections regenerate successfully.
- [ ] Metrics and charts pass manual review.
- [ ] The proceed, repeat, correct, or stop decision is recorded.
