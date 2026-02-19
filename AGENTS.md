# Repository Guidelines

## Project Structure & Module Organization
- Core training logic lives in `train_gpt2.py` (baseline single-GPU GPT-2 run).
- Autonomous experiment orchestration is in `orchestrator.py`; experiment execution and log parsing are in `run_experiment.py`.
- Data helpers are in `data/` (notably `data/cached_fineweb10B.py` for dataset download/prep).
- UI monitoring tools live in `dashboard/` (`serve.py` + `index.html`).
- Experiment artifacts are stored in `experiments/<exp_name>/` (typically `run.sh`, `train_gpt2.py`, and logs). Aggregate metadata is tracked in `experiments.json`.
- Reference assets/docs: `img/`, `README.md`, `CLAUDE.md`.

## Build, Test, and Development Commands
- Install dependencies: `pip install -r requirements.txt`
- Download/process FineWeb subset: `python data/cached_fineweb10B.py`
- Run baseline: `./run.sh`
- Run one experiment through harness:
  `python run_experiment.py --name exp0_baseline --run-sh experiments/exp0_baseline/run.sh --description "baseline"`
- Start autonomous loop:
  `export ANTHROPIC_API_KEY=... && python orchestrator.py`
- Start dashboard:
  `export DASHBOARD_USER=admin DASHBOARD_PASS=... && python dashboard/serve.py`

## Coding Style & Naming Conventions
- Use Python with 4-space indentation and clear, minimal functions.
- Follow existing naming patterns: `snake_case` for functions/variables, `UPPER_SNAKE_CASE` for constants.
- Experiment names/directories should follow `exp<number>_<snake_case_description>` (example: `exp37_swa_snapshots`).
- Keep baseline run arguments stable unless a change is explicitly intended and documented.

## Testing Guidelines
- There is no formal unit-test suite in this repository.
- Validate changes by running a targeted experiment and confirming parsed metrics in `experiments.json`.
- For training-script edits, verify log lines still match harness regex expectations (step/loss/time and peak memory patterns in `run_experiment.py`).
- Include reproducible run commands in PR descriptions.

## Commit & Pull Request Guidelines
- Recent history favors concise, imperative commit subjects (for example: `Fix JSON parsing...`, `Add HTTP Basic Auth...`).
- Prefer small, focused commits scoped to one change.
- PRs should include:
  - What changed and why
  - Related issue/context
  - Reproduction command(s)
  - Before/after metrics or dashboard screenshot when behavior/performance changes

## Security & Configuration Tips
- Do not commit secrets (`ANTHROPIC_API_KEY`, dashboard credentials, W&B tokens).
- Keep machine-specific settings in environment variables, not hardcoded values.
