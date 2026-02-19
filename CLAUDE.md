# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NoCap-Test is an autonomous research framework that uses the Claude API to iteratively design, run, and evaluate experiments for optimizing GPT-2 training speed. The goal is to reach validation loss ≤ 3.3821 on FineWeb faster than the baseline (~5.4 hours / 4768 steps on RTX 4090) through **novel algorithmic improvements only** — not hyperparameter tuning.

## Key Commands

```bash
# Setup
pip install -r requirements.txt
python data/cached_fineweb10B.py        # Download/tokenize FineWeb data

# Run baseline training
./run.sh

# Run a specific experiment
python run_experiment.py --name exp_name --run-sh experiments/exp_name/run.sh --description "description"

# Start the autonomous orchestrator (infinite loop: Claude proposes → run → parse → repeat)
export ANTHROPIC_API_KEY=sk-ant-...
python orchestrator.py

# Dashboard (port 8080, HTTP Basic Auth)
export DASHBOARD_USER=admin
export DASHBOARD_PASS=nocap2026
python dashboard/serve.py
```

There is no test suite, linter, or build system — this is a research benchmark.

## Architecture

**Core loop:** `orchestrator.py` calls Claude Opus API → generates modified `train_gpt2.py` → `run_experiment.py` executes it → parses logs → updates `experiments.json` → feeds results back to Claude.

- **train_gpt2.py** — GPT-2 training script (RoPE, RMSNorm, AdamW, distributed training via `torchrun`). This is the file experiments modify.
- **orchestrator.py** — Autonomous experiment loop. Sends baseline script + all prior results to Claude, receives JSON with a new experiment (name, hypothesis, modified training script). Hardcodes `BASELINE_RUN_ARGS` that must not be changed.
- **run_experiment.py** — Experiment harness. Runs training, parses output via regex for val/train loss, memory, timing. Updates `experiments.json` in a background thread.
- **experiments.json** — Central registry of all experiments with status, metrics, loss histories, and findings.
- **dashboard/** — `serve.py` (authenticated HTTP server) + `index.html` (Chart.js visualization of experiment results with auto-refresh).
- **experiments/** — One directory per experiment (exp0 through exp41+), each containing a modified training script and logs.

## Critical Constraints

The baseline run arguments are **fixed and must never be modified** by experiments:
```
--model d12 --batch_size 16 --grad_accumulation_steps 32 --sequence_length 1024
--val_loss_every 128 --val_batch_size 16 --num_iterations 4768
--weight_decay 0.1 --learning_rate 0.0018 --warmup_iters 256 --warmdown_iters 1024
```

Experiments may only modify the training script logic (e.g., loss functions, architecture, training algorithm, data handling). New parameters must be hardcoded, not added via argparse.

## Log Parsing Patterns

Training output is parsed with these regex patterns:
- Validation: `step:(\d+)/(\d+) \| val loss ([\d.]+)`
- Training: `step:(\d+)/(\d+) \| loss ([\d.]+) \| train_time:([\d.]+)s`
- Memory: `peak memory consumption: (\d+) MiB`

## Experiment Naming

Format: `exp{number}_{snake_case_description}` (e.g., `exp37_swa_snapshots`). Directories under `experiments/` mirror this naming.
