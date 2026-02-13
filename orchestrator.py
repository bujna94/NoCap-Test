#!/usr/bin/env python3
"""
Autonomous experiment orchestrator.
Runs an infinite loop: train → analyze with Claude API → generate next experiment → repeat.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python orchestrator.py [--skip-baseline]
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

import anthropic

PROJECT_ROOT = Path(__file__).parent.resolve()
EXPERIMENTS_JSON = PROJECT_ROOT / "experiments.json"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
TARGET_VAL_LOSS = 3.3821
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"  # fast + capable for code generation


def load_experiments():
    if EXPERIMENTS_JSON.exists():
        with open(EXPERIMENTS_JSON) as f:
            return json.load(f)
    return {"target_val_loss": TARGET_VAL_LOSS, "baseline_time_seconds": None, "experiments": []}


def save_experiments(data):
    tmp = EXPERIMENTS_JSON.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.rename(EXPERIMENTS_JSON)


def get_baseline_script():
    """Read the original training script."""
    with open(PROJECT_ROOT / "train_gpt2.py") as f:
        return f.read()


def get_experiment_summary():
    """Build a summary of all experiments for Claude."""
    data = load_experiments()
    lines = []
    lines.append(f"Target val loss: {data['target_val_loss']}")
    lines.append(f"Baseline time: {data.get('baseline_time_seconds', 'unknown')}s")
    lines.append("")

    for exp in data["experiments"]:
        status = exp["status"]
        name = exp["display_name"] or exp["name"]
        val = exp.get("final_val_loss", "N/A")
        time_s = exp.get("total_time_seconds", "N/A")
        success = exp.get("success", False)
        findings = exp.get("key_findings", "")
        desc = exp.get("description", "")

        lines.append(f"## {name}")
        lines.append(f"  Status: {status} | Val Loss: {val} | Time: {time_s}s | Success: {success}")
        lines.append(f"  Hypothesis: {desc}")
        if findings:
            lines.append(f"  Findings: {findings}")

        # Show val loss progression
        if exp.get("val_loss_history"):
            vlh = exp["val_loss_history"]
            if len(vlh) > 5:
                # Show first, middle, and last few
                selected = vlh[:2] + vlh[len(vlh)//2:len(vlh)//2+1] + vlh[-3:]
            else:
                selected = vlh
            prog = ", ".join(f"step {p['step']}→{p['val_loss']:.4f}" for p in selected)
            lines.append(f"  Loss curve: {prog}")
        lines.append("")

    return "\n".join(lines)


def get_next_exp_number():
    """Get the next experiment number."""
    data = load_experiments()
    existing = [e["name"] for e in data["experiments"]]
    n = 0
    while f"exp{n}_" in " ".join(existing) or f"exp{n}_" in str(list(EXPERIMENTS_DIR.iterdir()) if EXPERIMENTS_DIR.exists() else []):
        n += 1
    return n


def call_claude_for_experiment(client, exp_num):
    """Ask Claude to analyze results and propose the next experiment."""
    baseline_script = get_baseline_script()
    summary = get_experiment_summary()

    prompt = f"""You are an AI researcher designing experiments to speed up GPT-2 training.

## Goal
Reach validation loss ≤ {TARGET_VAL_LOSS} on FineWeb faster than the baseline.

## Rules (from the competition)
- Focus on NOVEL algorithmic ideas that could scale
- Do NOT: just tune hyperparameters, copy known techniques (SwiGLU from LLaMA, MTP from Meta paper, etc.)
- ALLOWED: modify loss function, architecture, training algorithm, data handling, new ideas
- The ideas should be your OWN creative hypotheses, not standard recipes

## Hardware
- RTX 4090, 24GB VRAM
- Baseline: ~4078ms/step, 4768 steps, ~5.4 hours total

## Previous Experiments
{summary}

## Baseline Training Script
```python
{baseline_script}
```

## Your Task
1. Analyze all previous experiment results - what worked, what didn't, WHY
2. Formulate a NOVEL HYPOTHESIS about why training could be faster
3. Design a minimal experiment to test it
4. Write the COMPLETE modified training script (it must be self-contained and runnable)

## Response Format
Respond with EXACTLY this JSON structure (no markdown, no code fences, just raw JSON):
{{
    "hypothesis": "One sentence describing your hypothesis",
    "description": "2-3 sentences explaining the idea and why it should work",
    "experiment_name": "short_snake_case_name",
    "display_name": "Human Readable Name",
    "training_script": "... complete Python training script ...",
    "run_args": "... the torchrun command arguments (everything after train_gpt2.py) ...",
    "expected_impact": "What you expect to happen and why",
    "risk_assessment": "What could go wrong"
}}

IMPORTANT:
- The training_script must be COMPLETE and RUNNABLE (not a diff or partial)
- The run_args should include all necessary flags
- Be creative - try something that hasn't been tried before
- If previous experiments failed, learn from those failures
- Keep the same data loading format (binary shards with the existing header format)
"""

    print(f"[orchestrator] Calling Claude API to design experiment #{exp_num}...")
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()

    # Try to parse JSON - handle potential markdown fencing
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

    try:
        result = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"[orchestrator] Failed to parse Claude response as JSON: {e}")
        print(f"[orchestrator] Raw response (first 500 chars): {text[:500]}")
        raise

    required_keys = ["hypothesis", "description", "experiment_name", "training_script", "run_args"]
    for key in required_keys:
        if key not in result:
            raise ValueError(f"Claude response missing required key: {key}")

    return result


def setup_experiment(exp_data, exp_num):
    """Create experiment directory, script, and run.sh."""
    name = f"exp{exp_num}_{exp_data['experiment_name']}"
    exp_dir = EXPERIMENTS_DIR / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Write training script
    script_path = exp_dir / "train_gpt2.py"
    with open(script_path, "w") as f:
        f.write(exp_data["training_script"])

    # Write run script
    run_args = exp_data["run_args"]
    run_sh = exp_dir / "run.sh"
    with open(run_sh, "w") as f:
        f.write(f"""#!/bin/bash
cd {PROJECT_ROOT}
torchrun --standalone --nproc_per_node=1 {exp_dir.relative_to(PROJECT_ROOT)}/train_gpt2.py \\
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \\
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \\
  --output_dir {exp_dir.relative_to(PROJECT_ROOT)} \\
  {run_args}
""")
    run_sh.chmod(0o755)

    return name, exp_dir, run_sh


def smoke_test(exp_dir, exp_name):
    """Run a quick 10-step test to verify the experiment works."""
    print(f"[orchestrator] Smoke testing {exp_name}...")
    script = exp_dir / "train_gpt2.py"
    cmd = [
        "torchrun", "--standalone", "--nproc_per_node=1", str(script),
        "--input_bin", "data/fineweb10B/fineweb_train_*.bin",
        "--input_val_bin", "data/fineweb10B/fineweb_val_*.bin",
        "--output_dir", "/tmp/smoke_test",
        "--model", "d12",
        "--batch_size", "4",
        "--grad_accumulation_steps", "2",
        "--sequence_length", "256",
        "--val_loss_every", "5",
        "--val_batch_size", "4",
        "--num_iterations", "10",
        "--warmup_iters", "2",
        "--warmdown_iters", "2",
        "--weight_decay", "0.1",
        "--learning_rate", "0.0018",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            print(f"[orchestrator] Smoke test FAILED:")
            print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
            return False, result.stderr[-2000:]
        print(f"[orchestrator] Smoke test PASSED")
        return True, ""
    except subprocess.TimeoutExpired:
        print(f"[orchestrator] Smoke test TIMED OUT (>10 min)")
        return False, "Smoke test timed out after 10 minutes"
    except Exception as e:
        print(f"[orchestrator] Smoke test ERROR: {e}")
        return False, str(e)


def run_full_training(exp_name, run_sh_path, description):
    """Run the full training via the experiment harness."""
    print(f"[orchestrator] Starting full training: {exp_name}")
    cmd = [
        sys.executable, str(PROJECT_ROOT / "run_experiment.py"),
        "--name", exp_name,
        "--display-name", exp_name,
        "--run-sh", str(run_sh_path.relative_to(PROJECT_ROOT)),
        "--description", description,
    ]

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def record_findings(exp_name, findings):
    """Update experiment with key findings from Claude's analysis."""
    data = load_experiments()
    for exp in data["experiments"]:
        if exp["name"] == exp_name:
            exp["key_findings"] = findings
            break
    save_experiments(data)


def analyze_results(client, exp_name):
    """Ask Claude to analyze the results of the latest experiment."""
    summary = get_experiment_summary()

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": f"""Analyze the results of experiment "{exp_name}" and provide a brief (2-3 sentence) summary of what happened and what we learned. Be specific about numbers.

## All Experiments
{summary}

Respond with just the analysis text, no JSON or formatting."""}],
    )

    return response.content[0].text.strip()


def run_baseline(skip=False):
    """Run or skip the baseline."""
    data = load_experiments()
    baseline = next((e for e in data["experiments"] if e["name"] == "exp0_baseline" and e["status"] == "completed"), None)

    if baseline and skip:
        print(f"[orchestrator] Baseline already completed: {baseline['total_time_seconds']:.1f}s, val_loss={baseline['final_val_loss']:.4f}")
        return

    if baseline:
        print(f"[orchestrator] Baseline already completed, skipping.")
        return

    print("[orchestrator] Running baseline...")
    baseline_dir = EXPERIMENTS_DIR / "exp0_baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    run_sh = baseline_dir / "run.sh"
    if not run_sh.exists():
        with open(run_sh, "w") as f:
            f.write(f"""#!/bin/bash
cd {PROJECT_ROOT}
torchrun --standalone --nproc_per_node=1 train_gpt2.py \\
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \\
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \\
  --output_dir experiments/exp0_baseline \\
  --model d12 \\
  --batch_size 16 \\
  --grad_accumulation_steps 32 \\
  --sequence_length 1024 \\
  --val_loss_every 128 \\
  --val_batch_size 16 \\
  --num_iterations 4768 \\
  --weight_decay 0.1 \\
  --learning_rate 0.0018 \\
  --warmup_iters 256 \\
  --warmdown_iters 1024
""")
        run_sh.chmod(0o755)

    returncode = run_full_training(
        "exp0_baseline",
        run_sh,
        "Unmodified baseline to establish reference time",
    )

    if returncode == 0:
        data = load_experiments()
        baseline = next((e for e in data["experiments"] if e["name"] == "exp0_baseline"), None)
        if baseline and baseline.get("total_time_seconds"):
            data["baseline_time_seconds"] = baseline["total_time_seconds"]
            save_experiments(data)
            print(f"[orchestrator] Baseline complete: {baseline['total_time_seconds']:.1f}s")
    else:
        print("[orchestrator] WARNING: Baseline failed!")


def main():
    parser = argparse.ArgumentParser(description="Autonomous experiment orchestrator")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline if already run")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries for failed Claude API calls or smoke tests")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    print("=" * 60)
    print("  GPT-2 Training Speedup - Autonomous Orchestrator")
    print("=" * 60)
    print(f"  Target: val_loss ≤ {TARGET_VAL_LOSS}")
    print(f"  Dashboard: http://localhost:8080/dashboard/index.html")
    print("=" * 60)

    # Start dashboard server if not running
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:8080/experiments.json", timeout=2)
        print("[orchestrator] Dashboard already running.")
    except Exception:
        print("[orchestrator] Starting dashboard server...")
        # Use the same Python that's running this script
        subprocess.Popen(
            [sys.executable, str(PROJECT_ROOT / "dashboard" / "serve.py")],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,  # detach from parent so it survives orchestrator restart
        )
        time.sleep(2)  # give it a moment to start
        try:
            urllib.request.urlopen("http://localhost:8080/experiments.json", timeout=3)
            print("[orchestrator] Dashboard started successfully.")
        except Exception:
            print("[orchestrator] WARNING: Dashboard may not have started. Run manually:")
            print(f"  {sys.executable} {PROJECT_ROOT / 'dashboard' / 'serve.py'} &")

    # Phase 1: Baseline
    run_baseline(skip=args.skip_baseline)

    # Phase 2: Infinite experiment loop
    exp_num = get_next_exp_number()

    while True:
        print(f"\n{'=' * 60}")
        print(f"  EXPERIMENT LOOP - Iteration #{exp_num}")
        print(f"{'=' * 60}")

        # Ask Claude to design the next experiment
        for attempt in range(args.max_retries):
            try:
                exp_data = call_claude_for_experiment(client, exp_num)
                break
            except Exception as e:
                print(f"[orchestrator] Claude API error (attempt {attempt+1}/{args.max_retries}): {e}")
                if attempt < args.max_retries - 1:
                    time.sleep(10)
                else:
                    print("[orchestrator] Skipping this iteration due to API errors")
                    exp_num += 1
                    continue

        print(f"[orchestrator] Hypothesis: {exp_data.get('hypothesis', 'N/A')}")
        print(f"[orchestrator] Experiment: {exp_data.get('display_name', exp_data['experiment_name'])}")

        # Setup experiment files
        exp_name, exp_dir, run_sh = setup_experiment(exp_data, exp_num)

        # Smoke test
        smoke_passed = False
        smoke_error = ""
        for attempt in range(args.max_retries):
            passed, error = smoke_test(exp_dir, exp_name)
            if passed:
                smoke_passed = True
                break
            smoke_error = error
            if attempt < args.max_retries - 1:
                print(f"[orchestrator] Smoke test failed, asking Claude to fix (attempt {attempt+1})...")
                # Ask Claude to fix the script
                try:
                    fix_response = client.messages.create(
                        model=CLAUDE_MODEL,
                        max_tokens=16000,
                        messages=[{"role": "user", "content": f"""The training script for experiment "{exp_data['experiment_name']}" failed the smoke test with this error:

```
{smoke_error}
```

Here is the current script:
```python
{exp_data['training_script']}
```

Please provide the COMPLETE fixed training script. Respond with ONLY the Python code, no markdown fences or explanation."""}],
                    )
                    fixed_script = fix_response.content[0].text.strip()
                    if fixed_script.startswith("```"):
                        fixed_script = re.sub(r"^```(?:python)?\n?", "", fixed_script)
                        fixed_script = re.sub(r"\n?```$", "", fixed_script)
                    exp_data["training_script"] = fixed_script
                    with open(exp_dir / "train_gpt2.py", "w") as f:
                        f.write(fixed_script)
                except Exception as e:
                    print(f"[orchestrator] Failed to get fix from Claude: {e}")

        if not smoke_passed:
            print(f"[orchestrator] Smoke test failed after {args.max_retries} attempts, skipping experiment")
            # Record the failure
            data = load_experiments()
            data["experiments"].append({
                "name": exp_name,
                "display_name": exp_data.get("display_name", exp_name),
                "description": exp_data.get("description", ""),
                "status": "failed",
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_time_seconds": None,
                "total_steps": 4768,
                "peak_memory_mib": None,
                "final_val_loss": None,
                "success": False,
                "val_loss_history": [],
                "train_loss_history": [],
                "log_file": "",
                "key_findings": f"Failed smoke test: {smoke_error[:500]}",
            })
            save_experiments(data)
            exp_num += 1
            continue

        # Run full training
        description = exp_data.get("description", exp_data.get("hypothesis", ""))
        returncode = run_full_training(exp_name, run_sh, description)

        # Analyze results
        try:
            findings = analyze_results(client, exp_name)
            record_findings(exp_name, findings)
            print(f"[orchestrator] Analysis: {findings}")
        except Exception as e:
            print(f"[orchestrator] Failed to analyze results: {e}")

        # Check for success
        data = load_experiments()
        exp_result = next((e for e in data["experiments"] if e["name"] == exp_name), None)
        if exp_result and exp_result.get("success"):
            baseline_time = data.get("baseline_time_seconds", 0)
            exp_time = exp_result.get("total_time_seconds", 0)
            speedup = ((1 - exp_time / baseline_time) * 100) if baseline_time else 0
            print(f"\n{'🎉' * 10}")
            print(f"  SUCCESS! {exp_name} beat the baseline!")
            print(f"  Time: {exp_time:.1f}s vs baseline {baseline_time:.1f}s ({speedup:+.1f}%)")
            print(f"  Val loss: {exp_result['final_val_loss']:.6f}")
            print(f"{'🎉' * 10}\n")
            # Keep going to find even better results

        exp_num += 1
        print(f"[orchestrator] Moving to next experiment...")


if __name__ == "__main__":
    main()
