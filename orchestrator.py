#!/usr/bin/env python3
"""
Autonomous experiment orchestrator.
Runs an infinite loop: train → analyze with LLM → generate next experiment → repeat.

Supports two LLM backends:
  1. Claude API (default): export ANTHROPIC_API_KEY=sk-ant-...
  2. Ollama (local):       python orchestrator.py --llm-provider ollama --ollama-model qwen3:32b

Usage:
    python orchestrator.py [--skip-baseline] [--llm-provider claude|ollama]
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

import requests as _requests_lib

PROJECT_ROOT = Path(__file__).parent.resolve()
EXPERIMENTS_JSON = PROJECT_ROOT / "experiments.json"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
TARGET_VAL_LOSS = 3.3821
CLAUDE_MODEL = "claude-opus-4-6"  # most capable for novel research ideas


# ---------------------------------------------------------------------------
# LLM provider abstraction
# ---------------------------------------------------------------------------

class LLMProvider:
    """Unified interface for LLM providers."""
    def chat(self, prompt: str, max_tokens: int = 16000) -> str:
        raise NotImplementedError

    def start(self):
        """Start the backend if needed (no-op for remote APIs)."""
        pass

    def stop(self):
        """Stop the backend to free resources (no-op for remote APIs)."""
        pass


class ClaudeProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = CLAUDE_MODEL):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def chat(self, prompt: str, max_tokens: int = 16000) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()


class OllamaProvider(LLMProvider):
    def __init__(self, model: str = "qwen3:32b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._proc = None
        # Just check the model name is sane; don't require server to be up at init time
        print(f"[ollama] Provider configured: model={model} url={base_url}")
        print(f"[ollama] Server will be started before each LLM call and stopped before GPU work.")

    def start(self):
        """Start ollama serve if not already running."""
        # Check if already up (e.g. user started it manually)
        try:
            _requests_lib.get(f"{self.base_url}/api/tags", timeout=2).raise_for_status()
            print("[ollama] Server already running.")
            return
        except Exception:
            pass

        print("[ollama] Starting server...")
        self._proc = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Wait until the API responds (up to 30s)
        for _ in range(30):
            time.sleep(1)
            try:
                _requests_lib.get(f"{self.base_url}/api/tags", timeout=2).raise_for_status()
                print("[ollama] Server ready.")
                return
            except Exception:
                pass
        print("[ollama] WARNING: Server may not be ready yet, proceeding anyway.")

    def stop(self):
        """Stop ollama serve to free GPU VRAM for training."""
        if self._proc and self._proc.poll() is None:
            print("[ollama] Stopping server to free GPU VRAM...")
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
            print("[ollama] Server stopped.")
        else:
            # May have been started externally; kill by name to be safe
            result = subprocess.run(
                ["pkill", "-f", "ollama serve"],
                capture_output=True,
            )
            if result.returncode == 0:
                print("[ollama] Stopped external ollama serve process.")

    def chat(self, prompt: str, max_tokens: int = 16000) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7,
            },
        }
        r = _requests_lib.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=600,  # large models can be slow
        )
        r.raise_for_status()
        return r.json()["message"]["content"].strip()


def create_llm_provider(args) -> LLMProvider:
    """Factory: create the right LLM provider from CLI args."""
    provider = args.llm_provider

    if provider == "auto":
        # Auto-detect: prefer Claude if key is set, else try Ollama
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider = "claude"
        else:
            provider = "ollama"
            print("[orchestrator] No ANTHROPIC_API_KEY found, falling back to Ollama")

    if provider == "claude":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: Set ANTHROPIC_API_KEY environment variable (or use --llm-provider ollama)")
            sys.exit(1)
        return ClaudeProvider(api_key=api_key)

    elif provider == "ollama":
        return OllamaProvider(
            model=args.ollama_model,
            base_url=args.ollama_url,
        )

    else:
        print(f"ERROR: Unknown LLM provider: {provider}")
        sys.exit(1)

# Fixed baseline run args - Claude should NOT change these
BASELINE_RUN_ARGS = (
    "--model d12 "
    "--batch_size 16 "
    "--grad_accumulation_steps 32 "
    "--sequence_length 1024 "
    "--val_loss_every 128 "
    "--val_batch_size 16 "
    "--num_iterations 4768 "
    "--weight_decay 0.1 "
    "--learning_rate 0.0018 "
    "--warmup_iters 256 "
    "--warmdown_iters 1024"
)


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

    baseline_time = data.get("baseline_time_seconds")
    for exp in data["experiments"]:
        status = exp["status"]
        name = exp["display_name"] or exp["name"]
        val = exp.get("final_val_loss", "N/A")
        time_s = exp.get("total_time_seconds", "N/A")
        success = exp.get("success", False)
        partial = exp.get("partial_success", False)
        findings = exp.get("key_findings", "")
        desc = exp.get("description", "")

        val_ok = val != "N/A" and val is not None and val <= TARGET_VAL_LOSS
        time_ok = (time_s != "N/A" and time_s is not None and baseline_time is not None
                   and time_s < baseline_time)

        if success:
            outcome = "FULL SUCCESS (val loss + faster than baseline)"
        elif partial:
            parts = []
            if val_ok:
                parts.append("val loss OK")
            if time_ok:
                parts.append("faster than baseline")
            outcome = f"PARTIAL ({', '.join(parts)} — but not both)"
        else:
            outcome = "not achieved"

        lines.append(f"## {name}")
        lines.append(f"  Status: {status} | Val Loss: {val} | Time: {time_s}s | Outcome: {outcome}")
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


def call_claude_for_experiment(llm, exp_num):
    """Ask the LLM to analyze results and propose the next experiment."""
    baseline_script = get_baseline_script()
    summary = get_experiment_summary()

    prompt = f"""You are an AI researcher designing experiments to speed up GPT-2 training.

## Goal
Reach validation loss ≤ {TARGET_VAL_LOSS} on FineWeb faster than the baseline.
"Faster" means reaching the SAME val loss in FEWER steps or LESS wall-clock time.

## Rules (from the competition)
- Focus on NOVEL algorithmic ideas that could scale
- Do NOT: just tune hyperparameters, copy known techniques (SwiGLU from LLaMA, MTP from Meta paper, etc.)
- ALLOWED: modify loss function, architecture, training algorithm, data handling, new ideas
- The ideas should be your OWN creative hypotheses, not standard recipes

## CRITICAL: Fixed Training Configuration
The following run args are FIXED and MUST NOT be changed:
  {BASELINE_RUN_ARGS}

This means: batch_size=16, grad_accumulation_steps=32, sequence_length=1024,
num_iterations=4768, learning_rate=0.0018, etc.
Total tokens per step: 16 * 32 * 1024 = 524,288
You may ONLY modify the training SCRIPT (model architecture, loss function, optimizer logic, etc.)
Do NOT add new command-line arguments. Any new parameters should be hardcoded in the script.

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
    "expected_impact": "What you expect to happen and why",
    "risk_assessment": "What could go wrong"
}}

IMPORTANT:
- The training_script must be COMPLETE and RUNNABLE (not a diff or partial)
- Do NOT include "run_args" - they are fixed and provided by the harness
- Do NOT add new argparse arguments - hardcode any new parameters in the script
- Be creative - try something that hasn't been tried before
- If previous experiments failed, learn from those failures
- Keep the same data loading format (binary shards with the existing header format)
- The script must work with the EXACT same command-line args as the baseline
"""

    provider_name = type(llm).__name__.replace("Provider", "")
    print(f"[orchestrator] Calling {provider_name} to design experiment #{exp_num}...")
    text = llm.chat(prompt, max_tokens=16000)

    # Try to parse JSON - handle markdown fencing and surrounding text
    result = None

    # Method 1: direct parse
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        pass

    # Method 2: strip markdown fences
    if result is None:
        cleaned = re.sub(r"^```(?:json)?\n?", "", text)
        cleaned = re.sub(r"\n?```$", "", cleaned)
        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            pass

    # Method 3: find JSON object in the text (between first { and last })
    if result is None:
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            json_str = text[first_brace:last_brace + 1]
            try:
                result = json.loads(json_str)
            except json.JSONDecodeError:
                pass

    if result is None:
        print(f"[orchestrator] Failed to parse LLM response as JSON")
        print(f"[orchestrator] Raw response (first 500 chars): {text[:500]}")
        raise ValueError("Could not extract JSON from LLM response")

    required_keys = ["hypothesis", "description", "experiment_name", "training_script"]
    for key in required_keys:
        if key not in result:
            raise ValueError(f"LLM response missing required key: {key}")

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

    # Write run script - ALWAYS use baseline run args
    run_sh = exp_dir / "run.sh"
    with open(run_sh, "w") as f:
        f.write(f"""#!/bin/bash
cd {PROJECT_ROOT}
torchrun --standalone --nproc_per_node=1 {exp_dir.relative_to(PROJECT_ROOT)}/train_gpt2.py \\
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \\
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \\
  --output_dir {exp_dir.relative_to(PROJECT_ROOT)} \\
  {BASELINE_RUN_ARGS}
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


def analyze_results(llm, exp_name):
    """Ask the LLM to analyze the results of the latest experiment."""
    summary = get_experiment_summary()

    prompt = f"""Analyze the results of experiment "{exp_name}" and provide a brief (2-3 sentence) summary of what happened and what we learned. Be specific about numbers.

## All Experiments
{summary}

Respond with just the analysis text, no JSON or formatting."""

    return llm.chat(prompt, max_tokens=1000)


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
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries for failed API calls or smoke tests")
    parser.add_argument("--llm-provider", choices=["claude", "ollama", "auto"], default="auto",
                        help="LLM backend: 'claude' (needs ANTHROPIC_API_KEY), 'ollama' (local), or 'auto' (default)")
    parser.add_argument("--ollama-model", default="qwen3:32b", help="Ollama model name (default: qwen3:32b)")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama API base URL")
    args = parser.parse_args()

    llm = create_llm_provider(args)

    provider_label = type(llm).__name__.replace("Provider", "")
    model_label = getattr(llm, "model", CLAUDE_MODEL)
    print("=" * 60)
    print("  GPT-2 Training Speedup - Autonomous Orchestrator")
    print("=" * 60)
    print(f"  LLM:    {provider_label} ({model_label})")
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

        # Ask LLM to design the next experiment (start server, call, then stop before GPU work)
        llm.start()
        for attempt in range(args.max_retries):
            try:
                exp_data = call_claude_for_experiment(llm, exp_num)
                break
            except Exception as e:
                print(f"[orchestrator] LLM error (attempt {attempt+1}/{args.max_retries}): {e}")
                if attempt < args.max_retries - 1:
                    time.sleep(10)
                else:
                    llm.stop()
                    print("[orchestrator] Skipping this iteration due to API errors")
                    exp_num += 1
                    continue
        llm.stop()

        print(f"[orchestrator] Hypothesis: {exp_data.get('hypothesis', 'N/A')}")
        print(f"[orchestrator] Experiment: {exp_data.get('display_name', exp_data['experiment_name'])}")

        # Store full Claude response metadata for replication
        exp_metadata = {
            "hypothesis": exp_data.get("hypothesis", ""),
            "expected_impact": exp_data.get("expected_impact", ""),
            "risk_assessment": exp_data.get("risk_assessment", ""),
            "run_args": BASELINE_RUN_ARGS,
        }

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
                print(f"[orchestrator] Smoke test failed, asking LLM to fix (attempt {attempt+1})...")
                # Ask LLM to fix the script (start server, get fix, stop before next smoke test)
                try:
                    fix_prompt = f"""The training script for experiment "{exp_data['experiment_name']}" failed the smoke test with this error:

```
{smoke_error}
```

Here is the current script:
```python
{exp_data['training_script']}
```

Please provide the COMPLETE fixed training script. Respond with ONLY the Python code, no markdown fences or explanation."""
                    llm.start()
                    fixed_script = llm.chat(fix_prompt, max_tokens=16000)
                    llm.stop()
                    if fixed_script.startswith("```"):
                        fixed_script = re.sub(r"^```(?:python)?\n?", "", fixed_script)
                        fixed_script = re.sub(r"\n?```$", "", fixed_script)
                    exp_data["training_script"] = fixed_script
                    with open(exp_dir / "train_gpt2.py", "w") as f:
                        f.write(fixed_script)
                except Exception as e:
                    llm.stop()
                    print(f"[orchestrator] Failed to get fix from LLM: {e}")

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
                "claude_metadata": exp_metadata,
            })
            save_experiments(data)
            exp_num += 1
            continue

        # Run full training
        description = exp_data.get("description", exp_data.get("hypothesis", ""))
        returncode = run_full_training(exp_name, run_sh, description)

        # Store Claude metadata for replication
        data = load_experiments()
        for exp in data["experiments"]:
            if exp["name"] == exp_name:
                exp["claude_metadata"] = exp_metadata
                break
        save_experiments(data)

        # Analyze results (training is done, GPU is free — start LLM, analyze, stop)
        try:
            llm.start()
            findings = analyze_results(llm, exp_name)
            llm.stop()
            record_findings(exp_name, findings)
            print(f"[orchestrator] Analysis: {findings}")
        except Exception as e:
            llm.stop()
            print(f"[orchestrator] Failed to analyze results: {e}")

        # Check for success
        data = load_experiments()
        exp_result = next((e for e in data["experiments"] if e["name"] == exp_name), None)
        if exp_result:
            baseline_time = data.get("baseline_time_seconds", 0)
            exp_time = exp_result.get("total_time_seconds", 0)
            speedup = ((1 - exp_time / baseline_time) * 100) if baseline_time else 0
            if exp_result.get("success"):
                print(f"\n{'=' * 60}")
                print(f"  FULL SUCCESS! {exp_name} beat BOTH targets!")
                print(f"  Val loss: {exp_result['final_val_loss']:.6f} (target: {TARGET_VAL_LOSS})")
                print(f"  Time: {exp_time:.1f}s vs baseline {baseline_time:.1f}s ({speedup:+.1f}%)")
                print(f"{'=' * 60}\n")
                # Keep going to find even better results
            elif exp_result.get("partial_success"):
                val_ok = exp_result.get("final_val_loss") is not None and exp_result["final_val_loss"] <= TARGET_VAL_LOSS
                time_ok = exp_time and baseline_time and exp_time < baseline_time
                print(f"\n  PARTIAL SUCCESS: {exp_name}")
                print(f"  Val loss: {exp_result['final_val_loss']:.6f} ({'OK' if val_ok else 'MISS'})")
                print(f"  Speed: {speedup:+.1f}% vs baseline ({'OK' if time_ok else 'MISS'})\n")

        exp_num += 1
        print(f"[orchestrator] Moving to next experiment...")


if __name__ == "__main__":
    main()
