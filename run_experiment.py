#!/usr/bin/env python3
"""
Experiment harness: runs training, captures logs, updates experiments.json live.

Usage:
    python run_experiment.py --name exp0_baseline --run-sh experiments/exp0_baseline/run.sh \
        --description "Unmodified baseline on RTX 3080"
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
EXPERIMENTS_JSON = PROJECT_ROOT / "experiments.json"
TARGET_VAL_LOSS = 3.3821


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


def parse_log(log_path):
    """Parse training log for val loss, train loss, timing, and memory."""
    result = {
        "val_loss_history": [],
        "train_loss_history": [],
        "peak_memory_mib": None,
        "final_val_loss": None,
        "total_time_seconds": None,
        "last_step": 0,
        "total_steps": None,
    }
    if not os.path.exists(log_path):
        return result

    last_train_time = None
    with open(log_path, "r") as f:
        for line in f:
            # Validation loss: step:N/TOTAL | val loss F
            m = re.search(r"step:(\d+)/(\d+) \| val loss ([\d.]+)", line)
            if m:
                step, total, val_loss = int(m.group(1)), int(m.group(2)), float(m.group(3))
                result["val_loss_history"].append({"step": step, "val_loss": val_loss})
                result["final_val_loss"] = val_loss
                result["total_steps"] = total
                result["last_step"] = max(result["last_step"], step)

            # Training loss: step:N/TOTAL | loss F | train_time:Fs | step_avg:Fms
            m = re.search(r"step:(\d+)/(\d+) \| loss ([\d.]+) \| train_time:([\d.]+)s", line)
            if m:
                step = int(m.group(1))
                loss = float(m.group(3))
                train_time = float(m.group(4))
                last_train_time = train_time
                result["total_steps"] = int(m.group(2))
                result["last_step"] = max(result["last_step"], step)
                # Sample train loss every 64 steps to keep JSON small
                if step % 64 == 0:
                    result["train_loss_history"].append({"step": step, "loss": loss})

            # Peak memory
            m = re.search(r"peak memory consumption: (\d+) MiB", line)
            if m:
                result["peak_memory_mib"] = int(m.group(1))

    if last_train_time is not None:
        result["total_time_seconds"] = last_train_time

    return result


def update_experiment(name, updates):
    """Thread-safe update of a single experiment in experiments.json."""
    data = load_experiments()
    for exp in data["experiments"]:
        if exp["name"] == name:
            exp.update(updates)
            break
    save_experiments(data)


def log_monitor(name, log_path, stop_event):
    """Background thread that periodically parses the log and updates experiments.json."""
    while not stop_event.is_set():
        stop_event.wait(30)
        if stop_event.is_set():
            break
        try:
            parsed = parse_log(log_path)
            updates = {
                "val_loss_history": parsed["val_loss_history"],
                "train_loss_history": parsed["train_loss_history"],
                "final_val_loss": parsed["final_val_loss"],
                "total_time_seconds": parsed["total_time_seconds"],
                "peak_memory_mib": parsed["peak_memory_mib"],
                "last_step": parsed["last_step"],
            }
            if parsed["total_steps"]:
                updates["total_steps"] = parsed["total_steps"]
            update_experiment(name, updates)
        except Exception as e:
            print(f"[monitor] Error parsing log: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Run a training experiment")
    parser.add_argument("--name", required=True, help="Experiment name (e.g. exp0_baseline)")
    parser.add_argument("--run-sh", required=True, help="Path to the run script")
    parser.add_argument("--description", default="", help="Experiment description/hypothesis")
    parser.add_argument("--display-name", default="", help="Display name for dashboard")
    args = parser.parse_args()

    exp_dir = PROJECT_ROOT / "experiments" / args.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_path = exp_dir / "output.log"

    # Register experiment
    data = load_experiments()
    # Remove existing entry with same name if re-running
    data["experiments"] = [e for e in data["experiments"] if e["name"] != args.name]
    data["experiments"].append({
        "name": args.name,
        "display_name": args.display_name or args.name,
        "description": args.description,
        "status": "running",
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "total_time_seconds": None,
        "total_steps": 4768,
        "peak_memory_mib": None,
        "final_val_loss": None,
        "success": None,
        "val_loss_history": [],
        "train_loss_history": [],
        "log_file": str(log_path.relative_to(PROJECT_ROOT)),
        "key_findings": "",
    })
    save_experiments(data)

    # Start background log monitor
    stop_event = threading.Event()
    monitor = threading.Thread(target=log_monitor, args=(args.name, str(log_path), stop_event), daemon=True)
    monitor.start()

    # Run training
    run_sh = str(PROJECT_ROOT / args.run_sh)
    print(f"[harness] Starting: {run_sh}")
    print(f"[harness] Log: {log_path}")
    print(f"[harness] Dashboard: http://localhost:8080/dashboard/index.html")

    returncode = -1
    try:
        with open(log_path, "w") as logf:
            proc = subprocess.Popen(
                ["bash", run_sh],
                stdout=logf,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
            )
            returncode = proc.wait()
    except KeyboardInterrupt:
        print("\n[harness] Interrupted, marking as failed")
        proc.terminate()
        returncode = -1
    finally:
        stop_event.set()
        monitor.join(timeout=5)

    # Final parse
    parsed = parse_log(str(log_path))
    status = "completed" if returncode == 0 else "failed"

    # Load current data to get baseline time for speed comparison
    data = load_experiments()
    baseline_time = data.get("baseline_time_seconds")

    val_loss_ok = (parsed["final_val_loss"] is not None
                   and parsed["final_val_loss"] <= TARGET_VAL_LOSS)
    time_ok = (parsed["total_time_seconds"] is not None
               and baseline_time is not None
               and parsed["total_time_seconds"] < baseline_time)

    if args.name == "exp0_baseline":
        # Baseline only checks val loss (nothing to compare speed against)
        success = val_loss_ok
        partial_success = False
    elif status != "completed":
        success = False
        partial_success = False
    else:
        # Full success: beats BOTH val loss target AND baseline speed
        success = val_loss_ok and time_ok
        # Partial: val loss achieved but not faster than baseline
        partial_success = val_loss_ok and not success

    updates = {
        "status": status,
        "end_time": datetime.now().isoformat(),
        "total_time_seconds": parsed["total_time_seconds"],
        "peak_memory_mib": parsed["peak_memory_mib"],
        "final_val_loss": parsed["final_val_loss"],
        "success": success if status == "completed" else False,
        "partial_success": partial_success if status == "completed" else False,
        "val_loss_history": parsed["val_loss_history"],
        "train_loss_history": parsed["train_loss_history"],
    }
    if parsed["total_steps"]:
        updates["total_steps"] = parsed["total_steps"]

    # If this is the baseline, record baseline time
    if args.name == "exp0_baseline" and status == "completed" and parsed["total_time_seconds"]:
        data["baseline_time_seconds"] = parsed["total_time_seconds"]
        for exp in data["experiments"]:
            if exp["name"] == args.name:
                exp.update(updates)
        save_experiments(data)
    else:
        update_experiment(args.name, updates)

    outcome = "SUCCESS (val+speed)" if success else ("PARTIAL (val loss only)" if partial_success else "not achieved")
    print(f"\n[harness] {'=' * 50}")
    print(f"[harness] Experiment: {args.name}")
    print(f"[harness] Status: {status}")
    print(f"[harness] Final val loss: {parsed['final_val_loss']} (target: {TARGET_VAL_LOSS}) -> {'OK' if val_loss_ok else 'MISS'}")
    print(f"[harness] Total time: {parsed['total_time_seconds']:.1f}s" if parsed['total_time_seconds'] else "[harness] Total time: N/A")
    if baseline_time and args.name != "exp0_baseline":
        print(f"[harness] vs baseline: {baseline_time:.1f}s -> {'FASTER' if time_ok else 'SLOWER'}")
    print(f"[harness] Outcome: {outcome}")
    print(f"[harness] Peak memory: {parsed['peak_memory_mib']} MiB")
    print(f"[harness] {'=' * 50}")

    return 0 if (success or partial_success) else 1


if __name__ == "__main__":
    sys.exit(main())
