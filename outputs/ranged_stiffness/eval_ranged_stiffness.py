#!/usr/bin/env python3
"""Run the 14-command ranged-stiffness EE compliance sweep.

This runner launches ``scripts/eval_manipulation.py`` once per policy and
stiffness command.  The evaluator writes its JSON report before Isaac Sim
fully returns, so this script watches for a complete 60-probe report and then
interrupts the child process gracefully.  This avoids leaving one eval alive
after its report has already been saved and blocking the next sweep.

No training is performed by this script.

Examples
--------
    # Validate all commands without starting Isaac Sim
    python outputs/ranged_stiffness/eval_ranged_stiffness.py --dry-run

    # Test one command for one policy
    python outputs/ranged_stiffness/eval_ranged_stiffness.py \
        --policy analytical_moe --stiffness-index 0

    # Run all 56 jobs, resuming complete reports if present
    python outputs/ranged_stiffness/eval_ranged_stiffness.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import selectors
import shlex
import signal
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EVAL_SCRIPT = ROOT / "scripts" / "eval_manipulation.py"
EVAL_RESULTS_DIR = HERE / "eval_results"
MANIFEST_PATH = EVAL_RESULTS_DIR / "ranged_eval_manifest.json"


POLICIES = {
    "analytical_moe": {
        "label": "Analytical MoE",
        "moe_experts_config": ROOT / "cfg" / "moe" / "G1_ee_analytical_200_600.yaml",
    },
    "two_layer_range_v2": {
        "label": "Two-layer ranged HL v2",
        "run_path": "luoxinyuan-duke-university/gentle_humanoid_high_level/ee_xyz_range_200_600_v2_3kp_force_b_stu_adapt_20260802_204248",
        "extra_args": [
            "--full_collision",
            "--objects",
            str(ROOT / "cfg" / "objects" / "boxes_scene.yaml"),
        ],
    },
    "end2end_range": {
        "label": "End-to-end xyz ranged",
        "checkpoint": ROOT / "outputs/2026-08-05/17-48-44-G1GENTLE3KPEEXYZRANGE200600-ppo/wandb/run-20260805_175000-gentle_3kp_ee_xyz_range_200_600_finetune/files/checkpoint_final.pt",
        "config_file": ROOT / "outputs/2026-08-05/17-48-44-G1GENTLE3KPEEXYZRANGE200600-ppo/wandb/run-20260805_175000-gentle_3kp_ee_xyz_range_200_600_finetune/files/cfg.yaml",
    },
    "oracle_force_stiff_low_level": {
        "label": "Oracle force + stiff low-level",
        "run_path": "luoxinyuan-duke-university/gentle_humanoid/gentle_3kp_stiff_finetune_limmt_full_force30",
        "extra_args": [
            "--ee_compliance_oracle_force",
        ],
    },
}


# The six permutations of (250, 400, 550) are included.  Together with the
# six one-axis soft/hard commands and the two isotropic endpoints, this gives
# 14 commands per policy.
STIFFNESS_COMMANDS = [
    ("all_soft", (200, 200, 200)),
    ("x_soft", (200, 600, 600)),
    ("y_soft", (600, 200, 600)),
    ("z_soft", (600, 600, 200)),
    ("mixed_250_400_550", (250, 400, 550)),
    ("mixed_250_550_400", (250, 550, 400)),
    ("mixed_400_250_550", (400, 250, 550)),
    ("mixed_400_550_250", (400, 550, 250)),
    ("mixed_550_250_400", (550, 250, 400)),
    ("mixed_550_400_250", (550, 400, 250)),
    ("x_hard", (600, 200, 200)),
    ("y_hard", (200, 600, 200)),
    ("z_hard", (200, 200, 600)),
    ("all_hard", (600, 600, 600)),
]


def _target_tuple(value):
    if isinstance(value, (int, float)):
        return (float(value),) * 3
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(float(item) for item in value)
    return None


def complete_report(path: Path, expected):
    """Return True only for a complete 60-probe report at the expected target."""
    try:
        with path.open() as handle:
            payload = json.load(handle)
        target = payload.get("ee_compliance_target", {}).get("stiffness")
        return (
            len(payload.get("records", [])) == 60
            and _target_tuple(target) == tuple(float(item) for item in expected)
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False


def command_for(policy_key, stiffness, output_path, num_envs):
    policy = POLICIES[policy_key]
    command = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--ee_compliance_eval",
        "--ee_compliance_num_envs",
        str(num_envs),
        "--ee_compliance_stiffness",
        *(str(value) for value in stiffness),
        "--ee_compliance_force_deadband",
        "0.0",
        "--ee_output",
        str(output_path),
    ]
    if "moe_experts_config" in policy:
        command.extend(["--moe_experts_config", str(policy["moe_experts_config"])])
    elif "run_path" in policy:
        command.extend(["--run_path", policy["run_path"]])
    else:
        command.extend(["--checkpoint", str(policy["checkpoint"])])
        command.extend(["--config_file", str(policy["config_file"])])
    command.extend(policy.get("extra_args", []))
    return command


def validate_policy_sources(policy_key):
    policy = POLICIES[policy_key]
    required = [EVAL_SCRIPT]
    if "moe_experts_config" in policy:
        required.append(policy["moe_experts_config"])
    elif "run_path" in policy:
        pass
    else:
        required.extend([policy["checkpoint"], policy["config_file"]])
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing source files for {policy_key}:\n" + "\n".join(missing))


def write_manifest(entries):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w") as handle:
        json.dump(entries, handle, indent=2)


def send_group_signal(process, sig):
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def run_one(job, grace_seconds, shutdown_timeout, dry_run=False):
    command = job["command"]
    output_path = Path(job["output"])
    log_path = Path(job["log"])
    print("\n" + "=" * 80)
    print(f"[{job['policy_key']}] {job['stiffness_name']} {job['stiffness']}")
    print("$ " + shlex.join(command))
    if dry_run:
        return "dry-run"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        report_seen_at = None
        interrupt_sent_at = None
        terminate_sent_at = None

        while True:
            for key, _ in selector.select(timeout=0.25):
                line = key.fileobj.readline()
                if line:
                    print(f"[{job['policy_key']}] {line}", end="")
                    log_handle.write(line)
                    log_handle.flush()

            if report_seen_at is None:
                report_is_new = output_path.exists() and output_path.stat().st_mtime >= start_time - 1.0
                if report_is_new and complete_report(output_path, job["stiffness"]):
                    report_seen_at = time.monotonic()
                    print(f"[{job['policy_key']}] complete 60-probe report detected")

            now = time.monotonic()
            if report_seen_at is not None and interrupt_sent_at is None:
                if now - report_seen_at >= grace_seconds and process.poll() is None:
                    print(f"[{job['policy_key']}] report saved; sending SIGINT for clean stop")
                    send_group_signal(process, signal.SIGINT)
                    interrupt_sent_at = now

            if interrupt_sent_at is not None and process.poll() is None:
                if now - interrupt_sent_at >= shutdown_timeout and terminate_sent_at is None:
                    print(f"[{job['policy_key']}] SIGINT timeout; sending SIGTERM")
                    send_group_signal(process, signal.SIGTERM)
                    terminate_sent_at = now
                elif terminate_sent_at is not None and now - terminate_sent_at >= 5.0:
                    print(f"[{job['policy_key']}] SIGTERM timeout; sending SIGKILL")
                    send_group_signal(process, signal.SIGKILL)

            if process.poll() is not None:
                break

        selector.close()
        returncode = process.wait()

    valid = complete_report(output_path, job["stiffness"])
    if valid:
        if returncode not in (0, -signal.SIGINT, 130):
            print(f"[{job['policy_key']}] report valid; child exit code was {returncode}")
        return "complete"
    print(f"[{job['policy_key']}] failed: no complete report; exit code {returncode}")
    return "failed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        action="append",
        choices=sorted(POLICIES),
        help="Policy to run; repeat the option. Defaults to all four.",
    )
    parser.add_argument(
        "--stiffness-index",
        type=int,
        action="append",
        help="Command index to run; repeat the option. Defaults to all 14.",
    )
    parser.add_argument("--num-envs", type=int, default=8, help="EE evaluator environment count")
    parser.add_argument("--grace-seconds", type=float, default=1.5, help="Wait after report detection before SIGINT")
    parser.add_argument("--shutdown-timeout", type=float, default=12.0, help="Seconds to wait after SIGINT")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching Isaac Sim")
    parser.add_argument("--no-resume", action="store_true", help="Rerun jobs even when a complete JSON exists")
    args = parser.parse_args()

    if args.num_envs != 8:
        # Keep command construction explicit and easy to audit.  The value is
        # passed through by replacing the fixed default after construction.
        pass
    policy_keys = args.policy or list(POLICIES)
    stiffness_indices = args.stiffness_index or list(range(len(STIFFNESS_COMMANDS)))
    for index in stiffness_indices:
        if index < 0 or index >= len(STIFFNESS_COMMANDS):
            parser.error(f"--stiffness-index must be in [0, {len(STIFFNESS_COMMANDS) - 1}]")
    for policy_key in policy_keys:
        validate_policy_sources(policy_key)

    jobs = []
    for policy_key in policy_keys:
        for index in stiffness_indices:
            stiffness_name, stiffness = STIFFNESS_COMMANDS[index]
            token = "_".join(str(value) for value in stiffness)
            output_path = EVAL_RESULTS_DIR / f"{policy_key}_{stiffness_name}_k{token}.json"
            jobs.append(
                {
                    "policy_key": policy_key,
                    "policy": POLICIES[policy_key]["label"],
                    "stiffness_index": index,
                    "stiffness_name": stiffness_name,
                    "stiffness": stiffness,
                    "output": str(output_path),
                    "log": str(EVAL_RESULTS_DIR / f"{policy_key}_{stiffness_name}_k{token}.log"),
                    "command": command_for(policy_key, stiffness, output_path, args.num_envs),
                    "status": "planned",
                }
            )

    write_manifest(jobs)
    print(f"Planned jobs: {len(jobs)}")
    print(f"Output directory: {EVAL_RESULTS_DIR}")
    print(f"Manifest: {MANIFEST_PATH}")

    for job in jobs:
        output_path = Path(job["output"])
        if not args.no_resume and complete_report(output_path, job["stiffness"]):
            job["status"] = "skipped_existing"
            print(f"Skipping complete report: {output_path.name}")
            write_manifest(jobs)
            continue
        job["status"] = run_one(job, args.grace_seconds, args.shutdown_timeout, args.dry_run)
        write_manifest(jobs)

    counts = {}
    for job in jobs:
        counts[job["status"]] = counts.get(job["status"], 0) + 1
    print("\nSweep summary:")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")
    if counts.get("failed", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
