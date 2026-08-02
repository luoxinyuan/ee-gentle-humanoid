#!/usr/bin/env python3
"""Run the analytical EE MoE compliance evaluator over an xyz stiffness grid."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys


def _stiffness_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moe_experts_config", "--moe-experts-config", required=True)
    parser.add_argument(
        "--levels",
        type=float,
        nargs="+",
        default=[200.0, 300.0, 400.0, 500.0, 600.0],
        help="Stiffness values used independently for kx, ky, and kz.",
    )
    parser.add_argument("--num_envs", "--num-envs", type=int, default=1)
    parser.add_argument("--output_dir", "--output-dir", default="outputs/ee_moe_xyz_grid")
    parser.add_argument("--skip_existing", "--skip-existing", action="store_true")
    parser.add_argument("--dry_run", "--dry-run", action="store_true")
    parser.add_argument(
        "--max_cases",
        "--max-cases",
        type=int,
        default=None,
        help="Optional prefix limit for smoke tests.",
    )
    args = parser.parse_args()

    if not args.levels:
        parser.error("--levels must contain at least one stiffness value.")
    if any(value < 200.0 or value > 600.0 for value in args.levels):
        parser.error("All --levels values must be within [200, 600].")
    if args.num_envs < 1:
        parser.error("--num_envs must be positive.")

    repo_root = Path(__file__).resolve().parents[1]
    evaluator = repo_root / "scripts" / "eval_manipulation.py"
    experts_config = Path(args.moe_experts_config).expanduser().resolve()
    if not experts_config.is_file():
        parser.error(f"MoE experts config does not exist: {experts_config}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = list(itertools.product(args.levels, repeat=3))
    if args.max_cases is not None:
        cases = cases[: max(args.max_cases, 0)]

    records = []
    for case_index, (kx, ky, kz) in enumerate(cases, start=1):
        token = "_".join(_stiffness_token(value) for value in (kx, ky, kz))
        report_path = output_dir / f"k_{token}.json"
        command = [
            sys.executable,
            os.fspath(evaluator),
            "--moe_experts_config",
            os.fspath(experts_config),
            "--ee_compliance_eval",
            "--ee_compliance_num_envs",
            str(args.num_envs),
            "--ee_compliance_stiffness",
            str(kx),
            str(ky),
            str(kz),
            "--ee_output",
            os.fspath(report_path),
        ]

        if args.skip_existing and report_path.is_file():
            status = "skipped"
        elif args.dry_run:
            print(" ".join(command))
            status = "dry_run"
        else:
            print(
                f"[{case_index}/{len(cases)}] evaluating stiffness "
                f"[{kx:g}, {ky:g}, {kz:g}]",
                flush=True,
            )
            subprocess.run(command, cwd=repo_root, check=True)
            status = "completed"

        records.append(
            {
                "stiffness": [kx, ky, kz],
                "report": os.fspath(report_path),
                "status": status,
            }
        )

    index_path = output_dir / "grid_index.json"
    with index_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "moe_experts_config": os.fspath(experts_config),
                "levels": args.levels,
                "num_cases": len(cases),
                "records": records,
            },
            file,
            indent=2,
        )
    print(f"Grid index: {index_path}")


if __name__ == "__main__":
    main()
