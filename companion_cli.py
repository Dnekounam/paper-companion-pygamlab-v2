#!/usr/bin/env python3
"""Convenience CLI for the PyGamLab V2 paper companion repository."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MPLCONFIGDIR = ROOT / ".mplconfig"

TASKS = {
    "ai-workflow": {
        "script": ROOT / "ai_workflow_benchmark" / "compare_workflow.py",
        "group": "benchmarks",
        "description": "Compare the PyGamLab regression workflow against scikit-learn.",
    },
    "data-analysis": {
        "script": ROOT / "data_analysis_benchmark" / "compare_analysis.py",
        "group": "benchmarks",
        "description": "Compare PyGamLab laboratory-analysis functions against SciPy/NumPy references.",
    },
    "structures": {
        "script": ROOT / "structures_benchmark" / "compare_structures.py",
        "group": "benchmarks",
        "description": "Compare PyGamLab structure generators against ASE references.",
    },
    "gam-ai-protocol": {
        "script": ROOT / "gam_ai_protocol" / "create_cte_gam_ai.py",
        "group": "supplementary",
        "description": "Recreate the representative CTE .gam_ai training and packaging workflow.",
    },
    "gam-ai-details": {
        "script": ROOT / "gam_ai_details" / "extract_gam_ai_details.py",
        "group": "supplementary",
        "description": "Extract documentation details for all packaged .gam_ai models.",
    },
    "figure6": {
        "script": ROOT / "figure6_creation" / "create_figure6.py",
        "group": "supplementary",
        "description": "Recreate the Figure 6 chemical-congress plotting workflow.",
    },
}

ALIASES = {
    "benchmarks": [name for name, task in TASKS.items() if task["group"] == "benchmarks"],
    "supplementary": [name for name, task in TASKS.items() if task["group"] == "supplementary"],
    "all": list(TASKS.keys()),
}


def build_env(pygamlab_source: str | None) -> dict[str, str]:
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(MPLCONFIGDIR)
    if pygamlab_source:
        source = str(Path(pygamlab_source).resolve())
        current = env.get("PYTHONPATH")
        env["PYTHONPATH"] = source if not current else source + os.pathsep + current
    return env


def selected_tasks(target: str) -> list[str]:
    if target in TASKS:
        return [target]
    if target in ALIASES:
        return ALIASES[target]
    raise KeyError(f"Unknown task target: {target}")


def print_task_list() -> None:
    print("Available tasks:\n")
    for group_name in ("benchmarks", "supplementary"):
        print(f"{group_name}:")
        for name in ALIASES[group_name]:
            print(f"  {name:<18} {TASKS[name]['description']}")
        print()
    print("aggregates:")
    for name in ("benchmarks", "supplementary", "all"):
        print(f"  {name}")


def run_task(name: str, env: dict[str, str]) -> None:
    task = TASKS[name]
    script = task["script"]
    print(f"\n[{name}] {task['description']}")
    subprocess.run([sys.executable, str(script)], cwd=str(ROOT), env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run benchmarks and supplementary workflows for the PyGamLab V2 paper companion."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List all available tasks.")

    run_parser = subparsers.add_parser("run", help="Run one task or a task group.")
    run_parser.add_argument(
        "target",
        choices=list(TASKS.keys()) + list(ALIASES.keys()),
        help="Task name or grouped target.",
    )
    run_parser.add_argument(
        "--pygamlab-source",
        help="Optional path to a local PyGamLab source checkout to prepend to PYTHONPATH.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "list":
        print_task_list()
        return

    env = build_env(args.pygamlab_source)
    for name in selected_tasks(args.target):
        run_task(name, env)


if __name__ == "__main__":
    main()
