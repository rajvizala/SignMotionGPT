from __future__ import annotations

import argparse
import json
import os
import subprocess
from typing import List


DEFAULT_CONFIGS = [
    "agent_experiments/configs/baseline_repro.toml",
    "agent_experiments/configs/exp_a_rdrop_sam.toml",
    "agent_experiments/configs/exp_b_rdrop_sam_swa.toml",
    "agent_experiments/configs/exp_c_add_replay_ewc.toml",
    "agent_experiments/configs/exp_d_full_recipe.toml",
]


def _run(cmd: List[str]) -> int:
    print("[Run]", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment matrix across configs and seeds.")
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS, help="List of TOML configs.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44], help="Seed list.")
    parser.add_argument("--extra-set", action="append", default=[], help="Extra override section.key=value")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    args = parser.parse_args()

    for config in args.configs:
        base_name = os.path.splitext(os.path.basename(config))[0]
        for seed in args.seeds:
            cmd = [
                "python",
                "-m",
                "agent_experiments.experiments.train_sentence_generalization",
                "--config",
                config,
                "--set",
                f"data.seed={seed}",
                "--set",
                f"data.output_dir=./agent_experiments/outputs/{base_name}_seed_{seed}",
            ]
            for kv in args.extra_set:
                cmd.extend(["--set", kv])
            if args.dry_run:
                print(" ".join(cmd))
                continue
            code = _run(cmd)
            if code != 0:
                raise SystemExit(f"Run failed: config={config}, seed={seed}, code={code}")


if __name__ == "__main__":
    main()

