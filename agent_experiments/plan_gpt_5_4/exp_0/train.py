from __future__ import annotations

from .config import build_parser, from_args
from .model import build_recipe

from agent_experiments.plan_gpt_5_4.shared.llm_runner import run_llm_experiment


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = from_args(args)
    run_llm_experiment(config, build_recipe())


if __name__ == "__main__":
    main()
