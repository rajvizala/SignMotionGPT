"""Compatibility entrypoint for held-out dataset evaluation."""

from evaluation.test_dataset_eval import *  # noqa: F401,F403
from evaluation.test_dataset_eval import main


if __name__ == "__main__":
    main()
