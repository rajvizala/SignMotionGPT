"""Compatibility entrypoint for text-to-motion inference."""

from inference.inference import *  # noqa: F401,F403
from inference.inference import main


if __name__ == "__main__":
    main()
