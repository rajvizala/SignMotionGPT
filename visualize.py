"""Compatibility entrypoint for motion visualization."""

from inference.visualize import *  # noqa: F401,F403
from inference.visualize import main


if __name__ == "__main__":
    main()
