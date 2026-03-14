# Legacy folder

This directory contains archived or non-primary files that were cluttering the repository root.

## Contents

- `test_overfit.py`
  - original monolithic reference script used during early development and debugging.
- `templates.py`
  - older prompt helper tied to a previous token format.
- `collators.py`
  - older collator utility no longer used by the main pipelines.
- `install_artifacts/`
  - stray installation output files that do not belong in the main source tree.

## Guidance

Keep these files only if you still use them for comparison, debugging, or recovery. Otherwise, review `docs/cleanup_candidates.md` and remove the ones you no longer need.
