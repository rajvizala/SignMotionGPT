# Deprecated Files

These files are no longer used in the active training pipeline and are kept for reference only.

- `templates.py` - Legacy prompt templates using `<MOT_BEGIN>/<MOT_END>` format. Imports a non-existent `ids_to_motion_specials` function. Not imported anywhere.
- `collators.py` - Legacy `AssistantSpanCollator` for chat-format label masking. Not imported anywhere.
- `test_overfit.py` - One-off reference script used during early Colab development. Contains hardcoded paths and HF tokens.

These files can be safely deleted if no longer needed for reference.
