# Results Template for Each Experiment

Use this exact structure for every run (`seed x config`) to keep comparisons reliable.

## Run metadata

- Config file:
- Commit hash:
- GPU:
- Effective batch size:
- Wall clock training time:
- Notes:

## Final scalar metrics

- Best checkpoint step:
- Validation average loss:
- Validation worst-group loss:
- Test DTW-JPE:
- Test DTW-PA-JPE:
- Test FID:
- Test diversity:
- Test multimodality:

## Group breakdown (validation)

Copy `group_stats` from `train_history.json` for the best step:

- motion_length_bucket:
- text_length_bucket:
- lexical_richness_bucket:
- participant:

## Qualitative checks

List at least 20 unseen test sentences, and include:

- Input sentence
- Ground-truth motion summary (brief)
- Generated motion summary (brief)
- Failure mode category (length drift, handshape instability, timing drift, semantic mismatch, other)

## Interpretation

### What improved?

### What regressed?

### Is improvement robust across seeds?

### Does this look like true OOD gain or in-domain overfitting?

## Decision

- Promote to next ablation stage: yes/no
- If no, next change:
- If yes, frozen settings:

