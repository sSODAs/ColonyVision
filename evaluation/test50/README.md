# Test50 evaluation set

This folder contains the manual point annotations and sanitized manifest used for the final held-out evaluation.

- 50 held-out Petri-dish images.
- One manual point per visible colony.
- 3,698 manual points total; 3,650 fall inside the fixed counting ROI used by the final evaluation.
- The 50 frozen photographs are included under `raw/test50/` through Git LFS.
- `selection.csv` uses repository-relative paths to those Test50 images.
- `reference_metrics.csv` records the final frozen evaluation summary.

Primary point-matching tolerance: 0.5 × the per-plate median baseline colony diameter.
