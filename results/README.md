# Results

`results/` is where the *outputs* of this release land when you run `evaluate.py`,
`recompute_metrics.py`, `run_sensitivity.py`, or `reproduce_fig3.py`. It is empty in the
repo (no artifacts are shipped) and is created on demand.

- `evaluate.py --output-dir ./results/<run>` writes `evaluation_metrics.csv` (per-task stored
  FPR/FNR + image AUROC for a trained run dir).
- `recompute_metrics.py --out ./results/...` writes `replay_multiseed_per_run.csv`,
  `replay_multiseed_by_buffer.csv`, `replay_multiseed_paper_table.csv`.
- `run_sensitivity.py --out ./results/...` writes the delta-sensitivity sweep outputs.
- `reproduce_fig3.py --out fig3.pdf` writes the 1x4 operating-point figure (+ .png).

Nothing here is committed; these are generated at runtime.
