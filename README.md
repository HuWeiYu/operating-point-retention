# Beyond AUROC: Auditing Operating-Point Retention in Continual Industrial Anomaly Detection

Operating-point (the alarm threshold) is a first-class object in continual industrial anomaly
detection, not a fixed side effect of AUROC. This repository contains (a) an INP-Former
continual-training implementation, (b) stored-threshold FPR/FNR and image-AUROC evaluation,
and (c) offline, data-light tools for operating-point sensitivity analysis from saved run
artifacts.

> **Scope note.** This repository ships the training and evaluation code, but not trained
> checkpoints, raw per-image score bundles, or MVTec data. The training path has not been
> validated end-to-end on GPU in this checkout, so exact numerical reproduction is not
> guaranteed. The `src/metrics` / `src/analysis` / `src/plotting` stack can process compatible
> saved run artifacts offline on CPU.

---

## 1. What's in the repo

```
scripts/             runnable entry points
  train_continual.py   online training (needs GPU + data)
  evaluate.py          offline per-run stored-FPR/FNR + AUROC from a run dir
  recompute_metrics.py offline aggregation of replay runs
  run_sensitivity.py   offline operating-point delta sweep
  reproduce_fig3.py    offline figure render
src/
  continual/   continual_grouped.py (5-stage trainer: shared-FT/EWC/LoRA/subspace+replay),
                inflora_adaptation.py (subspace-LoRA internals)
  models/      INP-Former expert, DINOv2 encoder wrapper, vit blocks
  data/        MVTec dataset + transforms
  metrics/     operating-point metric primitives (pure NumPy)
  analysis/    run-artifact loaders + aggregation (no training)
  plotting/    operating-point analysis plots
  optimizers/  StableAdamW
  backbones/   vendored DINOv1 / DINOv2 / BEiT (see THIRD_PARTY_NOTICES)
configs/       stream + method/replay parameter sets
tests/         pytest suite (metric + CLI dispatch + replay loader)
results/       generated outputs (empty in repo)
```

## 2. Install

```
conda env create -f environment.yml        # or: pip install -r requirements.txt
conda activate inp-opret
```

Two tiers, one env: training needs a **CUDA GPU**; the offline `src/metrics`/`src/analysis`/
`src/plotting` path is pure NumPy/pandas and runs on CPU.

## 3. Data preparation (MVTec AD + DINOv2)

The repo ships **no data and no checkpoints** — download them yourself (licenses: MVTec AD is
CC BY-NC-SA 4.0; DINOv2 weights released by Meta).

1. Download MVTec AD. The loader expects the standard layout under a single root:
   `<data-path>/<category>/train/good/*.png`, `.../test/<defect>/*.png`,
   `.../ground_truth/<defect>/*.png` for each of bottle, cable, carpet, capsule, hazelnut, grid,
   metal_nut, pill, leather, screw, toothbrush, tile, transistor, zipper, wood.
2. Put the DINOv2-REG base/14 checkpoint at `_weights/dinov2_vitb14_reg4_pretrain.pth`
   (`src/models/vit_encoder.py` resolves `_weights/` relative to the repo root). It can also be
   auto-downloaded from the Meta URL if left absent.

## 4. Train

All training commands run from the repo root. `--stream` defaults to the bundled order-0 stream
(`configs/streams/mvtec_order_0.json`).

```
python scripts/train_continual.py --method shared_ft    --replay-buffer-size 0  \
    --data-path /path/to/mvtec --output-dir ./runs/shared_ft_replay0
python scripts/train_continual.py --method ewc          --replay-buffer-size 0  \
    --data-path /path/to/mvtec --output-dir ./runs/ewc_replay0
python scripts/train_continual.py --method shared_lora  --replay-buffer-size 0  \
    --data-path /path/to/mvtec --output-dir ./runs/shared_lora_replay0
python scripts/train_continual.py --method subspace_lora --replay-buffer-size 0 \
    --data-path /path/to/mvtec --output-dir ./runs/subspace_lora_replay0
```

`--validate-only` parses + builds the run config and exits (no GPU/data needed):

```
python scripts/train_continual.py --method shared_ft --replay-buffer-size 0 --validate-only
```

## 5. The four update methods

| `--method` | Update rule | Key hyper-parameters (config) |
|---|---|---|
| `shared_ft` | fine-tune the shared decoder/bottleneck | `configs/shared_ft.yaml` |
| `ewc` | shared FT + EWC penalty on the expert | `--ewc-lambda`, `--fisher-batches` (`configs/ewc.yaml`) |
| `shared_lora` | shared LoRA branches on the decoder | `--lora-rank`, `--lora-alpha`, `--lora-dropout` (`configs/shared_lora.yaml`) |
| `subspace_lora` | per-task LoRA in a cumulative subspace | `--subspace-lambda-start/end`, lora params (`configs/subspace_lora.yaml`) |

`run()` dispatches on `config.method`; the other upstream methods (`isolation`, `isolated_lora`,
`isolated_head`, `lwf`) remain available in the code but are not covered by the bundled configs.

## 6. Replay (buffer 0/10/30/100)

Replay is a per-class budget of prior *normal* images replayed at each new stage. The bundled
replay configurations use shared-FT with buffer sizes 0, 10, 30, and 100 (images/category):

```
python scripts/train_continual.py --method shared_ft --replay-buffer-size 10 \
    --data-path /path/to/mvtec --output-dir ./runs/shared_ft_replay10
# ... 30 -> --replay-buffer-size 30 ; 100 -> --replay-buffer-size 100
```

`--replay-selector` is `random` (uniform) or `score_anchor` (keep top-K hardest normals under
the about-to-be-overwritten state). Config sets live in `configs/replay_{0,10,30,100}.yaml`.

## 7. Evaluate and operating-point metrics

`evaluate.py` recomputes, from a trained run dir, the stored-threshold FPR/FNR (Eq.1: at the
acquisition threshold `tau_t`, never recalibrated) and image-level AUROC per old task + the
final task:

```
python scripts/evaluate.py --run-dir ./runs/shared_ft_replay10 \
    --output-dir ./results/shared_ft_replay10
```

Metric primitives (`src/metrics/operating_point.py`) are pure NumPy: `tau_sigma`,
`threshold_at`, `metrics` (strict `score > tau`), `image_auroc` (rank-based, ties averaged),
`feasible_delta`, `standardized_scores`.

## 8. Offline operating-point analysis

Recomputation is data-light: it reads *saved per-image score bundles* and `stage_metrics.csv`,
never re-runs the model, and needs no GPU.

```
python scripts/recompute_metrics.py --root <run-tree> --cohort configs/cohort.json --out results/agg
python scripts/run_sensitivity.py     --root <run-tree> --cohort configs/cohort.json --grid configs/delta_grid.json --out results/sens
python scripts/reproduce_fig3.py     --agg results/agg/replay_multiseed_by_buffer.csv \
    --sens results/sens/op_sensitivity_replay_macro.csv --out results/fig3_operating_point.pdf
```

The `delta` sweep moves the threshold to `tau_t + delta*sigma_t` (equivalent to the rule
`z > delta` in standardized score `z = (score - tau_t)/sigma_t`). `delta = 0` is the stored
operating point and should match the corresponding `FPR@fixed`/`FNR@fixed` calculation. A
common `delta` is **not** a common absolute shift — each task has its own `sigma_t`.

## 9. Expected output

- Training writes per-`stage_SS/` artifacts (`train.json` with the stored threshold, per-image
  `task_tt_*.npz` bundles, `validation_metrics.csv`, `stage_metrics.csv`) and `config.json`.
- `evaluate.py` → `evaluation_metrics.csv` (per-task stored FPR/FNR + AUROC).
- `recompute_metrics.py` → per-run and by-buffer CSVs for the supplied run cohort.
- `run_sensitivity.py` → `op_sensitivity_by_run.csv`, `op_sensitivity_replay_macro.csv`,
  `op_sensitivity_delta0_check.csv`; `delta=0` FPR/FNR equal the formal values to ~1e-17.
- `reproduce_fig3.py` → `fig3_operating_point.pdf` (+ .png), the 1x4 figure.

## 10. GPU requirement

- **Training** (`train_continual.py`) requires a CUDA GPU; `run()` raises `RuntimeError` without one.
- **Offline** (`evaluate.py`, `recompute_metrics.py`, `run_sensitivity.py`, `reproduce_fig3.py`)
  are CPU-only.

## 11. Reproducibility boundaries

- **Exact benchmark numbers.** They depend on GPU runs, random seeds, PyTorch builds, and
  DINOv2 checkpoint revisions. The same training setup may produce equivalent-but-not-identical
  results because of seeded RNG, floating-point, and driver differences. This repository does
  not include trained checkpoints.
- **Raw score bundles** consumed by the offline tools are not shipped; run training or supply
  compatible bundles to generate them.
- Cross-stack runs may differ in FPR/AUROC because of the above environment differences.

## 12. Protocol, limitations, and provenance (condensed)

The bundled order-0 stream uses 5 stages x 3 categories;
`inp_num=6`, `epochs=50`, `batch_size=8`, `lr=1e-3`→`1e-4`, `validation_fraction=0.1`,
`threshold_quantile=0.99`; `tau_t` is stored at acquisition and never recalibrated; `delta=0` is
the stored operating point. Stored-threshold metrics use a
strict `score > tau` rule and image-level AUROC; pixel-level AUROC is a diagnostic, not the
reported quantity. One known protocol asymmetry: a severity gate uses a *stage-wide* threshold
while per-task `tau_t` is computed per run. `src/analysis` applies the per-task convention.
Limitations: the operating-point analysis is limited to the bundled streams and methods; no
fresh real-world generalization claim is made here.

## 13. Citation

See `CITATION.cff` for repository citation metadata. If you use this code, cite this repository
and the upstream projects listed in `THIRD_PARTY_NOTICES.md`.

## 14. License and third-party code

**License is pending author confirmation** — see `LICENSE`. The repo mixes the MIT INP-Former
derivative work, vendored Apache-2.0 DINOv2, and code written for this release; nothing in the
repo is licensed as a unified whole yet. Third-party components are not owned here; see
`THIRD_PARTY_NOTICES.md`. MVTec data is CC BY-NC-SA 4.0 and is **not** redistributed. The UCAD
baseline is **not** included because its upstream has no explicit license; it is cited only:
`https://github.com/jiaq-liu/UCAD.git` (HEAD `c7f075e`).
