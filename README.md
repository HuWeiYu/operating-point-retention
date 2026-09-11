# Operating-Point Retention

Code for continual industrial anomaly detection with operating-point evaluation. The repository includes an INP-Former continual AD implementation and experiments with Shared-FT, EWC, Shared-LoRA, Subspace-LoRA, and replay buffers of 0/10/30/100 images per class.

## Install

Use Python 3.11 with the dependencies in `requirements.txt`, or create the provided Conda environment:

```bash
conda env create -f environment.yml
conda activate inp-opret
# or: pip install -r requirements.txt
```

Training requires a CUDA-capable PyTorch installation. The metric and analysis utilities can be used on CPU.

## Data and checkpoint

Prepare the MVTec AD dataset yourself. The loader expects the standard MVTec directory layout. Download the DINOv2 checkpoint yourself and place it at `_weights/dinov2_vitb14_reg4_pretrain.pth` (or use the model loader's configured download path).

This repository does not contain datasets, model weights, checkpoints, score bundles, or logs.

## Train

Run commands from the repository root. The default stream is `configs/streams/mvtec_order_0.json`.

```bash
python scripts/train_continual.py --method shared_ft \
  --replay-buffer-size 0 --data-path /path/to/mvtec \
  --output-dir ./runs/shared_ft_replay0

python scripts/train_continual.py --method ewc \
  --replay-buffer-size 10 --data-path /path/to/mvtec \
  --output-dir ./runs/ewc_replay10

python scripts/train_continual.py --method shared_lora \
  --replay-buffer-size 30 --data-path /path/to/mvtec \
  --output-dir ./runs/shared_lora_replay30

python scripts/train_continual.py --method subspace_lora \
  --replay-buffer-size 100 --data-path /path/to/mvtec \
  --output-dir ./runs/subspace_lora_replay100
```

Use the YAML files in `configs/` for method and replay settings. `--validate-only` checks command configuration without starting training:

```bash
python scripts/train_continual.py --method shared_ft \
  --replay-buffer-size 0 --validate-only
```

## Evaluate

Evaluate a completed run with:

```bash
python scripts/evaluate.py \
  --run-dir ./runs/shared_ft_replay0 \
  --output-dir ./results/shared_ft_replay0
```

For offline aggregation and operating-point sensitivity analysis, see `scripts/recompute_metrics.py`, `scripts/run_sensitivity.py`, and `scripts/reproduce_fig3.py`.

## Tests

```bash
pytest -q
```

In the release environment, `pytest` and `torch` must be installed before running the full test suite and training validation path. A lightweight syntax check is:

```bash
python -m compileall -q scripts src tests
```

See `THIRD_PARTY_NOTICES.md` for bundled third-party code and license information.
