#!/usr/bin/env python3
"""Train the INP-Former expert under a continual stream (paper Table I / Fig.3 runs).

This is the *training* entrypoint. It wraps :mod:`src.continual.continual_grouped` — the
full 5-stage stream trainer that implements shared-FT, EWC, shared-LoRA, subspace-LoRA and
experience replay — exposing the paper's CLI and adding ``--validate-only`` (parse + build
the run config without launching a training run) and ``--config`` (seed defaults from a YAML
param set, e.g. ``configs/shared_ft.yaml`` or ``configs/replay_10.yaml``).

Requirements
------------
* A CUDA GPU. ``run()`` raises ``RuntimeError`` if no CUDA device is present.
* MVTec AD data (``--data-path`` pointing at the dataset root that contains ``bottle/``,
  ``cable/``, ... ``wood/``) and the DINOv2 checkpoint in ``_weights/`` (see README).
* ``--method`` must be one of ``shared_ft | ewc | shared_lora | subspace_lora``
  (the full parser also accepts the other upstream methods; see ``--help``).

Usage
-----
    python scripts/train_continual.py --method shared_ft    --replay-buffer-size 0  \\
        --data-path /path/to/mvtec --output-dir ./runs/shared_ft_replay0
    python scripts/train_continual.py --method ewc          --replay-buffer-size 0  \\
        --data-path /path/to/mvtec --output-dir ./runs/ewc_replay0
    python scripts/train_continual.py --method shared_lora  --replay-buffer-size 0  \\
        --data-path /path/to/mvtec --output-dir ./runs/shared_lora_replay0
    python scripts/train_continual.py --method subspace_lora --replay-buffer-size 0 \\
        --data-path /path/to/mvtec --output-dir ./runs/subspace_lora_replay0

    # replay variants (paper Fig.3 cohort: shared-FT, buffer 0/10/30/100)
    python scripts/train_continual.py --method shared_ft    --replay-buffer-size 10 \\
        --data-path /path/to/mvtec --output-dir ./runs/shared_ft_replay10

    # seed defaults from a config, then override on the command line
    python scripts/train_continual.py --config configs/subspace_lora.yaml \\
        --replay-buffer-size 0 --data-path /path/to/mvtec --output-dir ./runs/sub

    # parse + build the run config only (no GPU/data needed)
    python scripts/train_continual.py --method shared_ft --replay-buffer-size 0 \\
        --validate-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml  # noqa: E402

from src.continual.continual_grouped import (  # noqa: E402
    build_parser,
    config_from_namespace,
    run,
)

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_STREAM = _REPO / "configs" / "streams" / "mvtec_order_0.json"
_ALLOWED_TRAIN_METHODS = ("shared_ft", "ewc", "shared_lora", "subspace_lora")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    # --config: seed parser defaults from a YAML param set (CLI still overrides).
    parser.add_argument("--config", default=None,
                        help="YAML param set (e.g. configs/shared_ft.yaml); CLI flags override it.")
    parser.add_argument("--validate-only", action="store_true",
                        help="parse and build the run config, then exit without training.")
    known_keys = {a.dest for a in parser._actions}
    if "--config" in argv:
        cfg_file = Path(argv[argv.index("--config") + 1]).resolve()
        if not cfg_file.exists():
            raise SystemExit(f"--config not found: {cfg_file}")
        values = (yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {})
        parser.set_defaults(**{k: v for k, v in values.items() if k in known_keys})

    # Continual_grouped declares --stream as required; default it to the bundled order-0
    # stream when the user does not pass one (so the README commands need not name it).
    if "--stream" not in argv:
        parser.set_defaults(stream=str(_DEFAULT_STREAM))
        argv += ["--stream", str(_DEFAULT_STREAM)]

    args = parser.parse_args(argv)
    if args.method not in _ALLOWED_TRAIN_METHODS:
        parser.error(
            f"--method {args.method!r} is not one of the paper's train methods "
            f"({', '.join(_ALLOWED_TRAIN_METHODS)})."
        )
    config = config_from_namespace(args, parser)
    if args.validate_only:
        print(f"[validate-only] method={args.method} replay_buffer_size="
              f"{args.replay_buffer_size} data_path={args.data_path}")
        print(f"[validate-only] stream={args.stream} encoder={args.encoder} "
              f"seed={args.seed} inp_num={args.inp_num} epochs={args.epochs} — run config OK")
        return 0
    run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
