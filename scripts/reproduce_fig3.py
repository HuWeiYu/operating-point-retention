#!/usr/bin/env python3
"""Re-render the operating-point audit figure (paper Fig.3) from stored CSVs.

Reads a per-buffer aggregate CSV (from :mod:`src.analysis.aggregate`) and the delta
sensitivity macro CSV, then renders the 1x4 figure via :func:`src.plotting.fig3.make_fig3`.
Pure plotting: no metrics are recomputed here and no publish directory is touched.

Usage
-----
    python scripts/reproduce_fig3.py --agg <agg.csv> --sens <sens.csv> --out fig.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.plotting.fig3 import make_fig3  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agg", required=True, help="per-buffer aggregate CSV")
    ap.add_argument("--sens", default=None, help="delta sensitivity macro CSV (optional)")
    ap.add_argument("--out", required=True, help="output PDF path (.png written alongside)")
    ap.add_argument("--width", type=float, default=7.0, help="figure width in inches")
    ap.add_argument("--height", type=float, default=2.0, help="figure height in inches")
    ap.add_argument("--overwrite", action="store_true", help="replace existing output")
    args = ap.parse_args()

    out = Path(args.out)
    if (out.exists() or out.with_suffix(".png").exists()) and not args.overwrite:
        raise SystemExit("refusing to overwrite an existing figure (pass --overwrite): "
                         f"{out} / {out.with_suffix('.png')}")

    make_fig3(args.agg, args.sens, out, width_in=args.width, height_in=args.height)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
