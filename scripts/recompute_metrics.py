#!/usr/bin/env python3
"""Re-aggregate the replay cohort into per-run / by-buffer / paper-table CSVs.

Wraps :mod:`src.analysis.aggregate` over a set of run directories. Tier B: reads saved
``stage_metrics.csv`` + ``drift_metrics.csv`` only (CPU, no training). Run dirs are given
explicitly (or via ``--cohort`` + ``--root``); nothing is hardcoded.

Usage
-----
    python scripts/recompute_metrics.py --root .. \\
        --cohort configs/cohort.json --out <out-dir>        # resolve cohort names
    python scripts/recompute_metrics.py --run <dir> ... --out <out-dir>   # explicit dirs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.aggregate import collect_runs, by_buffer, paper_table  # noqa: E402


def _run_dirs(args, root: Path):
    if args.run:
        return [root / Path(r) if not Path(r).is_absolute() else Path(r) for r in args.run]
    cohort = Path(args.cohort)
    if not cohort.exists():
        raise FileNotFoundError(args.cohort)
    names = json.loads(cohort.read_text(encoding="utf-8"))["run_names"]
    return [root / n for n in names]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".", help="run-tree root")
    ap.add_argument("--run", action="append", dest="run", help="explicit run dir(s)")
    ap.add_argument("--cohort", default="configs/cohort.json", help="cohort JSON (if no --run)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    writes = [out / "replay_multiseed_per_run.csv", out / "replay_multiseed_by_buffer.csv",
              out / "replay_multiseed_paper_table.csv"]
    if not args.overwrite:
        clash = [w for w in writes if w.exists()]
        if clash:
            raise SystemExit("refusing to overwrite (pass --overwrite): "
                             + ", ".join(str(x) for x in clash))

    rows, skipped = collect_runs(_run_dirs(args, root))
    if not rows:
        raise SystemExit("no completed runs to aggregate.")
    per_run = __import__("pandas").DataFrame(rows).sort_values(["buffer_size", "run"])
    by_buf = by_buffer(rows)
    tbl = paper_table(by_buf)

    per_run.to_csv(writes[0], index=False)
    by_buf.to_csv(writes[1], index=False)
    tbl.to_csv(writes[2], index=False)

    print(f"aggregated {len(rows)} runs; skipped {len(skipped)}")
    for s in skipped:
        print(f"  [skip] {s['run_dir']}: {s['reason']}")
    for w in writes:
        print(f"  wrote {w}")
    print("\n" + tbl.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
