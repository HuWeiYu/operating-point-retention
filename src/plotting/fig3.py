"""Render the four-panel operating-point figure (paper Fig.3).

Panels
  (a) Ranking        -- old-task vs current-task image-AUROC vs buffer.
  (b) Stored decisions -- old-task FPR and FNR at the stored per-task tau_t.
  (c) FPR sensitivity -- FPR vs delta = (tau_t + delta*sigma_t), delta in [-2,4].
  (d) FNR sensitivity -- FNR vs delta, same grid.

Inputs are two CSVs (paths passed by the caller, never hardcoded):
  * ``agg_csv`` -- per-buffer aggregate with columns from :mod:`src.analysis.aggregate`
    (``old_current_i_auroc_mean[_std]``, ``current_task_i_auroc_mean``,
    ``old_fpr_fixed_mean[_std]``).
  * ``sens_csv`` -- per-(buffer, delta) FPR/FNR, columns ``buffer, delta, fpr, fnr``
    (+ optional ``_std``).

The figure is written to ``out_pdf`` (and a sibling PNG). No publish dir is touched.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                     "xtick.labelsize": 6.4, "ytick.labelsize": 6.4,
                     "legend.fontsize": 5.5, "figure.constrained_layout.use": True})
COLORS = {"old": "#1f4e79", "cur": "#c00000", "buffer10": "#2e75b6", "buffer100": "#548235"}
BREAK_DELTAS = [0.0, 4.0]
XMIN, XMAX = -2.0, 4.0   # the paper's displayed delta window


def _load(path: Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"missing input CSV: {p}")
    return pd.read_csv(p)


def _col(df, *names):
    """Return the first present column among ``names`` (so both ``fpr`` and ``fpr_mean`` work)."""
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"none of {names} present in {list(df.columns)}")


def _buf_pen(buf):
    return COLORS["buffer10"] if int(buf) == 10 else COLORS["buffer100"]


def _draw_panel_a(ax, agg):
    xs = agg["buffer_size"].to_numpy()
    old = agg["old_current_i_auroc_mean"].to_numpy() * 100
    cur = agg["current_task_i_auroc_mean"].to_numpy() * 100
    old_sd = agg["old_current_i_auroc_std"].to_numpy() * 100 \
        if "old_current_i_auroc_std" in agg else None
    ax.errorbar(xs, old, yerr=old_sd, marker="o", ms=3.5, lw=1.4, capsize=2,
                color=COLORS["old"], label="old tasks (1-4)")
    ax.plot(xs, cur, marker="s", ms=3.5, lw=1.4, color=COLORS["cur"], label="current task 5")
    ax.set_xscale("log", base=10)
    ax.set_xticks(xs); ax.set_xticklabels([str(int(x)) for x in xs])
    ax.set_xlabel("replay buffer (images/category)")
    ax.set_ylabel("image-AUROC (%)")
    ax.set_ylim(45, 101); ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("(a) ranking")


def _draw_panel_b(ax, agg):
    xs = agg["buffer_size"].to_numpy()
    fpr = agg["old_fpr_fixed_mean"].to_numpy() * 100
    fpr_sd = agg["old_fpr_fixed_std"].to_numpy() * 100 if "old_fpr_fixed_std" in agg else None
    # FNR is not in the formal aggregate CSV; it is passed in separately when available.
    ax.errorbar(xs, fpr, yerr=fpr_sd, marker="o", ms=3.5, lw=1.4, capsize=2,
                color=COLORS["buffer10"], label="FPR (old)")
    if "old_fnr_fixed_mean" in agg:
        fnr = agg["old_fnr_fixed_mean"].to_numpy() * 100
        ax.plot(xs, fnr, marker="^", ms=3.5, lw=1.4, color=COLORS["buffer100"], label="FNR (old)")
    ax.set_xscale("log", base=10); ax.set_xticks(xs); ax.set_xticklabels([str(int(x)) for x in xs])
    ax.set_xlabel("replay buffer (images/category)")
    ax.set_ylabel("FPR / FNR at stored $\\tau_t$ (%)")
    ax.set_ylim(-2, 105); ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("(b) stored decisions")


def _draw_panel_c(ax, sens):
    fpr_col = _col(sens, "fpr", "fpr_mean")
    for buf, grp in sens.groupby("buffer"):
        grp = grp.sort_values("delta")
        ax.plot(grp["delta"], grp[fpr_col] * 100, marker="o", ms=2, lw=1.2,
                color=_buf_pen(buf), label=f"buf {int(buf)}")
    ax.axvline(BREAK_DELTAS[0], color="k", ls=":", lw=0.7)
    ax.axvline(BREAK_DELTAS[1], color="k", ls=":", lw=0.7)
    ax.set_xlim(XMIN, XMAX); ax.set_ylim(-2, 105); ax.grid(alpha=0.25)
    ax.set_xlabel("$\\delta$  (threshold = $\\tau_t + \\delta\\cdot\\sigma_t$)")
    ax.set_ylabel("FPR (%)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("(c) FPR sensitivity")


def _draw_panel_d(ax, sens):
    fnr_col = _col(sens, "fnr", "fnr_mean")
    for buf, grp in sens.groupby("buffer"):
        grp = grp.sort_values("delta")
        ax.plot(grp["delta"], grp[fnr_col] * 100, marker="o", ms=2, lw=1.2,
                color=_buf_pen(buf), label=f"buf {int(buf)}")
    ax.axvline(BREAK_DELTAS[0], color="k", ls=":", lw=0.7)
    ax.axvline(BREAK_DELTAS[1], color="k", ls=":", lw=0.7)
    ax.set_xlim(XMIN, XMAX); ax.set_ylim(-2, 105); ax.grid(alpha=0.25)
    ax.set_xlabel("$\\delta$  (threshold = $\\tau_t + \\delta\\cdot\\sigma_t$)")
    ax.set_ylabel("FNR (%)")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("(d) FNR sensitivity")


def make_fig3(agg_csv, sens_csv=None, out_pdf="fig3_operating_point.pdf",
              width_in=7.0, height_in=2.0, dpi=300) -> None:
    """Build the 1x4 figure from ``agg_csv`` (+ optional ``sens_csv``)."""
    agg = _load(agg_csv)
    fig, axes = plt.subplots(1, 4, figsize=(width_in, height_in), gridspec_kw={"wspace": 0.28})
    _draw_panel_a(axes[0], agg)
    _draw_panel_b(axes[1], agg)
    if sens_csv is not None and Path(sens_csv).exists():
        sens = _load(sens_csv)
        _draw_panel_c(axes[2], sens)
        _draw_panel_d(axes[3], sens)
    else:
        axes[2].axis("off"); axes[3].axis("off")
    out = Path(out_pdf)
    fig.savefig(out, bbox_inches="tight", dpi=dpi)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"wrote {out} (+ .png)")
