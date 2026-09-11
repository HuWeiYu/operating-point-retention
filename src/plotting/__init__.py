"""Plotting for the operating-point audit (matplotlib, reads CSVs, writes a PDF/PNG).

This subpackage renders the audit figure from already-computed aggregate/sensitivity
CSVs. It never recomputes metrics and never writes to a publish directory -- the
output path is a caller argument.
"""
from .fig3 import make_fig3

__all__ = ["make_fig3"]
