"""Compatibility entry point for the offline ALFWorld experiment."""

from __future__ import annotations

import runpy
import warnings


if __name__ == "__main__":
    warnings.warn(
        "ProcedureMem.alfworld_run is deprecated; use "
        "`python -m ProcedureMem.run_memp_offline`.",
        DeprecationWarning,
        stacklevel=1,
    )
    runpy.run_module("ProcedureMem.run_memp_offline", run_name="__main__")
