"""Compatibility entry point for the online ALFWorld experiment."""

from __future__ import annotations

import runpy
import warnings


if __name__ == "__main__":
    warnings.warn(
        "ProcedureMem.alfworld_run_update is deprecated; use "
        "`python -m ProcedureMem.run_memp_online`.",
        DeprecationWarning,
        stacklevel=1,
    )
    runpy.run_module("ProcedureMem.run_memp_online", run_name="__main__")
