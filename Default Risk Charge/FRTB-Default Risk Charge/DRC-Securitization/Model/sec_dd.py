"""
sec_dd.py — Step 1: due-diligence gate (CRE40.31-36).

Reads dd_status from the loader output (sourced from Position_Overrides,
default 'pass'). The actual 1250% routing happens in sec_hierarchy when
dd_status != 'pass'. This module exists to (a) document where the gate
lives and (b) provide a hook for future expansion (CRE40.32-34 substantive
DD content checks: pool-performance data, structural-feature dossiers).

Currently passive — the gate is enforced by sec_hierarchy.route_position
step 1. When the data team wires up substantive DD checks, replace
check_dd_substantive() with the real validation.
"""
from __future__ import annotations
import pandas as pd


def check_dd_substantive(row: pd.Series) -> tuple[bool, str]:
    """Return (passed, reason). Stub: assumes 'pass' tag is sufficient.

    Future extension hooks per CRE40.32-34:
      - dd_last_refresh_date staleness check
      - pool_perf_data_completeness vs the named-fields list (% 30/60/90 dpd,
        default rates, prepayment, foreclosure, occupancy, credit score, LTV,
        diversification)
      - structural_features_dossier_complete check
      - resecuritisation: nested look-through completeness
    """
    return True, "stub - dd_status tag treated as authoritative"


def is_due_diligence_pass(row: pd.Series) -> bool:
    """Truth function: dd_status == 'pass' AND substantive DD passes."""
    tag_ok = str(row.get("dd_status", "pass")).lower() == "pass"
    sub_ok, _ = check_dd_substantive(row)
    return tag_ok and sub_ok
