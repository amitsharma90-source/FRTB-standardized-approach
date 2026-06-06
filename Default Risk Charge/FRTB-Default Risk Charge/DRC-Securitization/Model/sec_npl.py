"""
sec_npl.py — NPL securitisation layer (CRE45).

Two responsibilities:
  1. F-IRB block check (CRE45.3) — already wired via sec_hierarchy step 3.
  2. 100% floor on SEC-IRBA / SEC-SA / look-through outputs (CRE45.4).
     SEC-ERBA branch is deliberately omitted from the floor.
  3. Senior 100% concession (CRE45.5): if traditional + senior + NRPPD>=50%,
     assign flat 100% RW (skip the SSFA result entirely).

This module wraps the post-RW output. Step 9 of the build sequence is the
end-to-end NPL test (flip W=0.95 on a fixture deal).
"""
from __future__ import annotations
import pandas as pd


def is_eligible_senior_concession(
    row: pd.Series, npl_concession_approaches: set
) -> bool:
    """CRE45.5 senior 100% concession eligibility check."""
    if not bool(row.get("is_npl_derived", False)):
        return False
    if not bool(row.get("is_senior_derived", False)):
        return False
    if not bool(row.get("is_traditional", True)):
        return False
    nrppd = row.get("nrppd_amount") or 0.0
    pool_out = row.get("pool_outstanding") or 0.0
    if pool_out <= 0:
        return False
    nrppd_ratio = float(nrppd) / float(pool_out)
    if nrppd_ratio < 0.50:
        return False
    if row.get("approach") not in npl_concession_approaches:
        return False
    return True


def apply_npl_layer(positions: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply CRE45 layer wrappers.

    Order:
      1. Senior 100% concession (CRE45.5) - overrides df['rw'] if eligible.
      2. 100% floor (CRE45.4)            - raises df['rw'] to 100% for IRBA/SA/look-through.

    Reads and writes the single 'rw' column (the engine's canonical
    current-RW carrier). Audit flags (npl_concession_applied,
    npl_floor_applied) are written for the transcript downstream.
    """
    df = positions.copy()
    npl_const = config["NPL_Constants"].set_index("Constant")
    senior_concession_rw = float(npl_const.at["npl_senior_concession_rw", "Value"])
    npl_floor_rw = float(npl_const.at["npl_floor_rw", "Value"])

    # Concession-eligible approaches: parse comma-separated string
    concession_approaches_str = str(
        npl_const.at["npl_concession_applies_to_approaches", "Value"]
    )
    concession_approaches = {
        a.strip() for a in concession_approaches_str.split(",")
    }

    # Floor-applicable approaches
    floor_approaches_str = str(
        npl_const.at["npl_floor_applies_to_approaches", "Value"]
    )
    floor_approaches = {a.strip() for a in floor_approaches_str.split(",")}

    df["npl_concession_applied"] = False
    df["npl_floor_applied"] = False

    for idx in df.index:
        row = df.loc[idx]
        # Senior 100% concession
        if is_eligible_senior_concession(row, concession_approaches):
            df.at[idx, "rw"] = senior_concession_rw
            df.at[idx, "npl_concession_applied"] = True
        # 100% floor on IRBA/SA/look-through (NOT ERBA)
        if (
            bool(row.get("is_npl_derived", False))
            and row.get("approach") in floor_approaches
        ):
            current = float(df.at[idx, "rw"]) if pd.notna(df.at[idx, "rw"]) else 0.0
            df.at[idx, "rw"] = max(current, npl_floor_rw)
            df.at[idx, "npl_floor_applied"] = True
    return df
