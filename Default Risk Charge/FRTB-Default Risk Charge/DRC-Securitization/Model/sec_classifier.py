"""
sec_classifier.py - Step 3a: pool IRB-modelability per CRE40.15-17.

Adds two columns to the loader frame:
  bank_has_irb_for_assetclass : bool  - from Bank_Capabilities lookup
  irb_modelable_share         : float - fraction of the underlying pool for
                                        which the bank can compute IRB
                                        parameters (K_IRB, PD, LGD, EAD).

Regulatory meaning (CRE40.15-17):
  - irb_modelable_share = 1.0 : IRB pool (CRE40.15) - bank can model ALL
                                underlying exposures
  - 0 < share < 1            : Mixed pool (CRE40.16) - bank can model SOME
  - irb_modelable_share = 0  : SA pool (CRE40.17) - bank cannot model any
                                of the underlying (no IRB approval / no
                                loan-level data / supervisor prohibits)

A separate categorical 'pool_classification' column was previously emitted
('IRB' / 'MIXED' / 'SA'), but those value labels collide visually with the
SEC-IRBA / SEC-SA approach names downstream and the categorical carried no
information beyond the numeric share itself. Routing logic
(sec_hierarchy.route_position) now reads irb_modelable_share directly:

  irba_eligible = irb_share >= mixed_pool_threshold   # CRE40.42 / 40.46

Operational rule for irb_modelable_share (current implementation):
  - investor on third-party deal:      0.0  (no loan-level data on third-party pools)
  - originator/sponsor + IRB approval: 1.0  (own deal, full data)
  - originator/sponsor without IRB:    0.0

This is a simplification: in reality a mixed pool with partial IRB coverage
could give 0 < share < 1, but for the v5 portfolio (all investor) the
distinction collapses to 0 vs 1. When mixed-pool deals enter the book,
extend compute_irb_modelable_share() to read a per-deal share field.
"""
from __future__ import annotations
import os
import pandas as pd

from .sec_loader import (
    build_sec_loader_frame,
    write_snapshot as _loader_snapshot,
    OUTPUT_DIR,
)


# Auxiliary file written only by this module's standalone main(); the canonical
# phase 2 output is `drc_sec_phase2_classify_route.xlsx` produced by sec_engine.
OUTPUT_SNAPSHOT = os.path.join(OUTPUT_DIR, "_aux_drc_sec_classifier.xlsx")


def lookup_bank_irb(bank_capabilities: pd.DataFrame, asset_class: str) -> bool:
    """Look up has_irb_for_assetclass for an asset class.

    Returns False if the asset class is unknown to Bank_Capabilities
    (conservative - unknown asset class implies no IRB approval).
    """
    bc = bank_capabilities.set_index("Asset_class")
    if asset_class not in bc.index:
        return False
    return bool(bc.at[asset_class, "has_irb_for_assetclass"])


def compute_irb_modelable_share(row: pd.Series, bank_has_irb: bool) -> float:
    """Effective fraction of pool for which K_IRB is computable per CRE40.15-17."""
    is_orig_or_sponsor = bool(row.get("is_originator_or_sponsor_deal", False))
    if not (is_orig_or_sponsor and bank_has_irb):
        return 0.0
    return 1.0


def classify_pools(
    positions: pd.DataFrame, bank_capabilities: pd.DataFrame
) -> pd.DataFrame:
    """Add bank_has_irb_for_assetclass + irb_modelable_share to the loader frame."""
    df = positions.copy()
    df["bank_has_irb_for_assetclass"] = df["Underlying Pool Type"].apply(
        lambda ac: lookup_bank_irb(bank_capabilities, ac)
    )
    df["irb_modelable_share"] = df.apply(
        lambda r: compute_irb_modelable_share(r, r["bank_has_irb_for_assetclass"]),
        axis=1,
    )
    return df


# ── SUMMARY / SNAPSHOT ────────────────────────────────────────────────────────

def summarise(df: pd.DataFrame) -> str:
    lines = []
    lines.append("Pool IRB-modelability breakdown (CRE40.15-17):")
    lines.append(df["irb_modelable_share"].value_counts().to_string())
    lines.append("")
    lines.append("Per-deal IRB-modelability (one row per Pool ID):")
    deal_view = (
        df.groupby("Pool ID")
        .agg(
            pool_type=("pool_type", "first"),
            asset_class=("Underlying Pool Type", "first"),
            bank_has_irb=("bank_has_irb_for_assetclass", "first"),
            is_originator=("is_originator_or_sponsor_deal", "first"),
            irb_share=("irb_modelable_share", "first"),
            n_positions=("ID", "count"),
        )
        .reset_index()
    )
    lines.append(deal_view.to_string(index=False))
    return "\n".join(lines)


def write_snapshot(df: pd.DataFrame, path: str = OUTPUT_SNAPSHOT) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    deal_view = (
        df.groupby("Pool ID")
        .agg(
            pool_type=("pool_type", "first"),
            asset_class=("Underlying Pool Type", "first"),
            bank_has_irb=("bank_has_irb_for_assetclass", "first"),
            is_originator=("is_originator_or_sponsor_deal", "first"),
            irb_share=("irb_modelable_share", "first"),
            n_positions=("ID", "count"),
        )
        .reset_index()
    )
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        df.to_excel(xl, sheet_name="positions", index=False)
        deal_view.to_excel(xl, sheet_name="by_deal", index=False)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

def main() -> None:
    sec, cfg = build_sec_loader_frame()
    classified = classify_pools(sec, cfg["Bank_Capabilities"])
    print(summarise(classified))
    write_snapshot(classified)
    print(f"\nSnapshot written: {os.path.normpath(OUTPUT_SNAPSHOT)}")


if __name__ == "__main__":
    main()
