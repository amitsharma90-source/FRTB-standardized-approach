"""
sec_crm.py — CRM decomposition for protected sec exposures (CRE40.56-65).

STUB at this stage. Pass-through for unprotected positions.

Full implementation (build step 11):
  - For protection_type == 'full_pro_rata':
      Apply standard CRM substitution (CRE22), eligibility per CRE40.56-58.
  - For protection_type == 'tranched':
      Decompose original tranche T into protected sub-tranche + unprotected
      sub-tranche(s). Each sub-tranche treated as a notional standalone tranche
      of the same deal (NOT a resecuritisation; CRE40.59 footnote 5).
      - For SEC-IRBA / SEC-SA paths: recompute A/D per CRE40.60; K stays
        anchored to original pool.
      - For SEC-ERBA path: highest-priority sub-tranche inherits parent RW;
        lower sub-tranches via inferred rating (CRE40.61(2)(a)) or SEC-SA
        with adjusted A/D floored at parent ERBA RW (CRE40.61(2)(b)).
      - Lower-priority sub-tranche always non-senior (CRE40.62), even if parent
        was senior.
"""
from __future__ import annotations
import pandas as pd


def decompose(positions: pd.DataFrame) -> pd.DataFrame:
    """Stub: pass-through. No protected positions in v5 portfolio."""
    df = positions.copy()
    df["crm_decomposed"] = False
    df["crm_subtranche_count"] = 1  # parent only
    if "protection_present" in df.columns:
        protected = df[df["protection_present"].astype(bool)]
        if not protected.empty:
            print(
                f"[sec_crm] WARNING: {len(protected)} protected position(s) "
                f"detected but CRM decomposition is stubbed. "
                f"IDs: {protected['ID'].tolist()}"
            )
    return df
