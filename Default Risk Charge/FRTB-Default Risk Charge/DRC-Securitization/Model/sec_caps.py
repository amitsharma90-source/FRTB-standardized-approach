"""
sec_caps.py - Risk-weight floors and caps for sec DRC (CRE40.50-51, 45.4,
41.15/42.7/44.26 — all RISK-WEIGHT level — plus MAR22.34(3) FV cap downstream).

  apply_floors_caps: per-tranche RW wrap chain
       look-through cap (40.50-51, senior only) ->
         NPL floor (45.4) -> standard floor (41.15 / 42.7 / 44.26)

These are all RISK-WEIGHT adjustments, which MAR22.34(1) imports verbatim
("the default risk weights ... as set out in CRE40 to CRE44").

NOT here: the CRE40.51-55 aggregated cap (K_P * P). That is a banking-book
"maximum CAPITAL requirement", not a risk weight, so MAR22.34(1) does NOT
import it (it scopes the import to risk weights). The only capital cap in the
trading book is MAR22.34(3)'s per-position fair-value cap, applied at tranche
level in sec_engine.step_10_bucket_hbr. See FRTB SA/CLAUDE.md "Caps stack" for
the full reasoning. The aggregated-cap implementation was removed accordingly.

All regulatory scalars (8% factor, floors, look-through recipe) are sourced
from SEC_Constants / SEC_Floors. No hardcoding.
"""
from __future__ import annotations
import pandas as pd

from . import sec_npl


# ── FLOORS ────────────────────────────────────────────────────────────────────

def lookup_standard_floor(
    floors_table: pd.DataFrame, approach: str, is_senior: bool, is_stc: bool
) -> float:
    """Pick the standard floor from SEC_Floors sheet.

    Returns the floor as a decimal (0.15 = 15%).
    If no row matches, returns 0.0 (no floor).
    """
    variant = "STC" if is_stc else "Standard"
    seniority = "Senior" if is_senior else "Non-senior"
    matches = floors_table[
        (floors_table["Variant"] == variant)
        & (floors_table["Approach"] == approach)
        & (floors_table["Seniority"].isin([seniority, "Any"]))
    ]
    if matches.empty:
        return 0.0
    return float(matches.iloc[0]["Floor_RW_decimal"])


# ── LOOK-THROUGH CAP (CRE40.50-51) ────────────────────────────────────────────

def _rw_factor_inv(config: dict) -> float:
    return 1.0 / float(
        config["SEC_Constants"].set_index("Constant").at["rw_factor_to_capital", "Value"]
    )


def compute_pool_avg_rw(row: pd.Series, rw_factor_inv: float) -> float:
    """Exposure-weighted average RW of the underlying pool, per CRE40.50.

    For an SA pool, EW-avg RW = K_SA * (1/0.08) = K_SA * 12.5.
    For an IRB pool, EW-avg RW = K_IRB * (1/0.08).
    For a mixed pool we use a K_IRB-weighted-by-share approximation; with
    only homogeneous-pool deals in the current portfolio this is exact for
    pure-IRB and pure-SA pools.

    Returns the cap RW as a decimal (e.g. 1.00 = 100%).
    """
    K_SA = float(row.get("K_SA", 0.0) or 0.0)
    K_IRB = row.get("K_IRB")
    K_IRB = float(K_IRB) if (K_IRB is not None and not pd.isna(K_IRB)) else None
    irb_share = float(row.get("irb_modelable_share", 0.0))

    if irb_share >= 1.0 and K_IRB is not None:
        K_pool = K_IRB
    elif irb_share > 0 and K_IRB is not None:
        K_pool = irb_share * K_IRB + (1.0 - irb_share) * K_SA
    else:
        K_pool = K_SA
    return K_pool * rw_factor_inv


def apply_lookthrough_cap(positions: pd.DataFrame, config: dict) -> pd.DataFrame:
    """CRE40.50-51: cap RW(senior) at the EW-avg RW of the underlying pool.

    Eligibility (CRE40.51):
      - is_senior_derived = True
      - continuous_look_through_known = True (Deal_Master flag)
    For ineligible rows the cap is a no-op.

    Reads df['rw'] and mutates it in place when the cap binds. Eligibility
    and cap RW are recorded in audit-flag columns; the transformation is
    transcribed into rw_details by apply_floors_caps at the end of the
    wrapper chain.

    Columns added (audit only):
      lookthrough_cap_eligible : bool
      lookthrough_cap_rw       : float (decimal; NaN when ineligible)
      lookthrough_cap_binds    : bool
    """
    df = positions.copy()
    rw_factor_inv = _rw_factor_inv(config)

    eligible = (
        df["is_senior_derived"].fillna(False).astype(bool)
        & df["continuous_look_through_known"].fillna(False).astype(bool)
    )
    cap_rw = df.apply(
        lambda r: compute_pool_avg_rw(r, rw_factor_inv) if eligible.at[r.name] else float("nan"),
        axis=1,
    )
    rw_current = df["rw"]
    df["lookthrough_cap_eligible"] = eligible
    df["lookthrough_cap_rw"] = cap_rw
    df["lookthrough_cap_binds"] = eligible & (rw_current > cap_rw)
    # Promote post-cap value into rw
    df["rw"] = rw_current.where(~df["lookthrough_cap_binds"], cap_rw)
    return df


# ── FLOORS + CAPS CHAIN ───────────────────────────────────────────────────────

def _transcript_for_row(
    existing_details: str,
    rw_pre_wrappers: float,
    rw_post_lookthrough: float,
    rw_post_npl: float,
    rw_final: float,
    standard_floor: float,
    row: pd.Series,
) -> str:
    """Append a human-readable wrapper-stack transcript to the rw_details
    string produced by the approach module in step 7. Result is the full
    per-row audit trail of how rw evolved from approach output to final
    floored/capped value.
    """
    def _fmt(x):
        if pd.isna(x):
            return "NaN"
        return f"{float(x):.6f}"

    lines = [str(existing_details), "", "WRAPPER STACK (phase 4):",
             f"  rw_from_approach_module = {_fmt(rw_pre_wrappers)}"]

    # 1. Look-through cap (CRE40.50-51)
    if bool(row.get("lookthrough_cap_eligible", False)):
        cap_rw = row.get("lookthrough_cap_rw", float("nan"))
        if bool(row.get("lookthrough_cap_binds", False)):
            lines.append(
                f"  CRE40.50-51 look-through cap: BINDS at {_fmt(cap_rw)} "
                f"-> rw={_fmt(rw_post_lookthrough)}"
            )
        else:
            lines.append(
                f"  CRE40.50-51 look-through cap: eligible but does not bind "
                f"(cap={_fmt(cap_rw)}, rw={_fmt(rw_post_lookthrough)})"
            )
    else:
        lines.append("  CRE40.50-51 look-through cap: not eligible -> rw unchanged")

    # 2. NPL layer (CRE45.5 concession; CRE45.4 100% floor)
    if bool(row.get("npl_concession_applied", False)):
        lines.append(
            f"  CRE45.5 NRPPD senior concession: APPLIED -> rw set to {_fmt(rw_post_npl)}"
        )
    elif bool(row.get("npl_floor_applied", False)):
        lines.append(
            f"  CRE45.4 NPL 100% floor: APPLIED -> rw={_fmt(rw_post_npl)}"
        )
    elif bool(row.get("is_npl_derived", False)):
        lines.append(
            "  NPL deal but no NPL wrapper bound (CRE45.4 / 5 not applicable to this approach)"
        )

    # 3. Standard floor (CRE41.15 / 42.7 / 44.26)
    if standard_floor > 0:
        if not pd.isna(rw_post_npl) and rw_post_npl < standard_floor:
            lines.append(
                f"  CRE41.15 / 42.7 / 44.26 standard floor: {_fmt(standard_floor)} "
                f"-> BINDS, rw raised from {_fmt(rw_post_npl)}"
            )
        else:
            lines.append(
                f"  CRE41.15 / 42.7 / 44.26 standard floor: {_fmt(standard_floor)} "
                f"-> does not bind"
            )

    lines.append(f"  rw_final = {_fmt(rw_final)}")
    return "\n".join(lines)


def apply_floors_caps(positions: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Wrap rw with the regulatory chain:
       look-through cap (CRE40.50-51)
         -> NPL concession + 100% floor (CRE45.4 / 45.5)
           -> standard floor (CRE41.15 / 42.7 / 44.26).

    Single 'rw' column carries the current value through each wrapper;
    rw_details is augmented with a per-row transformation transcript so
    every intermediate value is auditable without separate columns.
    """
    df = positions.copy()
    rw_pre_wrappers = df["rw"].copy()  # snapshot from step 7

    # 1. Look-through cap (mutates df['rw'])
    df = apply_lookthrough_cap(df, config)
    rw_post_lookthrough = df["rw"].copy()

    # 2. NPL layer (mutates df['rw']; sets npl_concession_applied / npl_floor_applied)
    df = sec_npl.apply_npl_layer(df, config)
    rw_post_npl = df["rw"].copy()

    # 3. Standard floor lookup per (approach, seniority, STC variant)
    floors = config["SEC_Floors"]
    rw_final = []
    floor_used = []
    for _, row in df.iterrows():
        rw_pre = row.get("rw", float("nan"))
        approach = row.get("approach", "")
        is_senior = bool(row.get("is_senior_derived", False))
        is_stc = bool(row.get("is_stc_compliant", False))
        floor = lookup_standard_floor(floors, approach, is_senior, is_stc)
        if pd.isna(rw_pre):
            rw_final.append(float("nan"))
        else:
            rw_final.append(max(float(rw_pre), floor))
        floor_used.append(floor)
    df["standard_floor_applied"] = floor_used
    df["rw"] = rw_final  # final post-wrapper-stack value

    # 4. Append wrapper-stack transcript to rw_details
    df["rw_details"] = [
        _transcript_for_row(
            existing_details=df.at[idx, "rw_details"],
            rw_pre_wrappers=rw_pre_wrappers.at[idx],
            rw_post_lookthrough=rw_post_lookthrough.at[idx],
            rw_post_npl=rw_post_npl.at[idx],
            rw_final=df.at[idx, "rw"],
            standard_floor=df.at[idx, "standard_floor_applied"],
            row=df.loc[idx],
        )
        for idx in df.index
    ]
    return df


# ── AGGREGATED CAP (CRE40.51-55, K_P*P) — DELIBERATELY NOT IMPLEMENTED ─────────
#
# The banking-book "maximum aggregated capital requirement" K_P*P (CRE40.53) is
# a CAPITAL cap, not a risk weight. MAR22.34(1) imports CRE40-44 only for the
# RISK WEIGHTS ("the default risk weights ... as set out in CRE40 to CRE44"),
# and MAR22.34(3) provides the trading book's own (and only) capital cap — the
# per-position fair-value cap, applied at tranche level in
# sec_engine.step_10_bucket_hbr. Importing K_P*P as well would (a) exceed the
# scope of 22.34(1), (b) make 22.34(3) redundant, and (c) have nowhere to
# attach in the MAR22.33/35 bucket-HBR aggregation. So it is intentionally
# absent. (Removed 2026-06-01 after a critical re-read of MAR22.34; previously
# this engine wrongly applied it, which understated sec DRC by ~$6.1m on v5.)
