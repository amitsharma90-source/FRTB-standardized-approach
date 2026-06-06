"""
sec_engine.py — Step 4 of build sequence: 10-step orchestrator skeleton.

Implements the chain documented in FRTB SA/CLAUDE.md "Engine 10-step chain":

  1. dd_gate            CRE40.31-36 (handled in routing via dd_status)
  2. spe_op_req_check   CRE40.21 (SPE def), 40.24-25 (op-reqs) (stub: assume pass)
  3. implicit_support   CRE40.49 (data flag from Deal_Master)
  4. pool_classify      CRE40.15-17, 40.46-47 (sec_classifier)
  5. hierarchy_route    CRE40.41-47 (sec_hierarchy)
  6. crm_decompose      CRE40.56-65 (sec_crm — stub)
  7. risk_weight        CRE41-44 + MAR22.34(1) M_T=1 (per-approach module)
  8. wrap_floors_caps   CRE40.50-55 + CRE45.4 + 41.15/42.7/44.26 (sec_caps)
  9a. gross_scaled_jtd  MAR22.27 gross JTD (=signed MV) + MAR22.30 maturity scaling (per position)
  9b. net_jtd           MAR22.28-29 per-tranche offsetting (PURE NETTING, no RW)
  10. bucket_hbr        MAR22.25 RW·net JTD + 22.31-33/35 bucket HBR + DRC + total; FV cap 22.34(3)

Steps 9a / 9b were historically one step (`aggregate_per_deal`). They are
split so the per-position gross + maturity-scaled JTD is materialised and
snapshot-visible (phase 5) BEFORE any netting, and the per-tranche offsetting
(phase 6) reconciles back to it via Σ maturity_scaled_jtd == net_jtd_tranche.
`step_9_net_jtd_per_position` is the thin 9a→9b wrapper for callers that want
the combined transform.

Step 10 applies the risk weight to net JTD per MAR22.25 (rw·8%·net JTD — this
is where RW finally meets net JTD, NOT in the netting phase), aggregates into
MAR22.31/32 buckets (asset class × region), applies the MAR22.23 hedge-benefit
ratio to net-short exposures within each bucket (MAR22.33), computes bucket DRC
(MAR22.25) and the total as a simple sum across buckets (MAR22.35). The HBR
uses the UN-weighted net JTD (MAR22.23(1)); only the capital sums use the
RW-weighted net JTD. The MAR22.34(3) cash-position fair-value cap is applied at
tranche level inside this step. Sec capital is a BUCKET-level quantity; there
is no per-position capital.

Per-tranche netting invariant (engine contract per CLAUDE.md):
    RW is keyed by (Pool ID, A, D, Position Sub-Type) => tranche_key. Two
    positions sharing a key MUST get identical RW. JTDs are netted (signed sum)
    within each tranche key (phase 6); the net JTD is then risk-weighted and
    fed into the bucket-level HBR aggregation (phase 7). Because RW is uniform
    within a tranche, netting-then-weighting == weighting-then-netting.

This skeleton runs end-to-end. Floors and caps apply only when rw is non-null
(NaN means the upstream approach module returned no value, e.g. missing input
data).
"""
from __future__ import annotations
import os
import pandas as pd

from .sec_loader import build_sec_loader_frame, filter_output_columns, OUTPUT_DIR
from .sec_classifier import classify_pools
from .sec_hierarchy import (
    route_approaches,
    APPROACH_IRBA, APPROACH_ERBA, APPROACH_IAA, APPROACH_SA, APPROACH_1250,
)
from . import sec_irba, sec_erba, sec_iaa, sec_sa
from . import sec_dd, sec_caps, sec_crm

OUTPUT_DIR_FALLBACK = OUTPUT_DIR
OUTPUT_PHASE2 = os.path.join(OUTPUT_DIR, "drc_sec_phase2_classify_route.xlsx")
OUTPUT_PHASE3 = os.path.join(OUTPUT_DIR, "drc_sec_phase3_riskweight.xlsx")
OUTPUT_PHASE4 = os.path.join(OUTPUT_DIR, "drc_sec_phase4_floors_caps.xlsx")
OUTPUT_PHASE5 = os.path.join(OUTPUT_DIR, "drc_sec_phase5_gross_scaled_jtd.xlsx")
OUTPUT_PHASE6 = os.path.join(OUTPUT_DIR, "drc_sec_phase6_tranche_net.xlsx")
OUTPUT_PHASE7 = os.path.join(OUTPUT_DIR, "drc_sec_phase7_bucket_hbr.xlsx")
OUTPUT_PHASE8 = os.path.join(OUTPUT_DIR, "drc_sec_phase8_total.xlsx")
# Back-compat: existing callers import OUTPUT_SNAPSHOT.
OUTPUT_SNAPSHOT = OUTPUT_PHASE8


def _rw_to_capital_factor(config: dict) -> float:
    """8% capital factor from SEC_Constants.rw_factor_to_capital (Basel framework)."""
    return float(
        config["SEC_Constants"].set_index("Constant").at["rw_factor_to_capital", "Value"]
    )


def _rw_no_approach(config: dict) -> float:
    """1250% residual RW from SEC_Constants.rw_no_approach (CRE40.41)."""
    return float(
        config["SEC_Constants"].set_index("Constant").at["rw_no_approach", "Value"]
    )


# ── STEPS 1-3: PRE-ROUTING GATES ──────────────────────────────────────────────

def step_1_dd_gate(positions: pd.DataFrame) -> pd.DataFrame:
    """Per CRE40.31, DD failure -> 1250%. Already enforced by sec_hierarchy
    step 1 reading dd_status. This step records sec_dd's substantive verdict
    for the audit trail."""
    df = positions.copy()
    df["dd_substantive_pass"] = df.apply(sec_dd.is_due_diligence_pass, axis=1)
    return df


def step_2_spe_op_req_check(positions: pd.DataFrame) -> pd.DataFrame:
    """SPE / op-req check (CRE40.21 SPE def; 40.24 traditional / 40.25 synthetic op-reqs). Stub: assume pass."""
    df = positions.copy()
    df["spe_op_req_pass"] = True
    return df


def step_3_implicit_support(positions: pd.DataFrame) -> pd.DataFrame:
    """Implicit support flag from Deal_Master (CRE40.49). Engine sets a
    bypass flag; downstream steps respect it (currently informational
    only — full bypass to underlying-asset capital comes in build step 13)."""
    df = positions.copy()
    df["sec_bypass_due_to_implicit_support"] = df["implicit_support_provided"].astype(bool)
    return df


# ── STEPS 4-5: CLASSIFICATION + ROUTING ───────────────────────────────────────

def step_4_pool_classify(positions: pd.DataFrame, bank_capabilities: pd.DataFrame) -> pd.DataFrame:
    return classify_pools(positions, bank_capabilities)


def step_5_hierarchy_route(
    positions: pd.DataFrame, juris_perms: pd.DataFrame, bank_capabilities: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    return route_approaches(positions, juris_perms, bank_capabilities, config)


# ── STEP 6: CRM DECOMPOSITION ─────────────────────────────────────────────────

def step_6_crm_decompose(positions: pd.DataFrame) -> pd.DataFrame:
    return sec_crm.decompose(positions)


# ── STEP 7: RISK WEIGHT (PER-TRANCHE, NOT PER-POSITION) ───────────────────────

APPROACH_DISPATCH = {
    APPROACH_IRBA: sec_irba.compute_rw,
    APPROACH_ERBA: sec_erba.compute_rw,
    APPROACH_IAA: sec_iaa.compute_rw,
    APPROACH_SA: sec_sa.compute_rw,
}


# Position-level attributes that MUST be identical for every position sharing
# the same tranche_key (= same Pool ID + A + D + Position Sub-Type). All come
# either from Deal_Master (per-deal, so trivially equal within a Pool ID) or
# are tranche-level properties one rating-agency / waterfall determines once
# per tranche. A mismatch is a data integrity defect, not a legitimate split.
_TRANCHE_CONSISTENCY_FIELDS = [
    "Rating", "inferred_rating", "is_senior_derived",
    "approach", "dd_status", "is_stc_compliant", "is_resecuritisation",
    "W", "K_SA", "K_IRB", "N", "LGD",
    "M_T_for_drc",            # MAR22.34(1) overlay (=1, for SEC-ERBA + SEC-IRBA RW formulas)
    "is_originator_or_sponsor_deal",
    # NB: M_T_position_years / mat_scaling_factor deliberately NOT in this
    # list - they are position-level (not tranche-level) and can legitimately
    # differ across positions sharing the same (Pool ID, A, D, Position
    # Sub-Type) when those positions have different remaining maturities.
]


def _nan_aware_unique(series: pd.Series) -> list:
    """Return unique values of a Series treating NaN values as equal to each other."""
    vals = series.tolist()
    seen = []
    for v in vals:
        is_nan = isinstance(v, float) and pd.isna(v)
        matched = False
        for s in seen:
            s_is_nan = isinstance(s, float) and pd.isna(s)
            if (is_nan and s_is_nan) or (not is_nan and not s_is_nan and v == s):
                matched = True
                break
        if not matched:
            seen.append(v)
    return seen


def assert_tranche_key_consistency(positions: pd.DataFrame) -> None:
    """Assert that every position sharing a tranche_key agrees on every
    RW-determining attribute. If any group disagrees, raise with a list of the
    offending tranche keys and the column(s) that disagree.

    This is the runtime guard that backs the tranche_key choice: by dropping
    Rating and is_senior_derived from the key (they're operationally derived
    once per tranche) we accept the responsibility of catching the data error
    here rather than silently splitting a tranche in two.
    """
    issues = []
    for tk, group in positions.groupby("tranche_key"):
        if len(group) < 2:
            continue
        for col in _TRANCHE_CONSISTENCY_FIELDS:
            if col not in group.columns:
                continue
            uniques = _nan_aware_unique(group[col])
            if len(uniques) > 1:
                offending_ids = group["ID"].tolist()
                issues.append(
                    f"  tranche_key={tk!r}: '{col}' differs across positions "
                    f"{offending_ids}: values={uniques}"
                )
    if issues:
        raise ValueError(
            "Tranche-key consistency assertion failed - positions sharing the "
            "same (Pool ID, A, D, Position Sub-Type) carry conflicting "
            "RW-determining attributes:\n" + "\n".join(issues) + "\n"
            "Fix the source data so every position on the same tranche shares "
            "the same Rating / approach / Deal_Master attributes."
        )


def step_7_risk_weight(positions: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compute rw ONCE per tranche_key (engine contract), then broadcast
    to all positions sharing that key.

    Output column: 'rw' (the engine's canonical current-RW carrier; this
    phase emits the pre-wrapper value, phase 4 overwrites with the
    post-wrapper-stack final).

    For approach == '1250%' the RW is constant 12.50 (=1250%).
    For other approaches, dispatches to the per-approach module's compute_rw.
    Stubs return NaN; the floor step will leave NaN as-is (NaN propagation).
    """
    df = positions.copy()

    # Data-integrity guard: every position sharing a tranche_key must agree
    # on every RW-determining attribute. Raises before any RW math runs.
    assert_tranche_key_consistency(df)

    # Tranche-level dedup: pick one canonical row per tranche_key for the RW call
    tranche_canonicals = df.drop_duplicates("tranche_key").set_index("tranche_key")

    rw_per_tranche = {}
    details_per_tranche = {}
    rw_1250 = _rw_no_approach(config)
    for tk, row in tranche_canonicals.iterrows():
        approach = row.get("approach", "")
        if approach == APPROACH_1250:
            rw_per_tranche[tk] = rw_1250
            details_per_tranche[tk] = {"approach": "1250%", "source": "CRE40.41 / CRE40.31"}
        elif approach in APPROACH_DISPATCH:
            rw, det = APPROACH_DISPATCH[approach](row, config)
            rw_per_tranche[tk] = rw
            details_per_tranche[tk] = det
        else:
            rw_per_tranche[tk] = float("nan")
            details_per_tranche[tk] = {"approach": approach, "stub": True, "source": "unrouted"}

    # Single 'rw' column carries the current RW at each phase boundary.
    # Phase 3 emits rw = output of the chosen approach module (pure SSFA /
    # table lookup / IAA mapping, before any floor or cap).
    # Phase 4's sec_caps.apply_floors_caps will overwrite rw with the final
    # post-wrapper-stack value and APPEND a transformation transcript to
    # rw_details, so a reviewer can read the wrapper chain row-by-row.
    df["rw"] = df["tranche_key"].map(rw_per_tranche)
    df["rw_details"] = df["tranche_key"].map(details_per_tranche).astype(str)
    return df


# ── STEP 8: FLOORS AND CAPS WRAPPERS ──────────────────────────────────────────

def step_8_wrap_floors_caps(positions: pd.DataFrame, config: dict) -> pd.DataFrame:
    return sec_caps.apply_floors_caps(positions, config)


# ── STEP 9: AGGREGATE PER DEAL (PER-TRANCHE NETTING + AGG CAP) ────────────────

def step_9a_gross_maturity_scaled_jtd(positions: pd.DataFrame) -> pd.DataFrame:
    """Per-position GROSS JTD (MAR22.27) → MATURITY-SCALED JTD (MAR22.30).

    NO netting happens here — this is the per-position foundation that the
    phase-6 tranche offsetting reconciles back to.

    MAR22.27: GROSS JTD for securitisations IS the (signed) market value of
    the exposure — NOT LGD x notional + P&L as for non-sec. LGD is not applied
    to sec exposures because it is already factored into the sec default risk
    weights (CRE41-44); applying it again would double-count. So gross JTD is
    EXACTLY the canonical `MV_USD` column — there is no transform and therefore
    NO separate `jtd_gross` column is materialised (project CLAUDE.md "No
    duplicate columns": a column equal to MV_USD on every row would be a pure
    alias). The phase-5 snapshot labels MV_USD as the gross JTD waypoint.

    MAR22.30 imports MAR22.15-18 verbatim for sec: the gross JTD is scaled by
    mat_scaling_factor = clip(M_T_position_years, 0.25, 1.0). This IS a genuine
    transform — it differs from MV_USD for any sub-1y position — so it gets its
    own canonical column:

      maturity_scaled_jtd = MV_USD * mat_scaling_factor (MAR22.30)

    Naming note (project CLAUDE.md "Meaningful, regulatorily-correct column
    names"): `maturity_scaled_jtd` matches the non-sec phase-2 canonical name
    for the same MAR22.15-18/30 transform; "scaled"/"weighted" alone would be
    ambiguous against the RW multiply that happens later (phase 6).
    """
    df = positions.copy()
    df["maturity_scaled_jtd"] = df["MV_USD"] * df["mat_scaling_factor"]  # MAR22.30
    return df


# Columns that are invariant within a tranche_key (one slice of one loss
# waterfall) and therefore survive the per-tranche collapse. All are guaranteed
# tranche-consistent by assert_tranche_key_consistency() (step 7); per-position
# columns (ID, Issuer, MV_USD, maturity_scaled_jtd, …) are intentionally dropped
# because a collapsed netted row cannot carry a single position's value.
_TRANCHE_LEVEL_COLUMNS = [
    "tranche_key", "Pool ID", "drc_sec_bucket",
    "Attachment Pt (%)", "Detachment Pt (%)",
    "Position Sub-Type", "approach", "is_senior_derived", "Rating", "rw",
    "net_jtd_tranche", "abs_net_jtd_tranche", "abs_mv_tranche",
    "n_positions_in_tranche",
]


def _net_jtd_per_position(positions: pd.DataFrame) -> pd.DataFrame:
    """Per-tranche offsetting (MAR22.28-29) broadcast back onto the per-position
    frame. PURE NETTING — no risk-weighting here.

    Sums the per-position MATURITY-SCALED JTD (from step 9a) within each
    tranche_key — this IS the MAR22.28-29 offsetting of long vs short of the
    same securitisation exposure (identical tranche). Perfect long+short hedges
    on one tranche net to zero here (100% offset).

      per tranche:    net_jtd_tranche = Σ maturity_scaled_jtd within tranche_key

    `rw` stays on the frame as a PASS-THROUGH (it was set in step 7) — it is NOT
    multiplied in here. Per MAR22.25 the risk-weighting of net JTD happens inside
    the bucket DRC calculation (step 10), not in the netting phase. The result is
    kept per-position so `collapse_to_tranche` can derive `abs_mv_tranche` and
    the combined workbook can report at position level.
    """
    df = positions.copy()
    grouped = df.groupby("tranche_key").agg(
        net_jtd_tranche=("maturity_scaled_jtd", "sum"),   # MAR22.28-29 offsetting
        n_positions_in_tranche=("ID", "count"),
    )
    grouped["abs_net_jtd_tranche"] = grouped["net_jtd_tranche"].abs()
    df = df.merge(
        grouped[["net_jtd_tranche", "abs_net_jtd_tranche",
                 "n_positions_in_tranche"]],
        left_on="tranche_key", right_index=True, how="left",
    )
    return df


def collapse_to_tranche(positions_net: pd.DataFrame) -> pd.DataFrame:
    """Project the per-position broadcast frame down to ONE row per tranche_key
    (the netted result), mirroring non-sec `aggregate_net_per_obligor`'s
    per-obligor collapse. Carries only tranche-invariant columns plus the
    tranche's gross fair value `abs_mv_tranche` (= Σ|MV_USD| in the tranche),
    so the row count drops from #positions to #tranches and the netted values
    appear exactly once instead of being broadcast across position rows.
    """
    df = positions_net.copy()
    df["abs_mv_tranche"] = df.groupby("tranche_key")["MV_USD"].transform(
        lambda s: s.abs().sum()
    )
    cols = [c for c in _TRANCHE_LEVEL_COLUMNS if c in df.columns]
    tranche = (
        df.drop_duplicates("tranche_key")[cols]
        .sort_values("abs_net_jtd_tranche", ascending=False)
        .reset_index(drop=True)
    )
    return tranche


def step_9b_net_jtd(positions: pd.DataFrame) -> pd.DataFrame:
    """Per-tranche offsetting (MAR22.28-29) → COLLAPSED per-tranche frame (one
    row per tranche_key). This is phase 6's `20_output`. PURE NETTING — no
    risk-weighting (that is step 10, per MAR22.25). Row count drops from
    #positions to #tranches (no broadcast duplication), exactly as non-sec
    phase 3 collapses per-leg JTD to one row per obligor.

    Conservation invariant (asserted in the phase-6 reconciliation sheet):
    Σ maturity_scaled_jtd over a tranche == that tranche's net_jtd_tranche.
    """
    return collapse_to_tranche(_net_jtd_per_position(positions))


def step_9_net_jtd_per_position(positions: pd.DataFrame) -> pd.DataFrame:
    """Back-compat wrapper: gross + maturity-scaled JTD per position, then
    per-tranche offsetting, returned as a PER-POSITION broadcast frame
    (pre-collapse). Retained for callers that want the combined transform
    (legacy `run_engine`, unit tests)."""
    return _net_jtd_per_position(step_9a_gross_maturity_scaled_jtd(positions))


# ── STEP 10: BUCKET HBR + BUCKET DRC (MAR22.25, 22.31-33, 22.35; FV cap 22.34(3)) ─

def _weight_and_fv_cap(tranche_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply the MAR22.25 risk-weighting (and the MAR22.34(3) FV cap) to the
    per-tranche NET JTD. This is where RW finally meets net JTD — NOT in the
    netting phase. Adds, per tranche:

      rw_weighted_net_jtd = rw · 8% · net_jtd_tranche        (signed; MAR22.25)
      capital_fv_capped   = min(|rw_weighted_net_jtd|, abs_mv_tranche)  (22.34(3))

    `rw` is the per-tranche default risk weight from CRE41-44 (step 7). The FV
    cap is on the |capital|, capped at the tranche's gross fair value; with
    JTD=MV it is inert except at the 1250% RW boundary.
    """
    df = tranche_df.copy()
    cf = _rw_to_capital_factor(config)                          # 8% (CRE/Basel)
    df["rw_weighted_net_jtd"] = df["rw"] * cf * df["net_jtd_tranche"]   # MAR22.25
    df["capital_fv_capped"] = df["rw_weighted_net_jtd"].abs().clip(
        upper=df["abs_mv_tranche"]                              # MAR22.34(3)
    )
    df["fv_cap_binds"] = df["rw_weighted_net_jtd"].abs() > df["abs_mv_tranche"]
    return df


def step_10_bucket_hbr(tranche_df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, float]:
    """Bucket-level DRC for securitisations (non-CTP): MAR22.25 + 22.31-33 + 22.35.

    Input is the COLLAPSED per-tranche frame (phase-6 output): one row per
    tranche_key with `drc_sec_bucket`, `net_jtd_tranche` (maturity-scaled,
    signed; UN-weighted), `rw` (default RW from CRE41-44), and `abs_mv_tranche`.

    Per MAR22.31/32 each exposure sits in one (asset class × region) bucket.
    Within a bucket the DRC requirement (MAR22.33) combines TWO distinct inputs:

      - HBR (MAR22.23) — uses the NET JTD, explicitly **NOT risk-weighted**:
            HBR = Σ_long netJTD / (Σ_long netJTD + Σ_short |netJTD|)
      - the capital sums (MAR22.25) — use the **RW-weighted** net JTD
        (rw·8%·netJTD), FV-capped per MAR22.34(3):
            DRC_b = Σ_long capital_fv − HBR · Σ_short capital_fv,  floored at 0

    RW is applied HERE (step 10), inside the DRC calculation, per MAR22.25 — it
    is not pre-applied in the netting phase. MAR22.35: no cross-bucket hedging,
    total = Σ_b DRC_b. Returns (bucket_df, total_sec_drc).
    """
    df = _weight_and_fv_cap(tranche_df, config)

    rows = []
    for bucket, g in df.groupby("drc_sec_bucket"):
        gl = g[g["net_jtd_tranche"] > 0]
        gs = g[g["net_jtd_tranche"] < 0]
        # HBR on UN-weighted net JTD (MAR22.23(1): "not risk-weighted").
        sum_long_jtd = float(gl["net_jtd_tranche"].sum())
        sum_short_jtd_abs = float(gs["net_jtd_tranche"].abs().sum())
        denom = sum_long_jtd + sum_short_jtd_abs
        hbr = (sum_long_jtd / denom) if denom > 0 else 0.0
        # Capital sums on RW-weighted (FV-capped) net JTD (MAR22.25).
        cap_long = float(gl["capital_fv_capped"].sum())
        cap_short = float(gs["capital_fv_capped"].sum())
        drc_b_raw = cap_long - hbr * cap_short
        drc_b = max(drc_b_raw, 0.0)
        rows.append({
            "drc_sec_bucket": bucket,
            "n_tranches": int(len(g)),
            "n_long_tranches": int(len(gl)),
            "n_short_tranches": int(len(gs)),
            "sum_long_net_jtd": sum_long_jtd,
            "sum_short_net_jtd_abs": sum_short_jtd_abs,
            "hbr": hbr,
            "capital_long": cap_long,
            "capital_short": cap_short,
            "drc_b_raw": drc_b_raw,
            "drc_b": drc_b,
        })
    bucket_df = pd.DataFrame(rows).sort_values("drc_b", ascending=False).reset_index(drop=True)
    total = float(bucket_df["drc_b"].sum())                   # MAR22.35
    return bucket_df, total


# ── ORCHESTRATOR ──────────────────────────────────────────────────────────────

def run_engine(positions: pd.DataFrame, config: dict) -> dict:
    """Run the full sec DRC chain (no snapshots). Returns a result dict:
        {"positions": per-position broadcast frame (through tranche netting),
         "tranches":  collapsed per-tranche frame (phase-6 shape),
         "buckets":   per-bucket DRC frame (phase-7 shape),
         "total":     total sec DRC (MAR22.35, float)}.
    """
    df = step_1_dd_gate(positions)
    df = step_2_spe_op_req_check(df)
    df = step_3_implicit_support(df)
    df = step_4_pool_classify(df, config["Bank_Capabilities"])
    df = step_5_hierarchy_route(df, config["Jurisdiction_Permissions"], config["Bank_Capabilities"], config)
    df = step_6_crm_decompose(df)
    df = step_7_risk_weight(df, config)
    df = step_8_wrap_floors_caps(df, config)
    pos = step_9a_gross_maturity_scaled_jtd(df)
    pos_net = _net_jtd_per_position(pos)                    # per-position netting (no RW)
    tranche = collapse_to_tranche(pos_net)                 # one row per tranche
    bucket_df, total = step_10_bucket_hbr(tranche, config)  # RW + bucket HBR + total
    return {
        "positions": pos_net,
        "tranches": tranche,
        "buckets": bucket_df,
        "total": total,
    }


# ── PHASE-AWARE ORCHESTRATOR + SNAPSHOTS ──────────────────────────────────────

def _phase_snapshot_writer():
    """Lazy import wrapper for phase_snapshot.write_phase_snapshot."""
    import sys
    here = os.path.dirname(__file__)
    parent = os.path.abspath(os.path.join(here, ".."))
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from phase_snapshot import write_phase_snapshot
    return write_phase_snapshot


def write_phase2_snapshot(df_in: pd.DataFrame, df_out: pd.DataFrame,
                          path: str = OUTPUT_PHASE2) -> None:
    """Phase 2 — DD gate + classify + senior derivation + hierarchy routing.

    Bundles engine steps 1-6 (DD, SPE op-req, implicit support, pool classify,
    hierarchy route, CRM decompose) — every step that decides "which approach
    does this position take and is it pre-empted by a gate".
    """
    write = _phase_snapshot_writer()
    by_approach = (
        df_out.groupby("approach").agg(
            n_positions=("ID", "count"),
            n_tranches=("tranche_key", "nunique"),
        ).reset_index()
    )
    classifier_cols = [
        "ID", "Pool ID", "Underlying Pool Type",
        "irb_modelable_share", "is_senior_derived", "Rating",
        "dd_substantive_pass", "spe_op_req_pass",
        "sec_bypass_due_to_implicit_support",
        "approach", "approach_reason",
    ]
    classifier_view = df_out[[c for c in classifier_cols if c in df_out.columns]]
    write(
        path,
        phase_num=2,
        phase_name="DD gate + classify + senior derivation + hierarchy routing",
        mar_ref="CRE40.15-18 (pool/senior), 40.31-36 (DD), 40.41-47 (hierarchy), MAR22.34(1)",
        source_module="drc_securitisation/sec_engine.py [steps 1-6]",
        input_df=filter_output_columns(df_in),
        output_df=filter_output_columns(df_out),
        audit={
            "by_approach": by_approach,
            "classifier_routing_view": classifier_view,
        },
        reconciliation=[
            ("rows_in (phase 1 output)", int(len(df_in))),
            ("rows_out", int(len(df_out))),
            ("dd_failures (->1250%)", int((~df_out["dd_substantive_pass"]).sum())),
            ("implicit_support_bypasses",
                int(df_out["sec_bypass_due_to_implicit_support"].sum())),
            ("approaches_used",
                "; ".join(sorted(df_out["approach"].dropna().unique().tolist()))),
        ],
        notes="No RW math here — only routing decisions. Phase 3 takes the "
              "approach column and dispatches to the per-approach RW module.",
    )


def write_phase3_snapshot(df_in: pd.DataFrame, df_out: pd.DataFrame,
                          path: str = OUTPUT_PHASE3) -> None:
    """Phase 3 — Risk weight per approach (CRE41/42/43/44, M_T=1)."""
    write = _phase_snapshot_writer()
    tranche_view = (
        df_out.drop_duplicates("tranche_key")[
            ["tranche_key", "Pool ID", "approach", "is_senior_derived",
             "Rating", "Attachment Pt (%)", "Detachment Pt (%)",
             "rw", "rw_details"]
        ]
    )
    n_nan = int(df_out["rw"].isna().sum())
    write(
        path,
        phase_num=3,
        phase_name="Risk weight per approach (pre wrappers)",
        mar_ref="CRE41 (SA), CRE42 (ERBA), CRE43 (IAA), CRE44 (IRBA); MAR22.34(1) M_T=1 for ERBA",
        source_module="drc_securitisation/sec_engine.py [step 7]",
        input_df=filter_output_columns(df_in),
        output_df=filter_output_columns(df_out),
        audit={"per_tranche_rw": tranche_view},
        reconciliation=[
            ("rows_in", int(len(df_in))),
            ("rows_out", int(len(df_out))),
            ("rows_with_nan_rw (stub branches)", n_nan),
            ("unique_tranches", int(df_out["tranche_key"].nunique())),
            ("min_rw", float(df_out["rw"].min(skipna=True))),
            ("max_rw", float(df_out["rw"].max(skipna=True))),
        ],
        notes="Single 'rw' column holds the approach-module output (pre "
              "wrappers); rw_details holds the per-row computation trace. "
              "Phase 4 overwrites rw with the post-wrapper-stack value and "
              "appends the transformation transcript to rw_details.",
    )


def write_phase4_snapshot(df_in: pd.DataFrame, df_out: pd.DataFrame,
                          path: str = OUTPUT_PHASE4) -> None:
    """Phase 4 — Floors & caps wrap (look-through, NPL floor, standard floor)."""
    write = _phase_snapshot_writer()
    caps_view = df_out[
        ["ID", "tranche_key", "approach", "rw",
         "lookthrough_cap_eligible", "lookthrough_cap_rw",
         "lookthrough_cap_binds", "npl_concession_applied",
         "npl_floor_applied", "standard_floor_applied", "rw_details"]
    ]
    write(
        path,
        phase_num=4,
        phase_name="Floors & caps wrap (rw is now post-wrapper-stack final)",
        mar_ref="CRE40.49-55 (look-through, agg cap), 45.4 (NPL floor), 41.15/42.7/44.26 (standard floor), MAR22.34(3)",
        source_module="drc_securitisation/sec_engine.py [step 8]",
        input_df=filter_output_columns(df_in),
        output_df=filter_output_columns(df_out),
        audit={"floors_caps_per_position": caps_view},
        reconciliation=[
            ("rows_in", int(len(df_in))),
            ("rows_out", int(len(df_out))),
            ("npl_concession_count", int(df_out["npl_concession_applied"].sum())),
            ("npl_floor_count", int(df_out["npl_floor_applied"].sum())),
            ("lookthrough_cap_binds_count",
                int(df_out["lookthrough_cap_binds"].sum())),
            ("standard_floor_bound_count",
                int((df_out["standard_floor_applied"] > 0).sum())),
        ],
        notes="Single 'rw' column now holds the FINAL post-wrapper-stack RW. "
              "rw_details transcribes the full transformation chain per row: "
              "rw_from_approach_module -> look-through cap -> NPL concession / "
              "100% floor -> standard floor -> rw_final.",
    )


def write_phase5_snapshot(df_in: pd.DataFrame, df_out: pd.DataFrame,
                          path: str = OUTPUT_PHASE5) -> None:
    """Phase 5 — Gross + maturity-scaled JTD per position (pre-netting).

    Materialises the per-position foundation that QA needs to trace the
    netting. Gross JTD (MAR22.27) for sec IS the signed `MV_USD` column — no
    separate column (it would be a pure alias). `maturity_scaled_jtd`
    (MAR22.30) is the one genuine transform here. NO offsetting yet — that is
    phase 6. This is the sec analogue of non-sec phase 2 (Gross JTD per leg).
    """
    write = _phase_snapshot_writer()
    # MV_USD is the gross JTD (MAR22.27); maturity_scaled_jtd is the transform.
    jtd_view = df_out[
        ["ID", "Issuer", "Security", "tranche_key", "approach",
         "signed_notional", "MV_USD",
         "M_T_position_years", "mat_scaling_factor",
         "maturity_scaled_jtd"]
    ].sort_values("MV_USD", key=lambda s: s.abs(), ascending=False)
    scaled_positions = df_out[df_out["mat_scaling_factor"] < 1.0][
        ["ID", "tranche_key", "M_T_position_years",
         "mat_scaling_factor", "MV_USD", "maturity_scaled_jtd"]
    ]
    write(
        path,
        phase_num=5,
        phase_name="Gross + maturity-scaled JTD per position (pre-netting)",
        mar_ref="MAR22.27 (gross JTD = MV_USD), MAR22.30 (maturity scaling, imports MAR22.15-18)",
        source_module="drc_securitisation/sec_engine.py [step 9a]",
        input_df=filter_output_columns(df_in),
        output_df=filter_output_columns(df_out),
        audit={
            "gross_and_scaled_jtd_per_position": jtd_view,
            "positions_with_sub_1y_scaling": scaled_positions,
        },
        reconciliation=[
            ("rows_in", int(len(df_in))),
            ("rows_out", int(len(df_out))),
            ("gross_jtd_is_MV_USD (MAR22.27)", "yes — no separate column"),
            ("sum_gross_jtd = sum_MV_USD (signed)", float(df_out["MV_USD"].sum())),
            ("sum_abs_gross_jtd", float(df_out["MV_USD"].abs().sum())),
            ("sum_maturity_scaled_jtd (signed)",
                float(df_out["maturity_scaled_jtd"].sum())),
            ("positions_scaled_below_1y",
                int((df_out["mat_scaling_factor"] < 1.0).sum())),
        ],
        notes="MAR22.27: sec gross JTD = signed market value (NOT LGD*notional+P&L "
              "as non-sec) — LGD is already inside the sec RWs (CRE41-44) — so the "
              "gross-JTD waypoint IS the MV_USD column (no duplicate). MAR22.30 "
              "maturity scaling: maturity_scaled_jtd = MV_USD * "
              "clip(M_T_position_years,0.25,1.0); differs from MV_USD only for "
              "sub-1y positions. Phase 6 nets maturity_scaled_jtd per tranche.",
    )


def write_phase6_snapshot(df_in: pd.DataFrame, df_out: pd.DataFrame,
                          path: str = OUTPUT_PHASE6) -> None:
    """Phase 6 — Per-tranche offsetting (MAR22.28-29). PURE NETTING — no RW.

    `df_in`  = per-position frame (phase-5 output, one row per position).
    `df_out` = COLLAPSED per-tranche frame (one row per tranche_key), the
               netted result. Row count drops from #positions to #tranches —
               the same per-obligor collapse non-sec phase 3 performs — so the
               netted values appear exactly once, not broadcast across rows.
               Carries `net_jtd_tranche` (UN-weighted) + `rw` pass-through; RW
               is applied in phase 7 (MAR22.25), not here.
    """
    write = _phase_snapshot_writer()
    tranche_view = df_out  # already one row per tranche, sorted by |net JTD|
    # Per-position contribution view: each leg's maturity_scaled_jtd next to the
    # tranche net it folds into, so a reviewer can re-add the offsetting by hand
    # and land on net_jtd_tranche. This is the bridge between the per-position
    # 10_input and the collapsed 20_output.
    netting_buildup = (
        df_in[["ID", "tranche_key", "approach", "signed_notional",
               "MV_USD", "maturity_scaled_jtd"]]
        .merge(
            df_out[["tranche_key", "net_jtd_tranche", "n_positions_in_tranche"]],
            on="tranche_key", how="left",
        )
        .sort_values(["tranche_key", "ID"])
    )
    perfect_hedges = tranche_view[
        (tranche_view["abs_net_jtd_tranche"] == 0)
        & (tranche_view["n_positions_in_tranche"] > 1)
    ]
    # Conservation invariant: Σ maturity_scaled_jtd over a tranche (from the
    # per-position input) == that tranche's net_jtd_tranche (in the output).
    recomputed = df_in.groupby("tranche_key")["maturity_scaled_jtd"].sum()
    declared = df_out.set_index("tranche_key")["net_jtd_tranche"]
    max_netting_residual = float((recomputed - declared).abs().max())
    write(
        path,
        phase_num=6,
        phase_name="Per-tranche offsetting (netting)",
        mar_ref="MAR22.28-29 (offsetting same securitisation exposure)",
        source_module="drc_securitisation/sec_engine.py [step 9b]",
        input_df=filter_output_columns(df_in),
        output_df=tranche_view,
        audit={
            "netting_buildup_per_position": netting_buildup,
            "perfect_hedges_netting_to_zero": perfect_hedges,
        },
        reconciliation=[
            ("rows_in (positions)", int(len(df_in))),
            ("rows_out (tranches)", int(len(df_out))),
            ("rows_collapsed (positions - tranches)",
                int(len(df_in) - len(df_out))),
            ("max_netting_residual (Σ scaled - net_jtd_tranche)",
                max_netting_residual),
            ("sum_net_jtd_tranche (signed)",
                float(df_out["net_jtd_tranche"].sum())),
            ("sum_abs_net_jtd_tranche", float(df_out["abs_net_jtd_tranche"].sum())),
            ("perfect_hedge_tranches", int(len(perfect_hedges))),
        ],
        notes="PURE NETTING (MAR22.28-29): 20_output is ONE row per tranche_key "
              "(the netted result), not a per-position broadcast — row count "
              "drops #positions -> #tranches, mirroring non-sec phase 3's "
              "per-obligor collapse. net_jtd_tranche = Σ maturity_scaled_jtd "
              "over the tranche (UN-weighted); max_netting_residual must be ~0. "
              "RW is applied in phase 7 (MAR22.25), not here. Perfect-hedge "
              "fixtures: v5 IDs (37,68) RMBS senior; (47,72) CC-ABS senior.",
    )


def write_phase7_snapshot(df_in: pd.DataFrame, df_out: pd.DataFrame,
                          config: dict, path: str = OUTPUT_PHASE7) -> None:
    """Phase 7 — Risk-weight net JTD (MAR22.25) + bucket HBR + bucket DRC.

    `df_in`  = per-tranche frame (phase-6 output): UN-weighted `net_jtd_tranche`
               + `rw` pass-through.
    `df_out` = per-bucket frame (one row per drc_sec_bucket) from
               step_10_bucket_hbr.

    This is where RW is applied (MAR22.25). The 30_audit view shows, per tranche,
    the un-weighted net JTD, the RW, the RW-weighted net JTD, and the FV-capped
    capital — so a reviewer can watch RW go in and re-add each bucket's long/short
    sums. Row count collapses #tranches -> #buckets.
    """
    write = _phase_snapshot_writer()
    # Apply RW + FV cap per tranche (same helper step 10 uses), for the audit.
    tr = _weight_and_fv_cap(df_in, config)
    tr["net_side"] = tr["net_jtd_tranche"].apply(
        lambda x: "long" if x > 0 else ("short" if x < 0 else "flat")
    )
    bucket_assignment = tr[
        ["tranche_key", "drc_sec_bucket", "approach", "net_side",
         "net_jtd_tranche", "rw", "rw_weighted_net_jtd",
         "abs_mv_tranche", "capital_fv_capped"]
    ].sort_values(["drc_sec_bucket", "net_jtd_tranche"], ascending=[True, False])
    fv_binds = int(tr["fv_cap_binds"].sum())
    total = float(df_out["drc_b"].sum())
    write(
        path,
        phase_num=7,
        phase_name="RW net JTD (MAR22.25) + bucket HBR + bucket DRC",
        mar_ref="MAR22.25 (RW-weighted net JTD), 22.31/32 (buckets), 22.33+22.23 (HBR), 22.34(3) (FV cap)",
        source_module="drc_securitisation/sec_engine.py [step 10]",
        input_df=filter_output_columns(df_in),
        output_df=df_out,
        audit={"tranche_to_bucket_assignment": bucket_assignment},
        reconciliation=[
            ("rows_in (tranches)", int(len(df_in))),
            ("rows_out (buckets)", int(len(df_out))),
            ("buckets_with_net_short",
                int((df_out["n_short_tranches"] > 0).sum())),
            ("min_hbr", float(df_out["hbr"].min()) if len(df_out) else 0.0),
            ("max_hbr", float(df_out["hbr"].max()) if len(df_out) else 0.0),
            ("fv_cap_bound_tranches", fv_binds),
            ("sum_drc_b (total sec DRC)", total),
            ("buckets_floored_at_zero",
                int((df_out["drc_b_raw"] < 0).sum())),
        ],
        notes="RW is applied HERE (MAR22.25): rw_weighted_net_jtd = rw·8%·net JTD "
              "per tranche, then FV-capped (MAR22.34(3)). DRC_b = Σ_long cap − "
              "HBR·Σ_short cap, floored at 0; the HBR (MAR22.23) uses the "
              "UN-weighted net JTD, NOT the RW-weighted capital. MAR22.31/32 "
              "bucket = (Underlying Pool Type × Region). Net-short tranches "
              "receive the hedge-benefit discount instead of full weight.",
    )


def write_phase8_snapshot(bucket_df: pd.DataFrame, total: float,
                          path: str = OUTPUT_PHASE8) -> None:
    """Phase 8 — Total sec DRC (MAR22.35, simple sum across buckets)."""
    write = _phase_snapshot_writer()
    summary = pd.DataFrame([{
        "metric": "TOTAL SEC DRC (MAR22.31-35 + CRE40-45)",
        "value_$": total,
        "n_buckets": int(len(bucket_df)),
        "n_buckets_floored_at_zero": int((bucket_df["drc_b_raw"] < 0).sum()),
    }])
    write(
        path,
        phase_num=8,
        phase_name="Total sec DRC (simple sum across buckets)",
        mar_ref="MAR22.35 (no cross-bucket hedging; total = Σ bucket DRC)",
        source_module="drc_securitisation/sec_engine.py [step 10 total]",
        input_df=bucket_df,
        output_df=summary,
        audit={"per_bucket_drc": bucket_df},
        reconciliation=[
            ("buckets_in", int(len(bucket_df))),
            ("sum_drc_b", float(bucket_df["drc_b"].sum())),
            ("total_sec_drc", total),
            ("sum_drc_b_equals_total",
                bool(abs(float(bucket_df["drc_b"].sum()) - total) < 1e-6)),
        ],
        notes="MAR22.35: no hedging recognised between different buckets, so the "
              "total sec DRC is the simple sum of the (non-negative) bucket-level "
              "requirements.",
    )


def run_engine_with_phase_snapshots(
    positions: pd.DataFrame, config: dict,
) -> dict:
    """Same chain as `run_engine`, but writes a phase 2-8 snapshot at each
    boundary. Phase 1 is written by sec_loader.write_snapshot. Returns the same
    result dict as `run_engine`.
    """
    df = step_1_dd_gate(positions)
    df = step_2_spe_op_req_check(df)
    df = step_3_implicit_support(df)
    df = step_4_pool_classify(df, config["Bank_Capabilities"])
    df = step_5_hierarchy_route(df, config["Jurisdiction_Permissions"], config["Bank_Capabilities"], config)
    df = step_6_crm_decompose(df)
    write_phase2_snapshot(positions, df)

    df_pre_rw = df.copy()
    df = step_7_risk_weight(df, config)
    write_phase3_snapshot(df_pre_rw, df)

    df_pre_floors = df.copy()
    df = step_8_wrap_floors_caps(df, config)
    write_phase4_snapshot(df_pre_floors, df)

    df_pre_gross = df.copy()
    df = step_9a_gross_maturity_scaled_jtd(df)
    write_phase5_snapshot(df_pre_gross, df)

    # Phase 6: per-tranche offsetting (PURE NETTING, no RW). 10_input =
    # per-position (phase-5 output); 20_output = the COLLAPSED per-tranche frame
    # (one row per tranche_key), carrying un-weighted net_jtd_tranche + rw.
    pos_pre_net = df.copy()
    pos_net = _net_jtd_per_position(df)
    tranche = collapse_to_tranche(pos_net)
    write_phase6_snapshot(pos_pre_net, tranche)

    # Phase 7: apply RW (MAR22.25) + bucket HBR + bucket DRC. 10_input = the
    # per-tranche frame; 20_output = one row per (asset class × region) bucket.
    # RW meets net JTD here; the MAR22.34(3) FV cap is applied inside step 10.
    bucket_df, total = step_10_bucket_hbr(tranche, config)
    write_phase7_snapshot(tranche, bucket_df, config)

    # Phase 8: total sec DRC (MAR22.35, simple sum across buckets).
    write_phase8_snapshot(bucket_df, total)
    return {
        "positions": pos_net,
        "tranches": tranche,
        "buckets": bucket_df,
        "total": total,
    }


# ── SUMMARY / SNAPSHOT ────────────────────────────────────────────────────────

def summarise(result: dict) -> str:
    """Render the engine result dict (positions / tranches / buckets / total)."""
    pos = result["positions"]
    tranche = result["tranches"]
    bucket_df = result["buckets"]
    total = result["total"]
    lines = []
    n_stub = pos["rw"].isna().sum()
    if n_stub > 0:
        lines.append(f"Engine output ({n_stub} of {len(pos)} positions returned NaN RW — "
                     f"approach module(s) still stubbed):")
    else:
        lines.append("Engine output (all per-approach modules live):")
    lines.append("")
    lines.append(f"Positions:        {len(pos)}")
    lines.append(f"Unique tranches:  {pos['tranche_key'].nunique()}")
    lines.append(f"Deals:            {pos['Pool ID'].nunique()}")
    lines.append(f"DRC buckets:      {len(bucket_df)}")
    lines.append(f"TOTAL SEC DRC:    ${total:,.0f}  (MAR22.31-35 bucket HBR)")
    lines.append("")
    lines.append("By approach:")
    by_app = pos.groupby("approach").agg(
        n_positions=("ID", "count"),
        n_tranches=("tranche_key", "nunique"),
        rw_stub_count=("rw", lambda s: s.isna().sum()),
    )
    lines.append(by_app.to_string())
    lines.append("")
    lines.append("Per-bucket DRC (MAR22.33; HBR on net-short, MAR22.25):")
    bd = bucket_df.copy()
    for c in ("sum_long_net_jtd", "sum_short_net_jtd_abs",
              "capital_long", "capital_short", "drc_b_raw", "drc_b"):
        bd[c] = bd[c].round(0)
    bd["hbr"] = bd["hbr"].round(4)
    lines.append(bd.to_string(index=False))
    lines.append("")
    lines.append(f"TOTAL SEC DRC (Σ bucket DRC, MAR22.35):  ${total:,.0f}")
    return "\n".join(lines)


def write_snapshot(result: dict, path: str = OUTPUT_SNAPSHOT) -> None:
    """Back-compat wrapper. Emits ONLY the phase-8 (total) snapshot from an
    engine result dict so callers that want a single final-stage workbook keep
    working."""
    write_phase8_snapshot(result["buckets"], result["total"], path=path)


def main() -> None:
    sec, cfg = build_sec_loader_frame()
    # Phase 1: emit the loader snapshot.
    from .sec_loader import write_snapshot as write_phase1
    write_phase1(sec)
    # Phases 2-8: run the chain with snapshots between each phase.
    out = run_engine_with_phase_snapshots(sec, cfg)
    print(summarise(out))
    print(f"\nPhase outputs written under: {os.path.normpath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
