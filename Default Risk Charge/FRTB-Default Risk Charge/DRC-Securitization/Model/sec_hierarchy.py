"""
sec_hierarchy.py — Step 3b: approach routing per CRE40.41-47.

Adds two columns to the classifier frame:
  approach        : str  — "SEC-IRBA" / "SEC-ERBA" / "SEC-IAA" / "SEC-SA" / "1250%"
  approach_reason : str  — short string with the deciding CRE paragraph

Routing logic (in order — first match wins):

  Step  Condition                                       Approach    CRE
  ----  ----------------------------------------------  ----------  ------------------
   1    dd_status != 'pass'                             1250%       40.31
   2    is_resecuritisation                             SEC-SA      41.16 (ERBA forbidden)
   3    NPL deal AND bank uses F-IRB on pool            block IRBA  45.3  (drops to step 5+)
   4    IRB pool, OR mixed >= 95% IRB-modelable         SEC-IRBA    40.42 / 40.46
   5    SA pool, rated, jurisdiction permits ERBA       SEC-ERBA    40.43
   6    Unrated ABCP facility, IAA approval, permitted  SEC-IAA     40.44
   7    SA pool, none of the above                      SEC-SA      40.45
   8    None applicable                                 1250%       40.41

Step 3 (NPL + F-IRB block) requires a `bank_uses_f_irb_on_pool` field which is
not currently in the config. The hook is in place; when the data team wires it
into Deal_Master or Position_Overrides the block will activate automatically.

Step 6 (SEC-IAA path) requires an `is_abcp_facility` field distinguishing
ABCP CP investments (use SEC-ERBA short-term Table 1) from sponsor-side
liquidity / credit-enhancement facilities (use SEC-IAA → ERBA mapping).
Hook in place; v5 portfolio's only ABCP holding is the CP itself, which
correctly routes to SEC-ERBA via step 5.
"""
from __future__ import annotations
import os
import pandas as pd

from .sec_loader import build_sec_loader_frame, OUTPUT_DIR
from .sec_classifier import classify_pools

APPROACH_IRBA = "SEC-IRBA"
APPROACH_ERBA = "SEC-ERBA"
APPROACH_IAA = "SEC-IAA"
APPROACH_SA = "SEC-SA"
APPROACH_1250 = "1250%"


def _mixed_pool_irba_threshold(config: dict) -> float:
    """CRE40.46: min IRB-modelable share to use SEC-IRBA on a mixed pool.

    Sourced from SEC_Constants.mixed_pool_irba_share_threshold.
    """
    return float(
        config["SEC_Constants"]
        .set_index("Constant")
        .at["mixed_pool_irba_share_threshold", "Value"]
    )

# Auxiliary file written only by this module's standalone main(); the canonical
# phase 2 output is `drc_sec_phase2_classify_route.xlsx` produced by sec_engine.
OUTPUT_SNAPSHOT = os.path.join(OUTPUT_DIR, "_aux_drc_sec_hierarchy.xlsx")


def lookup_jurisdiction(juris_perms: pd.DataFrame, jurisdiction: str) -> dict:
    """Look up ERBA/IAA permissions for a jurisdiction.

    Conservative fallback: unknown jurisdiction → permissions denied.
    """
    jp = juris_perms.set_index("Jurisdiction")
    if jurisdiction not in jp.index:
        return {"erba_permitted": False, "iaa_permitted": False}
    return {
        "erba_permitted": bool(jp.at[jurisdiction, "erba_permitted"]),
        "iaa_permitted": bool(jp.at[jurisdiction, "iaa_permitted"]),
    }


def lookup_bank_iaa(bank_capabilities: pd.DataFrame, asset_class: str) -> bool:
    """Look up has_iaa_for_abcp from Bank_Capabilities."""
    bc = bank_capabilities.set_index("Asset_class")
    if asset_class not in bc.index:
        return False
    return bool(bc.at[asset_class, "has_iaa_for_abcp"])


def _resolve_rating(row: pd.Series) -> str | None:
    """Pick the effective rating for ERBA: explicit Rating, else inferred_rating."""
    r = row.get("Rating")
    if isinstance(r, str) and r.strip():
        return r.strip()
    inf = row.get("inferred_rating")
    if isinstance(inf, str) and inf.strip():
        return inf.strip()
    return None


def route_position(
    row: pd.Series, juris_perms: pd.DataFrame, bank_capabilities: pd.DataFrame,
    mixed_pool_threshold: float,
) -> tuple[str, str]:
    """Returns (approach, reason). The reason string spells out the exact
    conditional that fired, so a reviewer can trace each row's decision
    from the input columns visible on the same line.

    `mixed_pool_threshold` is the CRE40.46 IRB-modelable share cutoff
    (sourced from config by route_approaches).
    """
    # Step 1: DD failure (CRE40.31) - hard 1250%, no other gates evaluated
    if str(row.get("dd_status", "pass")).lower() != "pass":
        return APPROACH_1250, "CRE40.31 DD failed -> 1250% (skips hierarchy)"

    # Step 2: Resecuritisation forces SEC-SA (CRE41.16; ERBA forbidden for resec)
    if bool(row.get("is_resecuritisation", False)):
        return (
            APPROACH_SA,
            "CRE41.16 is_resecuritisation=True -> SEC-SA (ERBA forbidden for resec)",
        )

    # Step 3-4: SEC-IRBA path (CRE40.42 / 40.46 / 45.3)
    # Eligibility expressed directly on irb_modelable_share (CRE40.15-17 in
    # numeric form; no separate categorical column needed):
    #   - fully IRB-modelable (share >= 1.0)            -> CRE40.42
    #   - partially IRB-modelable AND share >= 95%      -> CRE40.46
    # Blocked by NPL deal + bank using F-IRB on the underlying (CRE45.3).
    irb_share = float(row.get("irb_modelable_share", 0.0))
    is_npl = bool(row.get("is_npl_derived", False))
    bank_uses_f_irb = bool(row.get("bank_uses_f_irb_on_pool", False))  # data hook

    irba_eligible = irb_share >= mixed_pool_threshold
    irba_blocked_by_npl_firb = irba_eligible and is_npl and bank_uses_f_irb
    if irba_eligible and not irba_blocked_by_npl_firb:
        if irb_share >= 1.0:
            return (
                APPROACH_IRBA,
                f"CRE40.42 fully IRB-modelable (irb_share={irb_share:.2f})"
                f" -> SEC-IRBA",
            )
        else:  # partially IRB-modelable with share >= threshold
            return (
                APPROACH_IRBA,
                f"CRE40.46 partially IRB-modelable (irb_share={irb_share:.2f}"
                f" >= {mixed_pool_threshold:.2f} threshold) -> SEC-IRBA",
            )

    # Step 5: SEC-ERBA path (CRE40.43)
    # Eligibility: rated tranche (external Rating or inferred_rating)
    #              AND jurisdiction permits ERBA per Jurisdiction_Permissions
    # The Jurisdiction_Permissions flag is a demo configuration; in a
    # production US deployment Dodd-Frank §939A would set US to FALSE and
    # rated US sec positions would fall through to step 6 / 7.
    rating = _resolve_rating(row)
    jurisdiction = row.get("jurisdiction", "US")
    perms = lookup_jurisdiction(juris_perms, jurisdiction)
    if rating and perms["erba_permitted"]:
        if irba_blocked_by_npl_firb:
            # Pool IS IRB-eligible, but the NPL+F-IRB combination blocks IRBA
            # per CRE45.3, so the next-ranked approach (ERBA) takes over.
            reason = (
                f"CRE45.3 IRBA blocked (NPL+F-IRB) -> CRE40.43 SEC-ERBA used "
                f"(external rating {rating} available)"
            )
        else:
            # Two independent conditions, evaluated in order:
            #   (a) Pool is NOT IRB-eligible (failed CRE40.42 / 40.46 gate)
            #   (b) Tranche has an external rating (or inferred_rating)
            reason = (
                f"CRE40.43 SEC-ERBA used (pool not IRB-eligible, "
                f"external rating {rating} available)"
            )
        return APPROACH_ERBA, reason

    # Step 6: SEC-IAA path (CRE40.44; unrated ABCP facility)
    # Eligibility: asset_class contains 'ABCP'
    #              AND position is a liquidity facility / credit enhancement
    #                  (Position Sub-Type = 'Liquidity_Facility')
    #              AND bank has IAA approval for the asset class
    #              AND jurisdiction permits IAA
    asset_class = str(row.get("Underlying Pool Type", ""))
    is_abcp = "ABCP" in asset_class
    is_facility = bool(row.get("is_abcp_facility", False))
    bank_has_iaa = lookup_bank_iaa(bank_capabilities, asset_class)
    if is_abcp and is_facility and bank_has_iaa and perms["iaa_permitted"]:
        return (
            APPROACH_IAA,
            "CRE40.44 unrated ABCP facility + bank has IAA approval -> SEC-IAA",
        )

    # Step 7: SEC-SA fallback (CRE40.45). Reachable when no higher-ranked
    # approach is eligible: typically an unrated, not-IRB-modelable position
    # that isn't an ABCP facility either. Also catches partially-modelable
    # pools below the 95% IRBA threshold and fully-modelable pools whose
    # IRBA path was blocked by CRE45.3 (NPL + F-IRB on underlying).
    why = []
    if irb_share == 0:
        why.append("not IRB-modelable (irb_share=0.00)")
    elif irb_share < mixed_pool_threshold:
        why.append(
            f"partially IRB-modelable (irb_share={irb_share:.2f}"
            f"<{mixed_pool_threshold:.2f})"
        )
    if not rating:
        why.append("no external rating / inferred_rating")
    elif not perms["erba_permitted"]:
        why.append(f"{jurisdiction} does not permit ERBA")
    if not (is_abcp and is_facility):
        why.append("not an unrated ABCP facility (IAA not eligible)")
    reason = "CRE40.45 SEC-SA fallback (" + ", ".join(why) + ")"
    if irba_blocked_by_npl_firb:
        reason = "CRE45.3 IRBA blocked (NPL+F-IRB) -> " + reason
    return APPROACH_SA, reason

    # Step 8: residual 1250% — currently unreachable; left as documentation.
    # Will become reachable if a K_SA-availability gate is added (CRE41.10
    # already covers W-unknown >5%, but K_SA itself going unavailable is
    # a different failure mode not yet modelled).
    # return APPROACH_1250, "CRE40.41 no approach applicable"


def route_approaches(
    positions: pd.DataFrame,
    juris_perms: pd.DataFrame,
    bank_capabilities: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Apply CRE40.41-47 routing to every position. `config` carries the
    full FRTB_Sec_Config dict so per-rule constants (e.g. mixed-pool IRBA
    share threshold) can be sourced from SEC_Constants rather than hardcoded."""
    df = positions.copy()
    mixed_pool_threshold = _mixed_pool_irba_threshold(config)
    routed = df.apply(
        lambda r: route_position(r, juris_perms, bank_capabilities, mixed_pool_threshold),
        axis=1,
        result_type="expand",
    )
    df["approach"] = routed[0]
    df["approach_reason"] = routed[1]
    return df


# ── SUMMARY / SNAPSHOT ────────────────────────────────────────────────────────

def summarise(df: pd.DataFrame) -> str:
    lines = []
    lines.append("Approach routing breakdown (CRE40.41-47):")
    lines.append(df["approach"].value_counts().to_string())
    lines.append("")
    lines.append("Routing reasons:")
    reasons = df.groupby("approach")["approach_reason"].agg(
        lambda s: s.value_counts().to_dict()
    )
    for ap, counts in reasons.items():
        lines.append(f"  {ap}:")
        for r, n in counts.items():
            lines.append(f"    {n}x  {r}")
    lines.append("")
    lines.append("Per-tranche routing (deduplicated by tranche_key):")
    tk_view = (
        df.drop_duplicates("tranche_key")
        [["Pool ID", "Attachment Pt (%)", "Detachment Pt (%)", "Rating",
          "is_senior_derived", "approach", "approach_reason"]]
        .sort_values(["Pool ID", "Attachment Pt (%)"])
    )
    lines.append(tk_view.to_string(index=False))
    return "\n".join(lines)


def write_snapshot(df: pd.DataFrame, path: str = OUTPUT_SNAPSHOT) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tk_view = (
        df.drop_duplicates("tranche_key")
        [["tranche_key", "Pool ID", "pool_type", "Attachment Pt (%)",
          "Detachment Pt (%)", "Rating", "is_senior_derived",
          "irb_modelable_share", "approach", "approach_reason"]]
        .sort_values(["Pool ID", "Attachment Pt (%)"])
    )
    by_approach = df["approach"].value_counts().reset_index()
    by_approach.columns = ["approach", "n_positions"]
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        df.to_excel(xl, sheet_name="positions", index=False)
        tk_view.to_excel(xl, sheet_name="tranche_routing", index=False)
        by_approach.to_excel(xl, sheet_name="by_approach", index=False)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

def main() -> None:
    sec, cfg = build_sec_loader_frame()
    classified = classify_pools(sec, cfg["Bank_Capabilities"])
    routed = route_approaches(
        classified, cfg["Jurisdiction_Permissions"], cfg["Bank_Capabilities"], cfg
    )
    print(summarise(routed))
    write_snapshot(routed)
    print(f"\nSnapshot written: {os.path.normpath(OUTPUT_SNAPSHOT)}")


if __name__ == "__main__":
    main()
