"""
drc_main.py — Unified DRC entry point.

Runs both DRC sub-branches (sec non-CTP and non-sec) on the v5 portfolio
and produces a single combined output workbook with:
  - top-level summary (sec total + non-sec total + grand total)
  - per-bucket detail
  - per-position attribution

Run:
    /c/Users/amits/anaconda3/python.exe drc_main.py

Authoritative spec: MAR22 (DRC requirement). Two sub-branches:
  - sec non-CTP: MAR22.33-35 wrapping CRE40-45 (drc_securitisation/)
  - non-sec:    MAR22.22-32                    (drc_nonsecuritisation/)
  - sec CTP:    MAR22.36-45 — out of scope for v1

CTP and any DRC-exempt rows (TRS-on-BCOM, FX swap legs) are filtered out
upstream by the bucket-aware loaders in each branch.
"""
from __future__ import annotations
import os
import pandas as pd

from drc_securitisation.sec_loader import (
    build_sec_loader_frame,
    write_snapshot as write_sec_phase1,
)
from drc_securitisation.sec_engine import run_engine_with_phase_snapshots

from drc_nonsecuritisation.nonsec_loader import (
    build_nonsec_loader_frame,
    write_snapshot as write_nonsec_phase1,
)
from drc_nonsecuritisation.nonsec_jtd import (
    add_jtd_columns,
    write_snapshot as write_nonsec_phase2,
)
from drc_nonsecuritisation.nonsec_netting import (
    aggregate_net_per_obligor,
    write_snapshot as write_nonsec_phase3,
)
from drc_nonsecuritisation.nonsec_engine import (
    run_engine as run_nonsec_engine,
    write_phase4_snapshot as write_nonsec_phase4,
    write_phase5_snapshot as write_nonsec_phase5,
)

OUTPUT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "Output data", "SA capital charge"
))
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "drc_total_capital.xlsx")


def run_full_drc(write_phase_snapshots: bool = True) -> dict:
    """Run both DRC sub-branches end-to-end.

    Phase taxonomy (DRC phase-output standard — see FRTB SA/CLAUDE.md):
      Non-sec : load -> gross JTD -> net JTD -> HBR/bucket -> total          (5 phases)
      Sec     : load -> classify/route -> RW -> floors/caps -> tranche-net -> total (6 phases)
    Combined  : drc_total_capital.xlsx (top-level summary + grand total)

    Instrument decomposition (callable bond split, equity index option
    look-through) is UPSTREAM of DRC — the sensitivity engine emits the
    already-decomposed legs with model-priced MVs on Sheet 2 of
    FRTB_Sensitivities.xlsx. DRC has no decomposition phase.
    """
    # ── SEC BRANCH ────────────────────────────────────────────────────────────
    # Phase 1: load
    sec_positions, sec_cfg = build_sec_loader_frame()
    if write_phase_snapshots:
        write_sec_phase1(sec_positions)
    # Phases 2-6: chain with snapshots between each phase
    if write_phase_snapshots:
        sec_engine_out = run_engine_with_phase_snapshots(sec_positions, sec_cfg)
    else:
        from drc_securitisation.sec_engine import run_engine as _run_sec
        sec_engine_out = _run_sec(sec_positions, sec_cfg)
    sec_total = float(sec_engine_out["capital_position"].sum())

    # ── NON-SEC BRANCH ────────────────────────────────────────────────────────
    # Phase 1: load (Sheet 2 ingest + config enrichment)
    nonsec_positions, _nonsec_cfg = build_nonsec_loader_frame()
    nonsec_phase1_out = nonsec_positions.copy()
    if write_phase_snapshots:
        write_nonsec_phase1(nonsec_phase1_out)

    # Phase 2: gross JTD
    nonsec_positions = add_jtd_columns(nonsec_positions)
    if write_phase_snapshots:
        write_nonsec_phase2(nonsec_phase1_out, nonsec_positions)

    # Phase 3: within-obligor net JTD
    nonsec_obligor = aggregate_net_per_obligor(nonsec_positions)
    if write_phase_snapshots:
        write_nonsec_phase3(nonsec_positions, nonsec_obligor)

    # Phase 4: HBR + bucket capital
    nonsec_bucket_df, nonsec_total = run_nonsec_engine(nonsec_obligor)
    if write_phase_snapshots:
        write_nonsec_phase4(nonsec_obligor, nonsec_bucket_df)
        # Phase 5: total
        write_nonsec_phase5(nonsec_bucket_df, nonsec_total,
                            nonsec_obligor, nonsec_positions)

    grand_total = sec_total + nonsec_total

    return {
        "sec_positions": sec_engine_out,
        "sec_total": sec_total,
        "nonsec_positions": nonsec_positions,
        "nonsec_obligor": nonsec_obligor,
        "nonsec_bucket_df": nonsec_bucket_df,
        "nonsec_total": nonsec_total,
        "grand_total": grand_total,
    }


def write_combined_output(result: dict, path: str = OUTPUT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Top-level summary
    summary = pd.DataFrame([
        {"line": "DRC SEC non-CTP (MAR22.33-35 + CRE40-45)",
         "n_positions": len(result["sec_positions"]),
         "capital_$": result["sec_total"]},
        {"line": "DRC NON-SEC (MAR22.22-32) [bonds + equities + decomposed legs]",
         "n_positions": int(result["nonsec_positions"]["is_drc_relevant"].sum()),
         "capital_$": result["nonsec_total"]},
        {"line": "DRC GRAND TOTAL",
         "n_positions": (
             len(result["sec_positions"])
             + int(result["nonsec_positions"]["is_drc_relevant"].sum())
         ),
         "capital_$": result["grand_total"]},
    ])

    # SEC per-tranche view (already deduped key)
    sec_tranches = (
        result["sec_positions"].drop_duplicates("tranche_key")[
            ["tranche_key", "Pool ID", "approach", "approach_reason",
             "is_senior_derived", "Rating",
             "Attachment Pt (%)", "Detachment Pt (%)",
             "rw_floored", "net_notional_tranche", "capital_tranche"]
        ].sort_values("capital_tranche", ascending=False)
    )

    # SEC per-position view
    sec_positions_view = result["sec_positions"][
        ["ID", "Issuer", "Security", "Pool ID", "tranche_key",
         "approach", "Long/Short", "signed_notional", "MV_USD",
         "rw_floored", "capital_position"]
    ].copy()

    # NON-SEC bucket view
    nonsec_bucket_view = result["nonsec_bucket_df"]

    # NON-SEC per-obligor view
    nonsec_obligor_view = result["nonsec_obligor"][[
        "Issuer", "DRC bucket", "credit_quality_category", "RW",
        "n_positions",
        "net_long_jtd", "net_short_jtd",
        "net_long_mat_scaled_jtd", "net_short_mat_scaled_jtd",
        "net_jtd_rank1_equity", "net_jtd_rank2_nonsen",
        "net_jtd_rank3_senior", "net_jtd_rank4_covered",
    ]]

    # NON-SEC per-position view
    nonsec_positions_view = result["nonsec_positions"][[
        "ID", "parent_position_id", "decomposition_leg",
        "Issuer", "obligor", "Security", "DRC bucket", "Position Type",
        "Long/Short", "seniority_class", "seniority_rank",
        "credit_quality_category", "LGD", "RW", "M_T_years",
        "effective_notional", "effective_market_value",
        "is_drc_relevant", "jtd_gross", "maturity_scaled_jtd",
    ]].copy()

    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        summary.to_excel(xl, sheet_name="summary", index=False)
        sec_tranches.to_excel(xl, sheet_name="sec_tranches", index=False)
        sec_positions_view.to_excel(xl, sheet_name="sec_positions", index=False)
        nonsec_bucket_view.to_excel(xl, sheet_name="nonsec_buckets", index=False)
        nonsec_obligor_view.to_excel(xl, sheet_name="nonsec_obligors", index=False)
        nonsec_positions_view.to_excel(xl, sheet_name="nonsec_positions", index=False)


def print_summary(result: dict) -> None:
    print("=" * 72)
    print("FRTB DRC capital report (sec + non-sec, v5 portfolio)")
    print("=" * 72)
    print()
    print(f"  DRC sec non-CTP (MAR22.33-35 + CRE40-45):")
    print(f"    Positions:      {len(result['sec_positions'])}")
    print(f"    Tranches:       {result['sec_positions']['tranche_key'].nunique()}")
    print(f"    Capital:        ${result['sec_total']:>15,.0f}")
    print()
    print(f"  DRC non-sec (MAR22.22-32) [bonds + equities + upstream-decomposed legs]:")
    n_legs_drc = int(result['nonsec_positions']['is_drc_relevant'].sum())
    print(f"    Legs (DRC-relevant): {n_legs_drc}")
    print(f"    Obligors:            {len(result['nonsec_obligor'])}")
    print(f"    Capital:             ${result['nonsec_total']:>15,.0f}")
    print()
    print("-" * 72)
    print(f"  DRC GRAND TOTAL: ${result['grand_total']:>15,.0f}")
    print("-" * 72)
    print()
    legs = result["nonsec_positions"]
    leg_kinds = legs[legs["decomposition_leg"].fillna("").astype(str) != ""] \
        .groupby("decomposition_leg") \
        .agg(parents=("parent_position_id", "nunique"), legs=("ID", "count"))

    print("Decomposition status (consumed from Sheet 2):")
    if not leg_kinds.empty:
        for kind, row in leg_kinds.iterrows():
            label = {
                "vanilla": "Callable bonds (vanilla legs)",
                "short_call": "Callable bonds (short-call legs)",
                "index_constituent": "Equity index options (constituent legs)",
            }.get(kind, kind)
            print(f"  {label:50s}{int(row['parents']):>3} parent(s) -> {int(row['legs']):>3} leg(s)")
    else:
        print("  (no decomposed legs in non-sec frame)")

    excluded = legs[~legs["is_drc_relevant"]]
    if len(excluded):
        print()
        print("Excluded from DRC:")
        print(f"  {len(excluded)} leg(s) (matured)")


def _list_phase_outputs() -> list[str]:
    """Non-sec (5 phases) + sec (6 phases) + combined workbook this run produces."""
    return [
        "drc_nonsec_phase1_load.xlsx",
        "drc_nonsec_phase2_gross_jtd.xlsx",
        "drc_nonsec_phase3_net_jtd.xlsx",
        "drc_nonsec_phase4_hbr_bucket.xlsx",
        "drc_nonsec_phase5_total.xlsx",
        "drc_sec_phase1_load.xlsx",
        "drc_sec_phase2_classify_route.xlsx",
        "drc_sec_phase3_riskweight.xlsx",
        "drc_sec_phase4_floors_caps.xlsx",
        "drc_sec_phase5_tranche_net.xlsx",
        "drc_sec_phase6_total.xlsx",
        "drc_total_capital.xlsx",
    ]


def main() -> None:
    result = run_full_drc()
    print_summary(result)
    write_combined_output(result)
    print(f"\nCombined workbook: {os.path.normpath(OUTPUT_PATH)}")
    print()
    print("Phase outputs (one Excel per phase, 5-sheet layout):")
    for name in _list_phase_outputs():
        print(f"  {os.path.join(os.path.normpath(OUTPUT_DIR), name)}")


if __name__ == "__main__":
    main()
