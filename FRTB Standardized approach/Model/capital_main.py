"""
FRTB SA Capital Engine — Stage 2, Phases 5 & 6
capital_main.py: three-scenario wrapper (MAR21.6) + orchestrator + output writer.

Pipeline:
    Phase 0: capital_loader.load_sensitivity_grid + load_config
    Phase 1: weighted_sensitivities.compute_weighted_sensitivities
    Phase 2: intra_bucket.compute_intra_bucket          per scenario
    Phase 3: inter_bucket.compute_inter_bucket          per scenario
    Phase 4: curvature.compute_curvature                per scenario
    Phase 5: sum Delta + Vega + Curvature per scenario, take max [MAR21.5-6]
    Phase 6: write FRTB_SA_Capital.xlsx with full detail + summary

Scenarios (MAR21.6): low / medium / high. Final SA capital = max across scenarios.
"""
import os
import pandas as pd

from capital_loader import load_sensitivity_grid, load_config
from weighted_sensitivities import compute_weighted_sensitivities
from intra_bucket import (compute_intra_bucket,
                          compute_intra_bucket_with_trace,
                          _write_rho_blocks_to_sheet)
from inter_bucket import (compute_inter_bucket,
                          compute_inter_bucket_with_trace,
                          _write_blocks_to_sheet)
from curvature import compute_curvature, compute_curvature_with_trace


SCENARIOS = ("low", "medium", "high")


def run_one_scenario(ws_df: pd.DataFrame, cfg: dict, scenario: str) -> dict:
    """Run Phases 2-4 for one scenario and return aggregated charges."""
    p2 = compute_intra_bucket(ws_df, cfg, scenario=scenario)
    p3 = compute_inter_bucket(p2, cfg, scenario=scenario)
    curv_rc, curv_detail = compute_curvature(ws_df, cfg, scenario=scenario)

    delta_vega_total = float(p3["charge"].sum())
    curv_total = float(curv_rc["charge"].sum()) if len(curv_rc) else 0.0

    # Break out delta vs vega separately for reporting
    delta_rows = p3[p3["kind"] == "delta"]
    vega_rows  = p3[p3["kind"] == "vega"]
    delta_total = float(delta_rows["charge"].sum())
    vega_total = float(vega_rows["charge"].sum())

    return dict(
        scenario=scenario,
        delta_total=delta_total,
        vega_total=vega_total,
        curvature_total=curv_total,
        total=delta_total + vega_total + curv_total,
        phase2=p2,
        phase3=p3,
        curv_rc=curv_rc,
        curv_detail=curv_detail,
    )


def run_all_scenarios(ws_df: pd.DataFrame, cfg: dict) -> dict:
    return {s: run_one_scenario(ws_df, cfg, s) for s in SCENARIOS}


def build_summary(results: dict) -> pd.DataFrame:
    """Scenario x (delta, vega, curvature, total) summary."""
    rows = []
    for s in SCENARIOS:
        r = results[s]
        rows.append(dict(
            scenario=s,
            delta=r["delta_total"],
            vega=r["vega_total"],
            curvature=r["curvature_total"],
            total=r["total"],
        ))
    df = pd.DataFrame(rows)
    # Append the MAR21.5 max row
    max_row = df.loc[df["total"].idxmax()].copy()
    df.loc[len(df)] = dict(scenario="FRTB_SA_CAPITAL (max)", **{
        "delta": max_row["delta"],
        "vega": max_row["vega"],
        "curvature": max_row["curvature"],
        "total": max_row["total"],
    })
    return df


def write_output(results: dict, summary: pd.DataFrame, ws_df: pd.DataFrame,
                 out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="Capital_Summary", index=False)

        # Per-scenario risk-class detail
        for s in SCENARIOS:
            p3 = results[s]["phase3"].copy()
            curv_rc = results[s]["curv_rc"].copy()
            if len(curv_rc):
                curv_rc["kind"] = "curvature"
            combined = pd.concat([p3, curv_rc], ignore_index=True)
            combined = combined[["kind", "risk_class", "n_buckets",
                                 "charge", "scenario"]]
            combined.to_excel(xw, sheet_name=f"{s}_by_risk_class", index=False)

        # Per-scenario bucket-level detail (intra-bucket Kb, Sb)
        for s in SCENARIOS:
            results[s]["phase2"].to_excel(xw, sheet_name=f"{s}_intra", index=False)

        # Curvature bucket detail
        for s in SCENARIOS:
            if len(results[s]["curv_detail"]):
                results[s]["curv_detail"].to_excel(
                    xw, sheet_name=f"{s}_curv_detail", index=False)

        # Phase 1 weighted sensitivities (all rows)
        ws_df.to_excel(xw, sheet_name="Phase1_WS", index=False)


def write_phase_traces(ws_df: pd.DataFrame, cfg: dict, out_dir: str) -> None:
    """Mirror the per-phase trace workbooks that the standalone __main__ blocks
    of intra_bucket / inter_bucket / curvature produce. Running capital_main
    therefore yields the SAME outputs as running each module individually --
    plus the consolidated FRTB_SA_Capital workbook on top.
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── Phase 1: weighted_sensitivities ──
    p1_path = os.path.join(out_dir, "phase1_weighted_sensitivities.xlsx")
    readme1 = pd.DataFrame([
        ("Sheet", "Purpose"),
        ("README", "Column glossary."),
        ("Weighted_Sensitivities",
         "Tidy long-format grid with per-row RW decomposition: rw_base, "
         "divisor, divisor_reference, rw, ws (= rw * value). Curvature rows "
         "carry NaN for rw/ws (handled by Phase 4)."),
        ("Divisor_Applied",
         "Filtered view of rows where MAR21.44 (GIRR specified ccy) or "
         "MAR21.88 (FX specified pair) sqrt(2) divisor was applied."),
    ], columns=["A", "B"])
    with pd.ExcelWriter(p1_path, engine="openpyxl") as xw:
        readme1.to_excel(xw, sheet_name="README", index=False, header=False)
        ws_df.to_excel(xw, sheet_name="Weighted_Sensitivities", index=False)
        div_rows = ws_df[ws_df["divisor"] > 1.0]
        if len(div_rows):
            div_rows.to_excel(xw, sheet_name="Divisor_Applied", index=False)

    # ── Phase 2: intra-bucket trace ──
    p2_path = os.path.join(out_dir, "phase2_intra_bucket.xlsx")
    readme2 = pd.DataFrame([
        ("Sheet", "Purpose"),
        ("README", "Column / sheet glossary."),
        ("factors_distinct",
         "One row per MAR21 distinct risk factor in each bucket. "
         "SCENARIO-INDEPENDENT (WS = RW * value; no rho involvement)."),
        ("factors_per_position",
         "Every original tidy row with full RW decomposition. "
         "Sum of ws within a factor_label = summed_ws in the distinct sheet."),
        ("{scen}_summary",
         "Per (kind, risk_class, bucket_id): Sb, Kb, n_distinct_factors, mode."),
        ("{scen}_rho_matrices",
         "Stacked correlation matrices over distinct risk factors. "
         "Single-factor and 'simple_sum' buckets are omitted."),
    ], columns=["A", "B"])
    with pd.ExcelWriter(p2_path, engine="openpyxl") as xw:
        readme2.to_excel(xw, sheet_name="README", index=False, header=False)
        _, pos_df, dist_df, _ = compute_intra_bucket_with_trace(
            ws_df, cfg, scenario="medium")
        dist_df.drop(columns=["scenario"], errors="ignore").to_excel(
            xw, sheet_name="factors_distinct", index=False)
        pos_df.drop(columns=["scenario"], errors="ignore").to_excel(
            xw, sheet_name="factors_per_position", index=False)
        for scen in SCENARIOS:
            summary, _, _, rho_blocks = compute_intra_bucket_with_trace(
                ws_df, cfg, scenario=scen)
            summary.to_excel(xw, sheet_name=f"{scen}_summary", index=False)
            sheet = xw.book.create_sheet(f"{scen}_rho_matrices")
            _write_rho_blocks_to_sheet(rho_blocks, sheet)

    # ── Phase 3: inter-bucket trace ──
    p3_path = os.path.join(out_dir, "phase3_inter_bucket.xlsx")
    readme3 = pd.DataFrame([
        ("Sheet", "Purpose"),
        ("README", "Column / sheet glossary."),
        ("{scen}_summary", "Per-risk-class charge with sum_Kb2_sqrt baseline."),
        ("{scen}_bucket_inputs",
         "Per (risk_class, bucket): Kb / Sb feeding the inter-bucket sum."),
        ("{scen}_gamma_matrices",
         "Stacked inter-bucket gamma matrices for risk classes with >=2 buckets."),
    ], columns=["A", "B"])
    with pd.ExcelWriter(p3_path, engine="openpyxl") as xw:
        readme3.to_excel(xw, sheet_name="README", index=False, header=False)
        for scen in SCENARIOS:
            p2 = compute_intra_bucket(ws_df, cfg, scenario=scen)
            summary, inputs, gamma_blocks = compute_inter_bucket_with_trace(
                p2, cfg, scenario=scen)
            summary.to_excel(xw, sheet_name=f"{scen}_summary", index=False)
            inputs.to_excel(xw, sheet_name=f"{scen}_bucket_inputs", index=False)
            sheet = xw.book.create_sheet(f"{scen}_gamma_matrices")
            _write_blocks_to_sheet(gamma_blocks, sheet, label_prefix="Risk class")

    # ── Phase 4: curvature trace ──
    p4_path = os.path.join(out_dir, "phase4_curvature.xlsx")
    readme4 = pd.DataFrame([
        ("Sheet", "Purpose"),
        ("README", "Column / sheet glossary."),
        ("factors_per_position",
         "Every CURV_*_UP / CURV_*_DN tidy row. SCENARIO-INDEPENDENT raw CVR."),
        ("{scen}_summary", "Per-risk-class curvature charge (MAR21.101)."),
        ("{scen}_buckets", "Per-bucket Kb (max of UP/DN), Sb, side selected."),
        ("{scen}_factors_distinct",
         "Per distinct risk factor: cvr_up_summed, cvr_dn_summed, side, "
         "contribution to bucket Sb."),
    ], columns=["A", "B"])
    with pd.ExcelWriter(p4_path, engine="openpyxl") as xw:
        readme4.to_excel(xw, sheet_name="README", index=False, header=False)
        _, _, _, pos_f = compute_curvature_with_trace(ws_df, cfg, scenario="medium")
        pos_f.drop(columns=["scenario"], errors="ignore").to_excel(
            xw, sheet_name="factors_per_position", index=False)
        for scen in SCENARIOS:
            summary, buckets, distinct_f, _ = compute_curvature_with_trace(
                ws_df, cfg, scenario=scen)
            summary.to_excel(xw, sheet_name=f"{scen}_summary", index=False)
            buckets.to_excel(xw, sheet_name=f"{scen}_buckets", index=False)
            distinct_f.to_excel(xw, sheet_name=f"{scen}_factors_distinct", index=False)

    print(f"Wrote per-phase trace files to: {out_dir}")
    for name in ("phase1_weighted_sensitivities.xlsx",
                  "phase2_intra_bucket.xlsx",
                  "phase3_inter_bucket.xlsx",
                  "phase4_curvature.xlsx"):
        print(f"  - {name}")


def main():
    print("Loading inputs (Phase 0)...")
    tidy = load_sensitivity_grid()
    cfg = load_config()

    print("Computing weighted sensitivities (Phase 1)...")
    ws_df = compute_weighted_sensitivities(tidy, cfg)

    print("Running three scenarios (Phases 2-4)...")
    results = run_all_scenarios(ws_df, cfg)

    summary = build_summary(results)
    print("\n" + "=" * 60)
    print("FRTB SA CAPITAL — SCENARIO SUMMARY")
    print("=" * 60)
    print(summary.to_string(
        index=False,
        formatters={c: "{:,.2f}".format
                    for c in ["delta", "vega", "curvature", "total"]}))

    # Detailed breakdown for the winning scenario
    winning = summary.iloc[summary["total"][:-1].idxmax()]["scenario"] \
        if len(summary) > 1 else summary.iloc[0]["scenario"]
    win_res = results[winning]
    print("\n" + "=" * 60)
    print(f"WINNING SCENARIO: {winning.upper()}  (Total = ${win_res['total']:,.2f})")
    print("=" * 60)
    rc_df = pd.concat([win_res["phase3"], win_res["curv_rc"].assign(kind="curvature")],
                      ignore_index=True)
    rc_df = rc_df[["kind", "risk_class", "charge"]]
    print(rc_df.to_string(index=False,
                          formatters={"charge": "{:,.2f}".format}))
    print(f"\n{'TOTAL':<40s} {win_res['total']:>15,.2f}")

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    out_path = os.path.join(out_dir, "FRTB_SA_Capital.xlsx")
    write_output(results, summary, ws_df, out_path)
    print(f"\nWrote: {out_path}")

    # Per-phase audit-trail workbooks (same content as running each module
    # standalone). These are independent of FRTB_SA_Capital and useful for
    # tracing any final number back to its inputs.
    write_phase_traces(ws_df, cfg, out_dir)


if __name__ == "__main__":
    main()
