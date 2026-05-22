"""
nonsec_engine.py — Bucket-level DRC capital orchestrator (MAR22.22-26).

Per MAR22.23, hedge benefit ratio at bucket level:
    HBR = sum_long(mat_scaled_net_jtd) / [sum_long(mat_scaled_net_jtd) + sum_short(|mat_scaled_net_jtd|)]

Per MAR22.25, bucket-level capital:
    DRC_b = sum_long(RW * mat_scaled_net_jtd) - HBR * sum_short(RW * |mat_scaled_net_jtd|)

Per MAR22.26, total non-sec DRC:
    DRC_total = sum_b max(DRC_b, 0)
    (No cross-bucket hedging recognised; no bucket capital below zero.)

Build sequence:
  loader -> JTD -> within-obligor netting -> bucket-level HBR + capital -> total
"""
from __future__ import annotations
import os
import pandas as pd

from .nonsec_loader import build_nonsec_loader_frame, OUTPUT_DIR
from .nonsec_jtd import add_jtd_columns
from .nonsec_netting import aggregate_net_per_obligor

OUTPUT_SNAPSHOT = os.path.join(OUTPUT_DIR, "drc_nonsec_phase4_hbr_bucket.xlsx")
OUTPUT_PHASE5_TOTAL = os.path.join(OUTPUT_DIR, "drc_nonsec_phase5_total.xlsx")


def compute_hbr(obligor_df: pd.DataFrame) -> float:
    """Per MAR22.23. Operates on maturity-scaled net JTDs (NOT RW-weighted).

    Returns HBR in [0, 1]. If the bucket has no longs, HBR = 0 (no hedging
    benefit to apply); if no shorts, HBR = 1 (irrelevant since no shorts to discount).
    """
    sum_long = obligor_df["net_long_mat_scaled_jtd"].sum()
    sum_short_abs = obligor_df["net_short_mat_scaled_jtd"].abs().sum()
    denom = sum_long + sum_short_abs
    if denom == 0:
        return 0.0
    return float(sum_long / denom)


def compute_bucket_capital(
    obligor_df: pd.DataFrame, hbr: float, capital_factor: float = 0.08
) -> dict:
    """Per MAR22.25. obligor_df is the per-obligor netted frame for ONE bucket.

    capital_factor of 0.08 converts RW * notional to capital (Basel 8% factor).
    Note: in DRC the formula is RW * net_JTD directly (no 8% factor) because
    DRC RWs are calibrated as capital ratios. So we use 1.0 here.
    Wait — actually MAR22 uses the RWs from Table 2 directly as percentages
    of net JTD, not as RWAs needing the 8% multiplier. The Basel general
    pattern (RWA = RW * 12.5 * capital -> RWA * 8% = capital) doesn't apply
    here. The DRC requirement IS the capital (no further 8% scaling).
    """
    long_contribs = (
        obligor_df["RW"] * obligor_df["net_long_mat_scaled_jtd"]
    )
    short_contribs = (
        obligor_df["RW"] * obligor_df["net_short_mat_scaled_jtd"].abs()
    )
    sum_rw_long = float(long_contribs.sum())
    sum_rw_short = float(short_contribs.sum())
    drc_b_raw = sum_rw_long - hbr * sum_rw_short
    drc_b = max(drc_b_raw, 0.0)
    return {
        "sum_rw_long": sum_rw_long,
        "sum_rw_short": sum_rw_short,
        "hbr": hbr,
        "drc_b_raw": drc_b_raw,
        "drc_b": drc_b,
    }


def run_engine(obligor_df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Returns (bucket_summary_df, total_drc)."""
    bucket_rows = []
    for bucket, grp in obligor_df.groupby("DRC bucket"):
        hbr = compute_hbr(grp)
        result = compute_bucket_capital(grp, hbr)
        bucket_rows.append({
            "DRC bucket": bucket,
            "n_obligors": len(grp),
            "sum_long_mat_scaled_jtd": float(grp["net_long_mat_scaled_jtd"].sum()),
            "sum_short_mat_scaled_jtd": float(grp["net_short_mat_scaled_jtd"].sum()),
            **result,
        })
    bucket_df = pd.DataFrame(bucket_rows)
    total_drc = float(bucket_df["drc_b"].sum())
    return bucket_df, total_drc


# -- SUMMARY / SNAPSHOT --------------------------------------------------------

def summarise(positions_df: pd.DataFrame, obligor_df: pd.DataFrame,
              bucket_df: pd.DataFrame, total_drc: float) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("FRTB DRC — Non-securitisation (phase 1: plain bonds + equities)")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Phase-1 positions:    {positions_df['is_drc_relevant'].sum()} "
                 f"of {len(positions_df)} non-sec rows")
    lines.append(f"Obligors:             {len(obligor_df)} (after within-obligor netting)")
    lines.append(f"Buckets:              {len(bucket_df)}")
    lines.append("")
    lines.append("Per-bucket DRC capital (MAR22.25):")
    bdf = bucket_df.copy()
    for col in ("sum_long_mat_scaled_jtd", "sum_short_mat_scaled_jtd",
                "sum_rw_long", "sum_rw_short", "drc_b_raw", "drc_b"):
        bdf[col] = bdf[col].round(0)
    bdf["hbr"] = bdf["hbr"].round(4)
    lines.append(bdf.to_string(index=False))
    lines.append("")
    lines.append("-" * 72)
    lines.append(f"TOTAL NON-SEC DRC (sum across buckets, no negatives, MAR22.26):")
    lines.append(f"    ${total_drc:,.0f}")
    lines.append("-" * 72)
    lines.append("")
    lines.append("Top 10 obligors by RW × maturity-scaled long contribution:")
    od = obligor_df.copy()
    od["rw_long_contrib"] = od["RW"] * od["net_long_mat_scaled_jtd"]
    od["rw_short_contrib"] = od["RW"] * od["net_short_mat_scaled_jtd"].abs()
    top = od.nlargest(10, "rw_long_contrib")[
        ["Issuer", "DRC bucket", "credit_quality_category", "RW",
         "net_long_mat_scaled_jtd", "net_short_mat_scaled_jtd",
         "rw_long_contrib", "rw_short_contrib"]
    ]
    lines.append(top.to_string(index=False))
    return "\n".join(lines)


def write_phase4_snapshot(
    obligor_df: pd.DataFrame, bucket_df: pd.DataFrame,
    path: str = OUTPUT_SNAPSHOT,
) -> None:
    """Phase 4 — Bucket HBR + bucket capital (MAR22.23, MAR22.25).

    `obligor_df` = phase 3 output (per-(obligor, bucket) net JTD).
    `bucket_df`  = per-bucket frame with HBR, sum_rw_long, sum_rw_short, drc_b.
                   Consumed by phase 5.
    """
    import sys
    here = os.path.dirname(__file__)
    parent = os.path.abspath(os.path.join(here, ".."))
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from phase_snapshot import write_phase_snapshot

    od = obligor_df.copy()
    od["rw_long_contrib"] = od["RW"] * od["net_long_mat_scaled_jtd"]
    od["rw_short_contrib"] = od["RW"] * od["net_short_mat_scaled_jtd"].abs()
    top_long = od.nlargest(min(10, len(od)), "rw_long_contrib")[
        ["Issuer", "DRC bucket", "credit_quality_category", "RW",
         "net_long_mat_scaled_jtd", "net_short_mat_scaled_jtd",
         "rw_long_contrib", "rw_short_contrib"]
    ]
    top_short = od.nlargest(min(10, len(od)), "rw_short_contrib")[
        ["Issuer", "DRC bucket", "credit_quality_category", "RW",
         "net_long_mat_scaled_jtd", "net_short_mat_scaled_jtd",
         "rw_long_contrib", "rw_short_contrib"]
    ]

    hbr_in_range = bool(((bucket_df["hbr"] >= 0.0) & (bucket_df["hbr"] <= 1.0)).all())
    drc_b_nonneg = bool((bucket_df["drc_b"] >= 0.0).all())

    write_phase_snapshot(
        path,
        phase_num=4,
        phase_name="Bucket HBR + bucket capital",
        mar_ref="MAR22.23 (HBR formula), MAR22.25 (DRC_b = Σ_long RW*net - HBR*Σ_short RW*|net|)",
        source_module="drc_nonsecuritisation/nonsec_engine.py",
        input_df=obligor_df,
        output_df=bucket_df,
        audit={
            "top_long_RW_contributions": top_long,
            "top_short_RW_contributions": top_short,
        },
        reconciliation=[
            ("rows_in (phase 3 obligor x bucket)", int(len(obligor_df))),
            ("rows_out (one per bucket)", int(len(bucket_df))),
            ("hbr_in_[0,1]_for_all_buckets", hbr_in_range),
            ("drc_b_nonneg_for_all_buckets", drc_b_nonneg),
            ("sum_drc_b_raw", float(bucket_df["drc_b_raw"].sum())),
            ("sum_drc_b_clipped", float(bucket_df["drc_b"].sum())),
            ("buckets_with_negative_drc_b_raw",
                int((bucket_df["drc_b_raw"] < 0).sum())),
        ],
        notes="HBR is computed on maturity-scaled net JTDs, NOT on "
              "RW-multiplied amounts. drc_b is floored at 0 per MAR22.25; "
              "drc_b_raw preserves the unfloored value for audit.",
    )


def write_phase5_snapshot(
    bucket_df: pd.DataFrame, total_drc: float,
    obligor_df: pd.DataFrame, positions_df: pd.DataFrame,
    path: str = OUTPUT_PHASE5_TOTAL,
) -> None:
    """Phase 5 — Total non-sec DRC (MAR22.26).

    `bucket_df`    = phase 4 output.
    `output_df`    = a single-row total summary; bucket detail in audit.
    """
    import sys
    here = os.path.dirname(__file__)
    parent = os.path.abspath(os.path.join(here, ".."))
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from phase_snapshot import write_phase_snapshot

    summary = pd.DataFrame([{
        "metric": "TOTAL NON-SEC DRC (MAR22.26)",
        "value_$": total_drc,
        "n_phase1_positions": int(positions_df["is_drc_relevant"].sum()),
        "n_obligors": int(len(obligor_df)),
        "n_buckets": int(len(bucket_df)),
    }])

    bucket_ranking = bucket_df.sort_values("drc_b", ascending=False)[
        ["DRC bucket", "n_obligors",
         "sum_long_mat_scaled_jtd", "sum_short_mat_scaled_jtd",
         "sum_rw_long", "sum_rw_short", "hbr",
         "drc_b_raw", "drc_b"]
    ]

    write_phase_snapshot(
        path,
        phase_num=5,
        phase_name="Total non-sec DRC",
        mar_ref="MAR22.26 (sum of per-bucket capital, no cross-bucket hedging, no negatives)",
        source_module="drc_nonsecuritisation/nonsec_engine.py",
        input_df=bucket_df,
        output_df=summary,
        audit={"bucket_ranking_by_drc_b": bucket_ranking},
        reconciliation=[
            ("buckets_in", int(len(bucket_df))),
            ("sum_drc_b", float(bucket_df["drc_b"].sum())),
            ("total_drc_reported", float(total_drc)),
            ("totals_match (tol 1e-6)",
                abs(float(bucket_df["drc_b"].sum()) - float(total_drc)) < 1e-6),
            ("any_bucket_drc_b_negative",
                bool((bucket_df["drc_b"] < 0).any())),
        ],
        notes="MAR22.26: total = Σ_b max(DRC_b, 0). No cross-bucket hedging; "
              "no bucket capital below zero.",
    )


# Back-compat shim: callers may invoke write_snapshot() to emit both
# phase-4 and phase-5 outputs together.
def write_snapshot(
    positions_df: pd.DataFrame, obligor_df: pd.DataFrame,
    bucket_df: pd.DataFrame, total_drc: float,
    path: str = OUTPUT_SNAPSHOT,
) -> None:
    write_phase4_snapshot(obligor_df, bucket_df, path=path)
    write_phase5_snapshot(bucket_df, total_drc, obligor_df, positions_df,
                          path=OUTPUT_PHASE5_TOTAL)


def main() -> None:
    df, _cfg = build_nonsec_loader_frame()
    df = add_jtd_columns(df)
    obligor = aggregate_net_per_obligor(df)
    bucket_df, total = run_engine(obligor)
    print(summarise(df, obligor, bucket_df, total))
    write_phase4_snapshot(obligor, bucket_df)
    write_phase5_snapshot(bucket_df, total, obligor, df)
    print(f"\nPhase 4 snapshot: {os.path.normpath(OUTPUT_SNAPSHOT)}")
    print(f"Phase 5 snapshot: {os.path.normpath(OUTPUT_PHASE5_TOTAL)}")


if __name__ == "__main__":
    main()
