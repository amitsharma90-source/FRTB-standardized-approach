"""
nonsec_netting.py — Within-obligor MAR22.19 asymmetric netting.

Rule: gross JTD risk positions of long and short exposures to the SAME OBLIGOR
may be offset where the short has the SAME OR LOWER seniority relative to the
long. Higher rank = more senior:
  rank 1 = equity (lowest)
  rank 2 = non-senior debt
  rank 3 = senior debt
  rank 4 = covered bond (highest)

Algorithm:
  for each obligor:
    1. Aggregate signed JTDs by seniority_rank within the obligor.
    2. Walk shorts from rank 1 upward; each short at rank R can offset longs
       at rank >= R. Consume shorts greedily.
    3. After offsetting, sum positive net JTDs as obligor net_long and
       negative net JTDs as obligor net_short.

Equity-rank shorts can offset bonds (longs at higher rank); bond-rank shorts
canNOT offset equity-rank longs. This is the "asymmetric netting" rule the
project's CLAUDE.md emphasises for the callable-bond decomposition.

Inputs: phase-1 relevant rows from nonsec_jtd (jtd_gross + maturity_scaled_jtd).
Outputs: per-obligor net_long_jtd / net_short_jtd / maturity-scaled variants, plus
         per-(obligor, rank) detail for audit.
"""
from __future__ import annotations
import os
import pandas as pd

from .nonsec_loader import build_nonsec_loader_frame, OUTPUT_DIR
from .nonsec_jtd import add_jtd_columns

OUTPUT_SNAPSHOT = os.path.join(OUTPUT_DIR, "drc_nonsec_phase3_net_jtd.xlsx")
RANKS = [1, 2, 3, 4]  # equity / non-senior / senior / covered


def net_within_obligor(
    obligor_rows: pd.DataFrame,
) -> tuple[dict[int, float], dict[int, float]]:
    """For one obligor, returns (net_jtd_per_rank, net_mat_scaled_jtd_per_rank)
    after applying MAR22.19 asymmetric short->long offsetting.

    Both dicts have keys in RANKS. A positive value at rank R means the
    obligor still has a net long position at that rank; a negative value
    means a net short remains.
    """
    # Step 1: aggregate signed gross JTD (and maturity-scaled) per rank
    net_jtd: dict[int, float] = {r: 0.0 for r in RANKS}
    net_w_jtd: dict[int, float] = {r: 0.0 for r in RANKS}
    for _, row in obligor_rows.iterrows():
        r = int(row["seniority_rank"])
        net_jtd[r] += float(row["jtd_gross"])
        net_w_jtd[r] += float(row["maturity_scaled_jtd"])

    # Step 2: greedy short->long offset, walking ranks low to high
    # A short at rank R can offset longs at any rank >= R.
    for R in sorted(RANKS):
        short_avail = max(0.0, -net_jtd[R])  # |net short| at this rank
        short_avail_w = max(0.0, -net_w_jtd[R])
        if short_avail <= 0:
            continue
        # Allocate to longs at higher ranks (R+1, R+2, ...)
        for L in sorted([x for x in RANKS if x > R]):
            long_avail = max(0.0, net_jtd[L])
            if long_avail <= 0:
                continue
            offset = min(short_avail, long_avail)
            # Apply same proportional offset on maturity-scaled JTD scale
            if long_avail > 0:
                offset_w = (offset / long_avail) * max(0.0, net_w_jtd[L])
            else:
                offset_w = 0.0
            net_jtd[L] -= offset
            net_jtd[R] += offset
            net_w_jtd[L] -= offset_w
            net_w_jtd[R] += offset_w
            short_avail -= offset
            short_avail_w -= offset_w
            if short_avail <= 0:
                break

    return net_jtd, net_w_jtd


def aggregate_net_per_obligor(df: pd.DataFrame) -> pd.DataFrame:
    """Returns one row per (obligor, bucket) with net_long_jtd / net_short_jtd
    and maturity-scaled variants — ready for bucket aggregation in nonsec_engine.
    """
    rel = df[df["is_drc_relevant"]].copy()
    if rel.empty:
        return pd.DataFrame(columns=[
            "Issuer", "DRC bucket",
            "net_long_jtd", "net_short_jtd",
            "net_long_mat_scaled_jtd", "net_short_mat_scaled_jtd",
            "credit_quality_category",
        ])

    out_rows = []
    # Group by the canonical legal entity (`obligor`), NOT the per-leg issuer
    # string. The loader resolves share-class / index-naming variants
    # (e.g. "Alphabet Class A/C" -> "Alphabet") via the config Issuer_Alias_Map
    # so MAR22.19 recognises a long single-name vs short index-constituent
    # hedge in the same legal entity. The emitted identity column stays named
    # "Issuer" (it is the obligor identity of the netted row) so downstream
    # consumers need no change.
    for (obligor, bucket), grp in rel.groupby(["obligor", "DRC bucket"], dropna=False):
        net_jtd, net_w_jtd = net_within_obligor(grp)
        net_long = sum(max(0.0, v) for v in net_jtd.values())
        net_short = -sum(max(0.0, -v) for v in net_jtd.values())
        net_long_w = sum(max(0.0, v) for v in net_w_jtd.values())
        net_short_w = -sum(max(0.0, -v) for v in net_w_jtd.values())
        # Credit quality: take the most-frequent (or first) within obligor.
        # In the v5 portfolio the same obligor has the same rating across
        # positions, so this is unambiguous; if it isn't, the loader should
        # surface a warning.
        cqc = grp["credit_quality_category"].mode().iloc[0]
        # RW: lookup for that quality
        rw = float(grp["RW"].iloc[0])
        out_rows.append({
            "Issuer": obligor,
            "DRC bucket": bucket,
            "credit_quality_category": cqc,
            "RW": rw,
            "n_positions": len(grp),
            "net_long_jtd": net_long,
            "net_short_jtd": net_short,
            "net_long_mat_scaled_jtd": net_long_w,
            "net_short_mat_scaled_jtd": net_short_w,
            # per-rank detail for audit
            "net_jtd_rank1_equity": net_jtd[1],
            "net_jtd_rank2_nonsen": net_jtd[2],
            "net_jtd_rank3_senior": net_jtd[3],
            "net_jtd_rank4_covered": net_jtd[4],
        })
    return pd.DataFrame(out_rows)


# ── SUMMARY / SNAPSHOT ────────────────────────────────────────────────────────

def summarise(df_positions: pd.DataFrame, df_obligor: pd.DataFrame) -> str:
    lines = []
    lines.append(f"Netted to {len(df_obligor)} (obligor, bucket) rows from "
                 f"{df_positions['is_drc_relevant'].sum()} phase-1 positions.")
    lines.append("")
    lines.append("Bucket-level totals (after within-obligor netting):")
    bkt = df_obligor.groupby("DRC bucket").agg(
        n_obligors=("Issuer", "count"),
        sum_net_long_jtd=("net_long_jtd", "sum"),
        sum_net_short_jtd=("net_short_jtd", "sum"),
        sum_net_long_mat_scaled=("net_long_mat_scaled_jtd", "sum"),
        sum_net_short_mat_scaled=("net_short_mat_scaled_jtd", "sum"),
    )
    lines.append(bkt.to_string())
    lines.append("")
    lines.append("Obligors with non-trivial netting (net_short < 0 OR partial offset):")
    audit = df_obligor[
        (df_obligor["net_short_jtd"] < 0) | (df_obligor["n_positions"] > 1)
    ].copy()
    if not audit.empty:
        cols = ["Issuer", "DRC bucket", "credit_quality_category",
                "n_positions", "net_long_jtd", "net_short_jtd",
                "net_jtd_rank1_equity", "net_jtd_rank2_nonsen",
                "net_jtd_rank3_senior", "net_jtd_rank4_covered"]
        lines.append(audit[cols].to_string(index=False))
    return "\n".join(lines)


def write_snapshot(df_positions: pd.DataFrame, df_obligor: pd.DataFrame,
                   path: str = OUTPUT_SNAPSHOT) -> None:
    """Phase 3 — Within-obligor net JTD (MAR22.19 asymmetric netting).

    `df_positions` = phase 2 output (per-row JTD frame).
    `df_obligor`   = per-(obligor, bucket) netted frame consumed by phase 4.
    """
    import sys
    here = os.path.dirname(__file__)
    parent = os.path.abspath(os.path.join(here, ".."))
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from phase_snapshot import write_phase_snapshot

    rank_detail_cols = [
        "Issuer", "DRC bucket", "credit_quality_category", "n_positions",
        "net_long_jtd", "net_short_jtd",
        "net_jtd_rank1_equity", "net_jtd_rank2_nonsen",
        "net_jtd_rank3_senior", "net_jtd_rank4_covered",
    ]
    obligor_with_netting = df_obligor[
        (df_obligor["net_short_jtd"] < 0) | (df_obligor["n_positions"] > 1)
    ][rank_detail_cols]
    bucket_totals = df_obligor.groupby("DRC bucket").agg(
        n_obligors=("Issuer", "count"),
        sum_net_long_jtd=("net_long_jtd", "sum"),
        sum_net_short_jtd=("net_short_jtd", "sum"),
        sum_net_long_mat_scaled=("net_long_mat_scaled_jtd", "sum"),
        sum_net_short_mat_scaled=("net_short_mat_scaled_jtd", "sum"),
    ).reset_index()

    rel = df_positions[df_positions["is_drc_relevant"]]
    sum_gross_long = float(rel.loc[rel["jtd_gross"] > 0, "jtd_gross"].sum())
    sum_gross_short = float(rel.loc[rel["jtd_gross"] < 0, "jtd_gross"].sum())
    sum_net_long = float(df_obligor["net_long_jtd"].sum())
    sum_net_short = float(df_obligor["net_short_jtd"].sum())

    write_phase_snapshot(
        path,
        phase_num=3,
        phase_name="Within-obligor net JTD (asymmetric netting)",
        mar_ref="MAR22.19 (short at rank R offsets longs at rank >= R)",
        source_module="drc_nonsecuritisation/nonsec_netting.py",
        input_df=df_positions,
        output_df=df_obligor,
        audit={
            "obligors_with_non_trivial_netting": obligor_with_netting,
            "bucket_totals_after_netting": bucket_totals,
        },
        reconciliation=[
            ("rows_in (phase 2 positions)", int(len(df_positions))),
            ("rows_in_phase1_relevant", int(rel.shape[0])),
            ("rows_out (obligor x bucket)", int(len(df_obligor))),
            ("sum_gross_long_jtd", sum_gross_long),
            ("sum_gross_short_jtd", sum_gross_short),
            ("sum_gross_net_jtd", sum_gross_long + sum_gross_short),
            ("sum_net_long_jtd_obligor", sum_net_long),
            ("sum_net_short_jtd_obligor", sum_net_short),
            ("sum_net_jtd_obligor_total", sum_net_long + sum_net_short),
            ("conservation_check (gross net == obligor net, tol 1e-6)",
                abs((sum_gross_long + sum_gross_short)
                    - (sum_net_long + sum_net_short)) < 1e-6),
        ],
        notes="MAR22.19 is mass-conserving: total signed gross JTD must equal "
              "total signed obligor-net JTD (the offset moves mass from longs "
              "to shorts at the same obligor; total Σ stays the same).",
    )


def main() -> None:
    df, cfg = build_nonsec_loader_frame()
    df = add_jtd_columns(df)
    obligor = aggregate_net_per_obligor(df)
    print(summarise(df, obligor))
    write_snapshot(df, obligor)
    print(f"\nPhase 3 snapshot written: {os.path.normpath(OUTPUT_SNAPSHOT)}")


if __name__ == "__main__":
    main()
