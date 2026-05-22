"""
nonsec_jtd.py — Per-position gross JTD computation per MAR22.11-14.

Sign convention (MAR22.13):
  notional positive (long) / negative (short)
  P&L = signed_MV - signed_notional
  JTD_long  = max(LGD * notional + P&L, 0)
  JTD_short = min(LGD * notional + P&L, 0)
  min(., 0) is a CEILING at zero, NOT a floor.
  Short positions can and must produce NEGATIVE gross JTDs that participate
  in MAR22.19 offsetting; if the engine ever zero-floors them, hedge
  recognition is broken (per FRTB SA/CLAUDE.md).

For equities (no face value): treat effective_notional = signed_MV so that
P&L collapses to 0 and JTD = LGD * signed_MV = signed_MV (since equity LGD=1).
This matches MAR22.16's treatment of cash equity positions.

Maturity scaling (MAR22.15-18) is applied to produce maturity_scaled_jtd,
which is the input to the bucket-level capital formula in MAR22.25.
  maturity_scaled_jtd = jtd_gross * scaling_factor
  scaling_factor = clip(M_T_years, 0.25, 1.0) — already applied in loader as
                   maturity_scaling_factor column.

NAMING — read this before assuming "weighted" means "risk-weighted":
  This column is NOT risk-weighted. The default risk weight (RW) is applied
  later, at bucket level in MAR22.25 (nonsec_engine.py: RW * net JTD). LGD is
  also NOT applied here — LGD is already inside jtd_gross via the MAR22.12
  `LGD * notional + P&L` term. The ONLY transform between jtd_gross and this
  column is the MAR22.15-18 maturity scaling factor. Hence maturity_scaled_jtd
  (renamed from the misleading legacy name `weighted_jtd`).
"""
from __future__ import annotations
import os
import pandas as pd

from .nonsec_loader import (
    build_nonsec_loader_frame, OUTPUT_DIR,
)

OUTPUT_SNAPSHOT = os.path.join(OUTPUT_DIR, "drc_nonsec_phase2_gross_jtd.xlsx")


def compute_jtd_gross(LGD: float, notional: float, mv: float, is_long: bool) -> float:
    """Per MAR22.11-13.

    notional and mv are SIGNED (long positive, short negative).
    is_long: True for long positions (uses max), False for shorts (uses min).
    """
    pnl = mv - notional
    raw = LGD * notional + pnl
    if is_long:
        return max(raw, 0.0)
    return min(raw, 0.0)


def compute_jtd_for_row(row: pd.Series) -> float:
    """Apply MAR22.11 to a single position row."""
    LGD = float(row["LGD"])
    notional = float(row["effective_notional"])
    mv = float(row["effective_market_value"])
    is_long = str(row["Long/Short"]).strip().lower() == "long"
    return compute_jtd_gross(LGD, notional, mv, is_long)


def add_jtd_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds jtd_gross and maturity_scaled_jtd columns to the loader frame.

    Operates on phase-1-relevant rows only; excluded rows get NaN JTD.
    """
    df = df.copy()
    jtd_gross = []
    maturity_scaled_jtd = []
    for idx in df.index:
        if not bool(df.at[idx, "is_drc_relevant"]):
            jtd_gross.append(float("nan"))
            maturity_scaled_jtd.append(float("nan"))
            continue
        jtd = compute_jtd_for_row(df.loc[idx])
        scaling = float(df.at[idx, "maturity_scaling_factor"])
        jtd_gross.append(jtd)
        maturity_scaled_jtd.append(jtd * scaling)
    df["jtd_gross"] = pd.Series(jtd_gross, index=df.index, dtype=float)
    df["maturity_scaled_jtd"] = pd.Series(maturity_scaled_jtd, index=df.index, dtype=float)
    return df


# ── SUMMARY / SNAPSHOT ────────────────────────────────────────────────────────

def summarise(df: pd.DataFrame) -> str:
    rel = df[df["is_drc_relevant"]].copy()
    lines = []
    lines.append(f"JTD computed for {len(rel)} of {len(df)} positions (phase-1 relevant).")
    lines.append("")
    lines.append("Aggregate JTD by bucket and direction:")
    rel["direction"] = rel["jtd_gross"].apply(
        lambda j: "long" if j > 0 else ("short" if j < 0 else "zero")
    )
    agg = rel.groupby(["DRC bucket", "direction"]).agg(
        n=("ID", "count"),
        sum_jtd=("jtd_gross", "sum"),
        sum_mat_scaled=("maturity_scaled_jtd", "sum"),
    )
    lines.append(agg.to_string())
    lines.append("")
    lines.append("Top 10 by |maturity_scaled_jtd|:")
    rel = rel.assign(abs_w=rel["maturity_scaled_jtd"].abs())
    top = rel.nlargest(10, "abs_w")[
        ["ID", "Issuer", "Security", "Position Type", "Long/Short",
         "seniority_class", "credit_quality_category",
         "effective_notional", "effective_market_value",
         "LGD", "RW", "M_T_years", "jtd_gross", "maturity_scaled_jtd"]
    ]
    lines.append(top.to_string(index=False))
    return "\n".join(lines)


def write_snapshot(
    df_in: pd.DataFrame, df_out: pd.DataFrame, path: str = OUTPUT_SNAPSHOT,
) -> None:
    """Phase 2 — Gross JTD per leg (MAR22.11-14).

    `df_in`  = phase 1 output (loader frame; legs already decomposed upstream).
    `df_out` = same rows + jtd_gross + maturity_scaled_jtd columns. NaN JTD on rows
               where is_drc_relevant is False (matured positions).
    """
    import sys
    here = os.path.dirname(__file__)
    parent = os.path.abspath(os.path.join(here, ".."))
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from phase_snapshot import write_phase_snapshot

    rel = df_out[df_out["is_drc_relevant"]].copy()
    rel["direction"] = rel["jtd_gross"].apply(
        lambda j: "long" if j > 0 else ("short" if j < 0 else "zero")
    )
    by_bucket = rel.groupby(["DRC bucket", "direction"]).agg(
        n=("ID", "count"),
        sum_jtd=("jtd_gross", "sum"),
        sum_mat_scaled=("maturity_scaled_jtd", "sum"),
    ).reset_index()
    rel = rel.assign(abs_w=rel["maturity_scaled_jtd"].abs())
    top10 = rel.nlargest(10, "abs_w")[
        ["ID", "Issuer", "Security", "Position Type", "Long/Short",
         "seniority_class", "credit_quality_category",
         "effective_notional", "effective_market_value",
         "LGD", "RW", "M_T_years", "jtd_gross", "maturity_scaled_jtd"]
    ]

    sum_long = float(rel.loc[rel["jtd_gross"] > 0, "jtd_gross"].sum())
    sum_short = float(rel.loc[rel["jtd_gross"] < 0, "jtd_gross"].sum())
    sum_long_w = float(rel.loc[rel["maturity_scaled_jtd"] > 0, "maturity_scaled_jtd"].sum())
    sum_short_w = float(rel.loc[rel["maturity_scaled_jtd"] < 0, "maturity_scaled_jtd"].sum())

    write_phase_snapshot(
        path,
        phase_num=2,
        phase_name="Gross JTD per leg",
        mar_ref="MAR22.11-14 (sign convention, max/min, maturity scaling)",
        source_module="drc_nonsecuritisation/nonsec_jtd.py",
        input_df=df_in,
        output_df=df_out,
        audit={
            "by_bucket_direction": by_bucket,
            "top10_by_abs_maturity_scaled_jtd": top10,
        },
        reconciliation=[
            ("rows_in (phase 1 output)", int(len(df_in))),
            ("rows_out (with JTD columns)", int(len(df_out))),
            ("phase1_relevant_count", int(df_out["is_drc_relevant"].sum())),
            ("rows_with_nan_jtd",
                int(df_out["jtd_gross"].isna().sum())),
            ("sum_jtd_gross_long", sum_long),
            ("sum_jtd_gross_short", sum_short),
            ("sum_jtd_gross_net", sum_long + sum_short),
            ("sum_mat_scaled_jtd_long", sum_long_w),
            ("sum_mat_scaled_jtd_short", sum_short_w),
        ],
        notes="Short positions can and must produce negative gross JTDs "
              "(min(.,0) is a CEILING at zero, not a floor — see project rule).",
    )


def main() -> None:
    df, cfg = build_nonsec_loader_frame()
    df_in = df.copy()
    df = add_jtd_columns(df)
    print(summarise(df))
    write_snapshot(df_in, df)
    print(f"\nPhase 2 snapshot written: {os.path.normpath(OUTPUT_SNAPSHOT)}")


if __name__ == "__main__":
    main()
