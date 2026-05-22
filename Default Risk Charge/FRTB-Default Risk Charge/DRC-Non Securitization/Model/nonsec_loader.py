"""
nonsec_loader.py — DRC non-sec phase 1.

Reads:
  * `Portfolio_MV_Decomposed` (Sheet 2 of FRTB_Sensitivities.xlsx) — per-leg
    model-priced MV blotter emitted by the sensitivity engine. Callable bonds
    are already split into vanilla + short_call. Equity index options are
    already exploded into per-constituent legs. This is the single source of
    MV for DRC; the holdings-file `Market Value ($)` column is intentionally
    NOT consumed.
  * `Combined Holdings` (FRTB_Combined_Portfolio_v5.xlsx) — joined to Sheet 2
    on `parent_position_id` to pull metadata not in Sheet 2 (DRC bucket,
    Maturity, Call Date, Issue Date, etc.).
  * `FRTB_NonSec_Config.xlsx` — seniority / LGD / RW / rating lookups.

Loader emits a per-leg frame with derived fields ready for nonsec_jtd /
nonsec_netting / nonsec_engine:
  Quantity/Notional       signed notional from Sheet 2, float-normalised
                            (single canonical notional; no signed_* alias)
  seniority_class         from (Position Type, Seniority) lookup
  seniority_rank          equity=1, non-sen=2, senior=3, covered=4
  LGD                     from seniority_class lookup
  credit_quality_category from Rating normalisation
  RW                      from credit_quality_category lookup
  M_T_years               from Maturity, clipped to [0.25, 1.0] per MAR22.15-18
  effective_notional      face for bond legs, 0 for option legs (incl.
                            short_call leg of a callable and index constituent
                            legs of an SPX option), signed MV for cash equity
  effective_market_value  single canonical MV: signed model NPV from Sheet 2
                            `MV_USD`, float, NaN->0 (raw MV_USD not retained)
  is_drc_relevant         True for all decomposed + undecomposed non-sec rows
                            with a positive maturity scaling factor; False for
                            matured positions only (Sheet 2 has already removed
                            the unsupported-instrument-type rows by virtue of
                            being emitted only for classified instruments).

The loader does no JTD math; that's phase 2.
"""
from __future__ import annotations
import os
import datetime as dt
import pandas as pd

# ── PATHS ─────────────────────────────────────────────────────────────────────

PORTFOLIO_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..",
    "Input data", "FRTB_Combined_Portfolio_v5.xlsx",
))
CONFIG_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..",
    "Input data", "FRTB_NonSec_Config.xlsx",
))
SENSITIVITIES_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..",
    "Output data", "Sensitivities output", "FRTB_Sensitivities.xlsx",
))
OUTPUT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..",
    "Output data", "SA capital charge",
))
OUTPUT_SNAPSHOT = os.path.join(OUTPUT_DIR, "drc_nonsec_phase1_load.xlsx")

DEFAULT_ASOF = dt.date(2026, 2, 5)

PORTFOLIO_SHEET = "Combined Holdings"
MV_SHEET = "Portfolio_MV_Decomposed"

CONFIG_SHEETS = [
    "NonSec_LGDs", "NonSec_RW", "NonSec_Buckets", "NonSec_Constants",
    "Seniority_Map", "Rating_Map", "Issuer_Alias_Map",
]

# Columns supplied by Sheet 2 — for these, the leg's value takes precedence
# over the parent's value during the holdings join.
SHEET2_OVERRIDE_COLS = {
    "Issuer", "Security", "Position Type", "Issue Type", "Currency",
    "Long/Short", "Rating", "Seniority (DRC)", "Quantity/Notional",
}


# ── LOADERS ───────────────────────────────────────────────────────────────────

def load_portfolio_holdings(path: str = PORTFOLIO_PATH) -> pd.DataFrame:
    """Read Combined Holdings indexed by ID. No filtering."""
    df = pd.read_excel(path, sheet_name=PORTFOLIO_SHEET)
    df = df[df["ID"].apply(
        lambda x: str(x).replace(".", "").isdigit() if pd.notna(x) else False
    )].copy()
    df["ID"] = df["ID"].astype(int)
    df = df.set_index("ID")
    return df


def load_mv_decomposed_sheet(path: str = SENSITIVITIES_PATH) -> pd.DataFrame:
    """Read Sheet 2 of FRTB_Sensitivities.xlsx (`Portfolio_MV_Decomposed`).

    Sheet 2 has one row per leg; deterministic ordering is preserved on read.
    """
    df = pd.read_excel(path, sheet_name=MV_SHEET)
    # Treat empty decomposition_leg as the canonical empty string
    if "decomposition_leg" in df.columns:
        df["decomposition_leg"] = df["decomposition_leg"].fillna("").astype(str)
    df["ID"] = df["ID"].astype(int)
    df["parent_position_id"] = df["parent_position_id"].astype(int)
    return df


def load_nonsec_config(path: str = CONFIG_PATH) -> dict[str, pd.DataFrame]:
    return {name: pd.read_excel(path, sheet_name=name) for name in CONFIG_SHEETS}


# ── JOIN ──────────────────────────────────────────────────────────────────────

def join_legs_to_holdings(
    mv_legs: pd.DataFrame, holdings: pd.DataFrame,
) -> pd.DataFrame:
    """For each Sheet 2 leg, merge in parent's holdings columns.

    Sheet 2's leg-specific values (Issuer for an SPX constituent, Long/Short
    for a short_call leg, etc.) take precedence over the parent's. Columns
    that don't exist in Sheet 2 (DRC bucket, Maturity, Call Date, Issue Date,
    Pool ID, etc.) are sourced from the parent's holdings row.
    """
    parents = holdings.copy()
    parent_cols_to_take = [c for c in parents.columns if c not in SHEET2_OVERRIDE_COLS]
    parents = parents[parent_cols_to_take]

    merged = mv_legs.merge(
        parents,
        how="left",
        left_on="parent_position_id",
        right_index=True,
        suffixes=("", "_parent"),
    )
    missing = merged[merged.iloc[:, len(mv_legs.columns)].isna()
                     if len(parents.columns) > 0 else []]
    if not missing.empty and len(parents.columns) > 0:
        bad_ids = sorted(set(missing["parent_position_id"].tolist()))[:10]
        print(f"[nonsec_loader] WARNING: {len(missing)} legs have parent_position_id "
              f"not in holdings (first 10: {bad_ids}). Holdings-derived columns will be NaN.")
    return merged


# ── DERIVATIONS ───────────────────────────────────────────────────────────────

def lookup_seniority(seniority_map: pd.DataFrame, position_type: str,
                     seniority_label: str) -> tuple[str, int]:
    pt = (str(position_type) if position_type else "").strip()
    sl = (str(seniority_label) if seniority_label else "").strip()
    sm = seniority_map[
        (seniority_map["position_type"] == pt)
        & (seniority_map["seniority_label"] == sl)
    ]
    if sm.empty:
        return "non_senior_debt", 2
    row = sm.iloc[0]
    return str(row["seniority_class"]), int(row["seniority_rank"])


def normalise_rating(rating_map: pd.DataFrame, raw_rating) -> str:
    if raw_rating is None or (isinstance(raw_rating, float) and pd.isna(raw_rating)):
        return "Unrated"
    raw = str(raw_rating).strip().upper()
    if raw in ("", "NAN"):
        return "Unrated"
    rm = rating_map[rating_map["raw_rating"].astype(str).str.upper() == raw]
    if rm.empty:
        return "Unrated"
    return str(rm.iloc[0]["credit_quality_category"])


def lookup_lgd(lgd_table: pd.DataFrame, seniority_class: str) -> float:
    row = lgd_table[lgd_table["seniority_class"] == seniority_class]
    if row.empty:
        return 1.00
    return float(row.iloc[0]["LGD"])


def lookup_rw(rw_table: pd.DataFrame, credit_quality: str) -> float:
    row = rw_table[rw_table["credit_quality_category"] == credit_quality]
    if row.empty:
        return float(rw_table[rw_table["credit_quality_category"] == "Unrated"].iloc[0]["RW_decimal"])
    return float(row.iloc[0]["RW_decimal"])


def compute_maturity_years(maturity, asof: dt.date,
                           floor: float, cap: float, equity_default: float,
                           position_type: str) -> float:
    pt = (str(position_type) if position_type else "").strip()
    has_maturity = not (maturity is None or pd.isna(maturity))
    if not has_maturity:
        # No contractual maturity — two regulatorily distinct cases:
        #  • Genuine cash equity (a stock): MAR22.16 lets the bank elect a
        #    maturity of either >1y or 3 months at its discretion. That
        #    election is `equity_default` (configured = 0.25, the 3-month
        #    election, applied as a consistent desk policy per MAR22.16).
        #  • Anything else missing a maturity (e.g. a perpetual, or a data
        #    gap): NOT eligible for the 22.16 cash-equity election. Fall back
        #    to the conservative full-horizon cap (1y) so a blank maturity can
        #    never *understate* DRC once the equity election is sub-1y.
        if pt == "Equity":
            return float(equity_default)
        return float(cap)
    # Has a contractual maturity. This includes equity legs generated by
    # MAR22.5 look-through of an equity index option: they take the OPTION
    # CONTRACT's maturity (MAR22.18 + the derivative-maturity principle —
    # an equity *derivative* is not cash equity), NOT the 22.16 election.
    asof_ts = pd.Timestamp(asof)
    delta_days = (pd.to_datetime(maturity) - asof_ts).days
    if delta_days <= 0:
        return 0.0
    years = delta_days / 365.25
    return max(float(floor), min(float(cap), years))


# ── ORCHESTRATOR ──────────────────────────────────────────────────────────────

def build_nonsec_loader_frame(
    portfolio_path: str = PORTFOLIO_PATH,
    sensitivities_path: str = SENSITIVITIES_PATH,
    config_path: str = CONFIG_PATH,
    asof: dt.date = DEFAULT_ASOF,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    holdings = load_portfolio_holdings(portfolio_path)
    mv_legs = load_mv_decomposed_sheet(sensitivities_path)
    cfg = load_nonsec_config(config_path)

    df = join_legs_to_holdings(mv_legs, holdings)

    # Filter to non-sec rows by inherited DRC bucket
    mask = df["DRC bucket"].notna() & df["DRC bucket"].astype(str).str.startswith("NonSec:")
    df = df.loc[mask].copy().reset_index(drop=True)

    constants = cfg["NonSec_Constants"].set_index("Constant")["Value"]
    M_T_floor = float(constants.at["maturity_floor_years"])
    M_T_cap = float(constants.at["maturity_cap_years"])
    equity_default_M_T = float(constants.at["equity_default_maturity"])

    # Sheet 2 already emits signed amounts (long +, short -). Keep exactly ONE
    # canonical column per quantity — no float-cast aliases:
    #   Quantity/Notional      signed notional, float-normalised in place
    #   effective_market_value single MV column: signed model NPV, float,
    #                          NaN->0 (null-safe for the JTD min/max)
    # Raw Sheet-2 `MV_USD` is folded into effective_market_value and dropped;
    # it had no other consumer in the non-sec branch.
    df["Quantity/Notional"] = df["Quantity/Notional"].astype(float)
    df["effective_market_value"] = df["MV_USD"].astype(float).fillna(0.0)
    df = df.drop(columns=["MV_USD"])

    # Canonical obligor for MAR22.19 within-obligor netting. DRC offsets long
    # vs short by LEGAL ENTITY, not by the per-leg issuer string. The MAR22.5
    # index look-through emits constituent legs under the index provider's
    # naming ("Alphabet Class A/C", "Berkshire Hathaway B", "Meta Platforms");
    # the single-name book uses the parent-entity short form. The
    # Issuer_Alias_Map (config, not hardcoded) maps the former to the latter;
    # strings absent from the map are their own canonical obligor (identity).
    #
    # `obligor` is NOT a duplicate of `Issuer`: it is a genuine semantic
    # transform (display label -> legal-entity netting key), differing from
    # `Issuer` precisely on the aliased rows, with a distinct consumer
    # (nonsec_netting groups on `obligor`; `Issuer` stays the per-leg label
    # used by per-position audit/attribution). Same exemption class as
    # `effective_notional` per the FRTB SA "no duplicate columns" rule.
    alias_map = cfg["Issuer_Alias_Map"]
    alias = dict(zip(
        alias_map["raw_issuer"].astype(str).str.strip(),
        alias_map["canonical_issuer"].astype(str).str.strip(),
    ))
    _iss = df["Issuer"].astype(str).str.strip()
    df["obligor"] = _iss.map(lambda s: alias.get(s, s))

    # Seniority lookup (uses leg's Position Type + Seniority — already
    # overridden in Sheet 2 for index constituents).
    snr = df.apply(
        lambda r: lookup_seniority(cfg["Seniority_Map"], r["Position Type"], r["Seniority (DRC)"]),
        axis=1, result_type="expand",
    )
    df["seniority_class"] = snr[0]
    df["seniority_rank"] = snr[1]

    # LGD
    df["LGD"] = df["seniority_class"].apply(
        lambda s: lookup_lgd(cfg["NonSec_LGDs"], s)
    )

    # Credit quality + RW
    df["credit_quality_category"] = df["Rating"].apply(
        lambda r: normalise_rating(cfg["Rating_Map"], r)
    )
    df["RW"] = df["credit_quality_category"].apply(
        lambda q: lookup_rw(cfg["NonSec_RW"], q)
    )

    # Maturity scaling
    df["M_T_years"] = df.apply(
        lambda r: compute_maturity_years(
            r["Maturity"], asof, M_T_floor, M_T_cap, equity_default_M_T,
            r["Position Type"]
        ),
        axis=1,
    )
    df["maturity_scaling_factor"] = df["M_T_years"].clip(lower=M_T_floor, upper=M_T_cap)

    # Inclusion mask — Sheet 2 already represents only classified instruments;
    # exclude only matured positions.
    df["is_drc_relevant"] = df["M_T_years"] > 0.0

    # effective_notional for the JTD formula (MAR22.11-16). effective_market_value
    # is the single canonical MV column, already set above. Branch PRECEDENCE is
    # preserved exactly: a decomposed leg passes its signed notional through even
    # when it is also an equity/bond leg; cash equity (MAR22.16) uses signed MV;
    # an undecomposed single-name equity option (MAR22.14(1)(c)) uses 0.
    df["effective_notional"] = pd.NA

    pt = df["Position Type"].astype(str)
    it = df["Issue Type"].astype(str)
    leg = df["decomposition_leg"].fillna("").astype(str)
    is_decomposed_leg = leg.isin(("vanilla", "short_call", "index_constituent"))
    is_cash_equity = (pt == "Equity") & (it == "Corp")
    is_equity_option = (pt == "Equity") & (it == "Call Option")

    for idx in df.index:
        signed_n = float(df.at[idx, "Quantity/Notional"])
        if is_decomposed_leg.at[idx]:
            df.at[idx, "effective_notional"] = signed_n
        elif is_cash_equity.at[idx]:
            df.at[idx, "effective_notional"] = float(df.at[idx, "effective_market_value"])
        elif is_equity_option.at[idx]:
            # Undecomposed single-name option (no upstream look-through).
            df.at[idx, "effective_notional"] = 0.0
        else:  # bonds and everything else: pass signed notional through
            df.at[idx, "effective_notional"] = signed_n

    df["effective_notional"] = df["effective_notional"].astype(float)

    return df, cfg


# ── VALIDATION + SUMMARY ──────────────────────────────────────────────────────

def validate(df: pd.DataFrame) -> list[str]:
    issues = []
    required = [
        "ID", "parent_position_id", "decomposition_leg",
        "Issuer", "obligor", "Position Type", "DRC bucket",
        "Quantity/Notional",
        "seniority_class", "seniority_rank", "LGD",
        "credit_quality_category", "RW", "M_T_years",
        "effective_notional", "effective_market_value",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        issues.append(f"Missing required columns: {missing}")
    return issues


def summarise(df: pd.DataFrame) -> str:
    lines = []
    lines.append(f"Loaded {len(df)} non-sec DRC legs from Sheet 2 + holdings join.")
    lines.append("")
    lines.append("By decomposition leg kind:")
    lines.append(df["decomposition_leg"].fillna("").value_counts().to_string())
    lines.append("")
    lines.append("By DRC bucket:")
    lines.append(df["DRC bucket"].value_counts().to_string())
    lines.append("")
    lines.append("By Position Type:")
    lines.append(df["Position Type"].value_counts().to_string())
    lines.append("")
    lines.append("By seniority class:")
    lines.append(df["seniority_class"].value_counts().to_string())
    lines.append("")
    lines.append("Inclusion:")
    lines.append(f"  is_drc_relevant=True:  {df['is_drc_relevant'].sum()}")
    lines.append(f"  is_drc_relevant=False: {(~df['is_drc_relevant']).sum()}")
    excluded = df[~df["is_drc_relevant"]][
        ["ID", "Security", "Position Type", "Maturity", "M_T_years"]
    ]
    if not excluded.empty:
        lines.append("Excluded positions (matured):")
        lines.append(excluded.to_string(index=False))
    return "\n".join(lines)


def write_snapshot(df: pd.DataFrame, path: str = OUTPUT_SNAPSHOT) -> None:
    """Phase 1 — Position load + MV/decomposition ingest + config enrichment.

    Inputs: Sheet 2 of FRTB_Sensitivities.xlsx + Combined Holdings + NonSec config.
    Output: per-leg enriched frame consumed by phase 2 (gross JTD).
    """
    import sys
    here = os.path.dirname(__file__)
    parent = os.path.abspath(os.path.join(here, ".."))
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from phase_snapshot import write_phase_snapshot

    by_leg_kind = (df["decomposition_leg"].fillna("(undecomposed)").value_counts()
                   .rename_axis("decomposition_leg").reset_index(name="n"))
    by_bucket = df["DRC bucket"].value_counts().rename_axis("DRC bucket").reset_index(name="n")
    by_position_type = df["Position Type"].value_counts().rename_axis("Position Type").reset_index(name="n")
    by_seniority = df["seniority_class"].value_counts().rename_axis("seniority_class").reset_index(name="n")
    by_credit = df.groupby("credit_quality_category").agg(
        n=("ID", "count"),
        rw_pct=("RW", lambda s: round(float(s.iloc[0]) * 100, 2)),
    ).reset_index()
    excluded = df.loc[~df["is_drc_relevant"],
                      ["ID", "Security", "Position Type", "Maturity", "M_T_years"]]
    canon = df.loc[df["obligor"].astype(str) != df["Issuer"].astype(str)]
    issuer_canonicalisation = (
        canon.groupby(["Issuer", "obligor"])
        .agg(n_legs=("ID", "count"))
        .reset_index()
        if not canon.empty else
        pd.DataFrame(columns=["Issuer", "obligor", "n_legs"])
    )

    write_phase_snapshot(
        path,
        phase_num=1,
        phase_name="Position load + MV/decomposition ingest + config enrichment",
        mar_ref=("MAR22.10-13 (sign convention, LGD/RW lookup); "
                 "22.5 / 22.9 / 22.14(1)(c) consumed as already-applied upstream"),
        source_module="drc_nonsecuritisation/nonsec_loader.py",
        input_df=None,
        output_df=df,
        input_files=[
            os.path.normpath(SENSITIVITIES_PATH),
            os.path.normpath(PORTFOLIO_PATH),
            os.path.normpath(CONFIG_PATH),
        ],
        audit={
            "by_decomposition_leg_kind": by_leg_kind,
            "by_DRC_bucket": by_bucket,
            "by_position_type": by_position_type,
            "by_seniority_class": by_seniority,
            "by_credit_quality_category": by_credit,
            "excluded_positions": excluded,
            "issuer_canonicalisation (MAR22.19 obligor)": issuer_canonicalisation,
        },
        reconciliation=[
            ("legs_in (Sheet 2 + holdings join, non-sec filtered)", int(len(df))),
            ("legs_out (enriched)", int(len(df))),
            ("is_drc_relevant_true", int(df["is_drc_relevant"].sum())),
            ("is_drc_relevant_false (matured)", int((~df["is_drc_relevant"]).sum())),
            ("unique_parent_positions", int(df["parent_position_id"].nunique())),
            ("vanilla_legs", int((df["decomposition_leg"] == "vanilla").sum())),
            ("short_call_legs", int((df["decomposition_leg"] == "short_call").sum())),
            ("index_constituent_legs",
                int((df["decomposition_leg"] == "index_constituent").sum())),
            ("sum_signed_notional", float(df["Quantity/Notional"].sum())),
            ("sum_signed_market_value_usd", float(df["effective_market_value"].sum())),
            ("legs_canonicalised_to_a_different_obligor",
                int((df["obligor"].astype(str) != df["Issuer"].astype(str)).sum())),
            ("unique_obligors", int(df["obligor"].nunique())),
        ],
        notes="Loader does no JTD math. MV and instrument decomposition are "
              "consumed from the sensitivity engine's Portfolio_MV_Decomposed "
              "sheet. The holdings-file `Market Value ($)` column is "
              "deliberately not read. Notional/MV are stored as single "
              "canonical columns (Quantity/Notional float; effective_market_value "
              "float, NaN->0); the prior signed_notional / signed_market_value "
              "float-cast aliases and raw MV_USD were consolidated away with no "
              "value change (recon labels kept stable for QA continuity). "
              "`obligor` is the MAR22.19 legal-entity netting key (canonical via "
              "the config Issuer_Alias_Map); `Issuer` remains the per-leg display "
              "label. They differ only on aliased share-class / name-variant rows "
              "and have distinct consumers, so this is a genuine semantic column, "
              "not a prohibited duplicate.",
    )


def main() -> None:
    df, cfg = build_nonsec_loader_frame()
    issues = validate(df)
    print(summarise(df))
    print()
    if issues:
        print("VALIDATION ISSUES:")
        for i in issues:
            print(f"  - {i}")
    else:
        print("Validation: clean.")
    write_snapshot(df)
    print(f"\nPhase 1 snapshot written: {os.path.normpath(OUTPUT_SNAPSHOT)}")


if __name__ == "__main__":
    main()
