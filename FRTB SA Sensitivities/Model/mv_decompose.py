"""
MV decomposition helpers — Sheet 2 emit (Portfolio_MV_Decomposed).

Produces leg-level rows with model-priced MVs from prices already computed
during sensitivity calculation. See `CLAUDE.md → Output Contract` for the
full schema, synthetic-ID conventions, pricing-model vocabulary, and the
per-parent conservation invariant.

Three emit paths:
  * single_leg_row(...)             undecomposed positions (1 row)
  * decompose_callable_legs(...)    callable bond -> vanilla + short_call
  * decompose_spx_legs(...)         SPX call option -> N constituent rows

The 14 schema columns are fixed and ordered; emit dicts use these keys.
"""
from __future__ import annotations
import os
import pandas as pd
import QuantLib as ql

CALL_LEG_ID_OFFSET = 100_000
CONSTITUENT_LEG_ID_MULTIPLIER = 10_000

NONSEC_CONFIG_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "Input data", "FRTB_NonSec_Config.xlsx",
))
SPX_CONSTITUENTS_SHEET = "SPX_Constituents"

SCHEMA_COLUMNS = [
    "ID",
    "parent_position_id",
    "decomposition_leg",
    "Issuer",
    "Security",
    "Position Type",
    "Issue Type",
    "Currency",
    "Long/Short",
    "Rating",
    "Seniority (DRC)",
    "Quantity/Notional",
    "MV_USD",
    "pricing_model",
]

DECOMPOSITION_LEG_RANK = {
    "": 0,
    "vanilla": 1,
    "short_call": 2,
    "index_constituent": 3,
}


def load_spx_constituents(
    path: str = NONSEC_CONFIG_PATH, sheet: str = SPX_CONSTITUENTS_SHEET,
) -> pd.DataFrame:
    """Read SPX constituents from FRTB_NonSec_Config.xlsx.

    Header at row 7 (preserves the source-file disclaimer block above).
    Returns a DataFrame with at least ['Ticker', 'Issuer', 'Index_Weight'].
    Weights are normalised to sum to 1.0 if they don't already.
    """
    df = pd.read_excel(path, sheet_name=sheet, header=6)
    df = df.dropna(subset=["Ticker", "Issuer", "Index_Weight"]).copy()
    df["Index_Weight"] = pd.to_numeric(df["Index_Weight"], errors="coerce")
    df = df.dropna(subset=["Index_Weight"]).reset_index(drop=True)
    total = df["Index_Weight"].sum()
    if abs(total - 1.0) > 1e-3:
        print(f"[mv_decompose] WARNING: SPX_Constituents weights sum to "
              f"{total:.6f}, normalising on the fly.")
        df["Index_Weight"] = df["Index_Weight"] / total
    return df


def _portfolio_field(row: pd.Series, key: str, default=None):
    """NaN-safe read of a portfolio row field."""
    v = row.get(key, default)
    if v is None:
        return default
    try:
        if pd.isna(v):
            return default
    except (TypeError, ValueError):
        pass
    return v


def _base_row_from_portfolio(port_row: pd.Series, inst: dict) -> dict:
    """Common scaffold copied from the original portfolio row.

    Used by every leg as a starting point so undecomposed and decomposed
    rows carry the same identifying metadata.
    """
    return {
        "ID": inst["id"],
        "parent_position_id": inst["id"],
        "decomposition_leg": "",
        "Issuer": str(_portfolio_field(port_row, "Issuer", "") or ""),
        "Security": str(inst.get("security", "") or ""),
        "Position Type": str(inst.get("position_type", "") or ""),
        "Issue Type": str(inst.get("issue_type", "") or ""),
        "Currency": str(inst.get("currency", "") or ""),
        "Long/Short": str(inst.get("long_short", "") or ""),
        "Rating": str(inst.get("rating", "") or ""),
        "Seniority (DRC)": str(_portfolio_field(port_row, "Seniority (DRC)", "") or ""),
        "Quantity/Notional": float(inst.get("notional", 0) or 0),
        "MV_USD": 0.0,
        "pricing_model": "",
    }


def single_leg_row(port_row: pd.Series, inst: dict, mv_signed: float,
                   pricing_model: str) -> dict:
    """Emit one row for an undecomposed position."""
    row = _base_row_from_portfolio(port_row, inst)
    row["MV_USD"] = float(mv_signed)
    row["pricing_model"] = pricing_model
    return row


# ── CALLABLE BOND DECOMPOSITION ───────────────────────────────────────────────

def _build_ql_vanilla_for_callable(inst: dict, mkt: dict) -> tuple:
    """Build a vanilla FixedRateBond mirroring the callable's terms.

    Same notional / coupon / maturity / frequency / issue_date as the callable;
    same curve handle (SOFR-only). The single behavioural difference vs the
    callable is the absence of a callability schedule. Returns (npv_unsigned,
    scale, signed_npv).

    Discount handle matches calc_callable_bond_girr_delta to keep the
    conservation invariant exact:
        MV_callable_held = MV_vanilla_long + MV_short_call_leg
    """
    from sensitivity_calc import _build_ql_bond, _price_bond_ql
    coupon = inst.get("coupon", 0) / 100
    freq = inst.get("freq", 2)
    ccy = inst.get("currency", "USD")
    base_handle = mkt["usd_handle"] if ccy == "USD" else mkt["gbp_handle"]
    bond, scale = _build_ql_bond(
        inst["notional"], coupon, inst["maturity"], freq, inst.get("issue_date"))
    npv = _price_bond_ql(bond, base_handle, 0)
    return npv, scale, npv * scale


def decompose_callable_legs(port_row: pd.Series, inst: dict,
                            mkt: dict) -> tuple[list[dict], float]:
    """Decompose a held callable bond into vanilla + short_call legs.

    Requires inst['_computed_mv'] to be populated (set by
    calc_callable_bond_girr_delta — that's PV_callable × scale, signed by
    notional sign).

    Returns (rows, parent_total_check) where parent_total_check is the
    callable's signed MV. Σ row['MV_USD'] across the two legs MUST equal
    parent_total_check within the conservation tolerance.

    Sign convention:
      vanilla leg     -> Long  (held)  -> MV_USD = +|PV_vanilla × scale|
      short_call leg  -> Short (sold)  -> MV_USD = -(PV_vanilla - PV_callable) × |scale|

    Net: MV_vanilla + MV_short_call = +PV_vanilla + PV_callable - PV_vanilla = PV_callable
    """
    pv_callable_signed = float(inst.get("_computed_mv", 0.0))
    _, _, pv_vanilla_signed = _build_ql_vanilla_for_callable(inst, mkt)

    notional_sign = 1.0 if float(inst.get("notional", 0)) >= 0 else -1.0
    pv_vanilla_abs = abs(pv_vanilla_signed)
    pv_callable_abs = abs(pv_callable_signed)
    call_option_value = pv_vanilla_abs - pv_callable_abs

    base = _base_row_from_portfolio(port_row, inst)
    face = abs(float(inst.get("notional", 0)))

    vanilla = dict(base)
    vanilla["ID"] = inst["id"]
    vanilla["decomposition_leg"] = "vanilla"
    vanilla["Long/Short"] = "Long"
    vanilla["Quantity/Notional"] = face * notional_sign
    vanilla["MV_USD"] = pv_vanilla_signed
    vanilla["Security"] = f"{base['Security']} [vanilla leg]"
    vanilla["pricing_model"] = "QL_FixedRateBond_discount_curve"

    short_call = dict(base)
    short_call["ID"] = inst["id"] + CALL_LEG_ID_OFFSET
    short_call["decomposition_leg"] = "short_call"
    short_call["Long/Short"] = "Short"
    short_call["Quantity/Notional"] = 0.0
    short_call["MV_USD"] = -call_option_value * notional_sign
    short_call["Security"] = f"{base['Security']} [short call leg]"
    short_call["pricing_model"] = "HW1F_trinomial_tree"

    return [vanilla, short_call], pv_callable_signed


# ── SPX INDEX OPTION DECOMPOSITION (MAR22.5 LOOK-THROUGH) ─────────────────────

_SPX_CONSTITUENTS_CACHE: pd.DataFrame | None = None


def _get_spx_constituents() -> pd.DataFrame:
    global _SPX_CONSTITUENTS_CACHE
    if _SPX_CONSTITUENTS_CACHE is None:
        _SPX_CONSTITUENTS_CACHE = load_spx_constituents()
    return _SPX_CONSTITUENTS_CACHE


def decompose_spx_legs(port_row: pd.Series, inst: dict) -> tuple[list[dict], float]:
    """Decompose an SPX call option into per-constituent legs (MAR22.5).

    Requires inst['_computed_mv'] to be the priced option's signed MV
    (= _bsm_call(...) × notional × multiplier, with notional carrying
    Long/Short sign).

    Returns (rows, parent_total_check) where parent_total_check equals the
    parent option's signed MV. Σ row['MV_USD'] = parent_total_check × Σ
    weights ≈ parent_total_check.

    Each constituent leg:
      ID                = orig_id × CONSTITUENT_LEG_ID_MULTIPLIER + (k+1)
      Issuer            = constituent's issuer
      Position Type     = 'Equity'  (constituent treated as spot equity for DRC)
      Issue Type        = 'Corp'
      Quantity/Notional = 0   (call option, MAR22.14(1)(c) extension)
      MV_USD            = parent_signed_MV × Index_Weight
      Long/Short        = parent's Long/Short label
      Seniority (DRC)   = 'Non Senior'  (equity is rank 1)
    """
    parent_signed_mv = float(inst.get("_computed_mv", 0.0))
    constituents = _get_spx_constituents()

    base = _base_row_from_portfolio(port_row, inst)
    parent_label = str(inst.get("long_short", "Short")).strip() or "Short"

    rows = []
    for k, c in constituents.iterrows():
        weight = float(c["Index_Weight"])
        if weight <= 0:
            continue
        leg = dict(base)
        leg["ID"] = inst["id"] * CONSTITUENT_LEG_ID_MULTIPLIER + (k + 1)
        leg["decomposition_leg"] = "index_constituent"
        leg["Issuer"] = str(c["Issuer"])
        leg["Security"] = f"{base['Security']} -> {c['Issuer']}"
        leg["Position Type"] = "Equity"
        leg["Issue Type"] = "Corp"
        leg["Long/Short"] = parent_label
        leg["Rating"] = str(c.get("Rating", "")) if pd.notna(c.get("Rating", None)) else ""
        leg["Seniority (DRC)"] = "Non Senior"
        leg["Quantity/Notional"] = 0.0
        leg["MV_USD"] = parent_signed_mv * weight
        leg["pricing_model"] = "BS_with_SPX_vol_surface"
        rows.append(leg)
    return rows, parent_signed_mv


# ── ROW ORDERING ──────────────────────────────────────────────────────────────

def sort_rows(rows: list[dict]) -> list[dict]:
    """Deterministic ordering: (parent_position_id, decomposition_leg_rank, ID)."""
    def k(r):
        leg = r.get("decomposition_leg", "") or ""
        return (int(r["parent_position_id"]),
                DECOMPOSITION_LEG_RANK.get(leg, 99),
                int(r["ID"]))
    return sorted(rows, key=k)
