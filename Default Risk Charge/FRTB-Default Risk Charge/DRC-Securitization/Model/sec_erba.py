"""
sec_erba.py — SEC-ERBA risk weight (CRE42).

Implements:
  - Long-term Table 2 / Table 4 (STC) lookup by (rating, seniority)         CRE42.4 / 42.13
  - Short-term Table 1 / Table 3 (STC) lookup by rating                      CRE42.2 / 42.12
  - Linear maturity interpolation between 1y and 5y columns                  CRE42.5(1)
  - Non-senior thickness adjustment max(1 - (D - A), 0.5)                    CRE42.5(2)
  - Secondary floor: must not be lower than senior of same deal,
    same rating + maturity                                                   CRE42.7

The 15% standard floor (CRE42.7) is applied DOWNSTREAM by sec_caps as part
of the floors-and-caps wrap, NOT here. This module returns the pre-floor RW.

For DRC (MAR22.34(1)) M_T is pinned to 1, so the interpolation collapses to
the 1y column. The full interpolation is implemented anyway because the
banking-book parallel branch (not yet wired) needs it, and because the
implementation cost is trivial.

Rating normalisation handles common ECAI naming variants (S&P / Moody's /
Fitch). The CCC+/CCC/CCC- bucket per CRE42.4 collapses three notches into
one row; "Below CCC-" catches everything weaker including C and D.
"""
from __future__ import annotations
import re
import pandas as pd

# ── RATING NORMALISATION ──────────────────────────────────────────────────────

_LONG_TERM_TABLE_RATINGS = [
    "AAA", "AA+", "AA", "AA-", "A+", "A", "A-",
    "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-",
    "B+", "B", "B-", "CCC+/CCC/CCC-", "Below CCC-",
]
_CCC_BUCKET = "CCC+/CCC/CCC-"
_BELOW_CCC = "Below CCC-"

# Each short-term bucket maps to the regex patterns that resolve to it
# (S&P A-1/A-2/A-3, Moody's P-1/P-2/P-3, Fitch F1/F2/F3, with optional dash
# and Fitch's '+' suffix on F1).
_SHORT_TERM_BUCKETS = {
    "A-1/P-1": [r"A-?1\+?", r"P-?1", r"F1\+?"],
    "A-2/P-2": [r"A-?2", r"P-?2", r"F2"],
    "A-3/P-3": [r"A-?3", r"P-?3", r"F3"],
}
_SHORT_TERM_OTHER = "All other ratings"


def _normalise(rating) -> str:
    if rating is None:
        return ""
    if isinstance(rating, float) and pd.isna(rating):
        return ""
    return str(rating).strip().upper().replace(" ", "")


def is_short_term_rating(rating) -> bool:
    norm = _normalise(rating)
    if not norm:
        return False
    for patterns in _SHORT_TERM_BUCKETS.values():
        for pat in patterns:
            if re.fullmatch(pat, norm):
                return True
    return False


def canonicalise_short_term(rating) -> str:
    """Map an ECAI short-term rating to its Table 1/Table 3 row label."""
    norm = _normalise(rating)
    for bucket, patterns in _SHORT_TERM_BUCKETS.items():
        for pat in patterns:
            if re.fullmatch(pat, norm):
                return bucket
    return _SHORT_TERM_OTHER


def canonicalise_long_term(rating) -> str:
    """Map an ECAI long-term rating to its Table 2/Table 4 row label."""
    norm = _normalise(rating)
    if norm in {"CCC+", "CCC", "CCC-"}:
        return _CCC_BUCKET
    if norm in _LONG_TERM_TABLE_RATINGS:
        return norm
    # Below-CCC catch-all: CC, C, D, NR (not rated should not get here)
    if norm in {"CC", "C", "D"} or norm.startswith("D"):
        return _BELOW_CCC
    return _BELOW_CCC


# ── TABLE LOOKUP ──────────────────────────────────────────────────────────────

def lookup_long_term(
    table: pd.DataFrame, rating: str, is_senior: bool
) -> tuple[float, float]:
    """Return (RW_1y, RW_5y) as decimals. Raises if rating not in table."""
    canon = canonicalise_long_term(rating)
    row = table[table["Rating"] == canon]
    if row.empty:
        raise KeyError(f"Long-term rating canonical {canon!r} not in table")
    sen_or_non_sen = "Senior" if is_senior else "NonSenior"
    return (
        float(row.iloc[0][f"{sen_or_non_sen}_RW_pct_1y"]) / 100.0,
        float(row.iloc[0][f"{sen_or_non_sen}_RW_pct_5y"]) / 100.0,
    )


def lookup_short_term(table: pd.DataFrame, rating: str) -> float:
    """Return short-term RW as a decimal."""
    canon = canonicalise_short_term(rating)
    row = table[table["External_credit_assessment"] == canon]
    if row.empty:
        raise KeyError(f"Short-term rating canonical {canon!r} not in table")
    return float(row.iloc[0]["Risk_weight_pct"]) / 100.0


def _select_long_term_table(config: dict, is_stc: bool) -> pd.DataFrame:
    return config["SEC_ERBA_LongTerm_Table4_STC" if is_stc else "SEC_ERBA_LongTerm_Table2"]


def _select_short_term_table(config: dict, is_stc: bool) -> pd.DataFrame:
    return config["SEC_ERBA_ShortTerm_Table3_STC" if is_stc else "SEC_ERBA_ShortTerm_Table1"]


# ── CORE COMPUTATIONS ─────────────────────────────────────────────────────────

def _load_erba_constants(config: dict) -> dict:
    """Pull CRE42 scalars from SEC_Constants."""
    sc = config["SEC_Constants"].set_index("Constant")
    return {
        "mt_floor":      float(sc.at["mt_floor_years", "Value"]),
        "mt_cap":        float(sc.at["mt_cap_years",   "Value"]),
        "thickness_min": float(sc.at["erba_thickness_adjustment_min", "Value"]),
    }


def linear_interpolate_maturity(
    rw_1y: float, rw_5y: float, M_T: float, mt_floor: float, mt_cap: float
) -> float:
    """CRE42.5(1): linear interpolation between the 1y and 5y columns.

    M_T should already be clipped to [mt_floor, mt_cap] by the loader;
    defence-in-depth here. mt_floor/mt_cap come from SEC_Constants
    (mt_floor_years / mt_cap_years), per CRE40.22.
    """
    M_T = max(mt_floor, min(mt_cap, M_T))
    span = mt_cap - mt_floor
    fraction = (M_T - mt_floor) / span
    return rw_1y + fraction * (rw_5y - rw_1y)


def thickness_adjustment(A: float, D: float, thickness_min: float) -> float:
    """CRE42.5(2): non-senior tranche thickness adjustment.

    RW_non_senior = RW_table * max(1 - (D - A), thickness_min)
    A and D are decimals. `thickness_min` is sourced from
    SEC_Constants.erba_thickness_adjustment_min (0.5 per CRE42.5(2)).
    """
    return max(1.0 - (D - A), thickness_min)


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def compute_rw(row: pd.Series, config: dict) -> tuple[float, dict]:
    """SEC-ERBA risk weight per CRE42.

    Inputs read from row:
      Rating (or inferred_rating)
      is_senior_derived
      is_stc_compliant
      M_T_for_drc        (always 1 in DRC pipeline per MAR22.34(1))
      Attachment Pt (%) / Detachment Pt (%)  (for thickness adjustment)

    Returns: (rw_decimal, details_dict)
    rw_decimal is the pre-floor RW as a decimal (e.g. 0.15 for 15%);
    sec_caps applies the 15% standard floor downstream.
    """
    # Resolve effective rating (inferred fallback per CRE42.9-10)
    rating = row.get("Rating")
    if rating is None or (isinstance(rating, float) and pd.isna(rating)):
        rating = row.get("inferred_rating")
    if rating is None or (isinstance(rating, float) and pd.isna(rating)):
        return float("nan"), {
            "approach": "SEC-ERBA",
            "error": "CRE42.8 op-req fail: no rating and no inferred_rating",
        }

    is_senior = bool(row.get("is_senior_derived", False))
    is_stc = bool(row.get("is_stc_compliant", False))
    consts = _load_erba_constants(config)
    M_T = float(row.get("M_T_for_drc", consts["mt_floor"]))
    A = float(row.get("Attachment Pt (%)", 0.0)) / 100.0
    D = float(row.get("Detachment Pt (%)", 100.0)) / 100.0

    # Short-term path (CRE42.2 / 42.12)
    if is_short_term_rating(rating):
        st_table = _select_short_term_table(config, is_stc)
        rw = lookup_short_term(st_table, rating)
        return rw, {
            "approach": "SEC-ERBA",
            "rating_class": "short_term",
            "rating": rating,
            "rating_canonical": canonicalise_short_term(rating),
            "is_stc": is_stc,
            "rw_pre_standard_floor": rw,
            "table_used": "Table 3 (STC)" if is_stc else "Table 1",
        }

    # Long-term path (CRE42.4 / 42.13)
    lt_table = _select_long_term_table(config, is_stc)
    rw_1y, rw_5y = lookup_long_term(lt_table, rating, is_senior)
    rw_interp = linear_interpolate_maturity(
        rw_1y, rw_5y, M_T, consts["mt_floor"], consts["mt_cap"]
    )

    if is_senior:
        return rw_interp, {
            "approach": "SEC-ERBA",
            "rating_class": "long_term",
            "rating": rating,
            "rating_canonical": canonicalise_long_term(rating),
            "is_senior": True,
            "is_stc": is_stc,
            "M_T": M_T,
            "rw_1y_table": rw_1y,
            "rw_5y_table": rw_5y,
            "rw_interp": rw_interp,
            "rw_pre_standard_floor": rw_interp,
            "table_used": "Table 4 (STC)" if is_stc else "Table 2",
        }

    # Non-senior path: thickness adjustment + secondary floor
    adj = thickness_adjustment(A, D, consts["thickness_min"])
    rw_after_thickness = rw_interp * adj

    # Secondary floor (CRE42.7): not lower than senior tranche of same securitisation,
    # same rating, same maturity. We compute the senior RW at the same M_T and take max.
    sen_1y, sen_5y = lookup_long_term(lt_table, rating, is_senior=True)
    rw_senior_at_M_T = linear_interpolate_maturity(
        sen_1y, sen_5y, M_T, consts["mt_floor"], consts["mt_cap"]
    )
    secondary_floor_binds = rw_after_thickness < rw_senior_at_M_T
    rw_final = max(rw_after_thickness, rw_senior_at_M_T)

    return rw_final, {
        "approach": "SEC-ERBA",
        "rating_class": "long_term",
        "rating": rating,
        "rating_canonical": canonicalise_long_term(rating),
        "is_senior": False,
        "is_stc": is_stc,
        "M_T": M_T,
        "rw_1y_table": rw_1y,
        "rw_5y_table": rw_5y,
        "rw_interp": rw_interp,
        "thickness": D - A,
        "thickness_adj_factor": adj,
        "rw_after_thickness": rw_after_thickness,
        "rw_senior_at_M_T_floor": rw_senior_at_M_T,
        "secondary_floor_binds": secondary_floor_binds,
        "rw_pre_standard_floor": rw_final,
        "table_used": "Table 4 (STC)" if is_stc else "Table 2",
    }
