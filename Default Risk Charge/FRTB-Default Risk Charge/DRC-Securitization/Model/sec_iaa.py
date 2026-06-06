"""
sec_iaa.py - SEC-IAA risk weight (CRE43).

The Internal Assessment Approach (IAA) is the unrated-ABCP-facility path.
The bank's internal credit grade becomes the rating input to SEC-ERBA -
once the grade is mapped to an ECAI-equivalent rating, the RW computation
IS SEC-ERBA. So sec_iaa is a thin adapter on top of sec_erba.

---------------------------------------------------------------------------
WHY IAA EXISTS: the difference between ABCP (CP) and ABCP Liquidity Facility
---------------------------------------------------------------------------
An Asset-Backed Commercial Paper (ABCP) conduit funds itself by issuing
short-term commercial paper to investors, backed by a pool of receivables.
The CP itself is externally rated (typically A-1/P-1) - investors buy it,
the conduit uses the proceeds to buy receivables, and rolls the CP at
maturity. Pricing route: SEC-ERBA short-term table (CRE42.2 Table 1).

A sponsor bank also extends a *liquidity facility* TO THE CONDUIT - a
revolving credit line drawn when (a) the CP cannot be rolled in a market
disruption, or (b) receivable cashflows fall short of redemption needs.
This facility is NOT externally rated - ECAIs do not rate bilateral
commitments. Without IAA, the facility would fall to the 1250% residual.

CRE43 was designed specifically so sponsor banks could use their own
internal models (with supervisory approval per CRE43.2's 14 op-reqs) to
assess the facility's credit grade, then map that grade to an ECAI
equivalent and route through SEC-ERBA. Result: the facility gets a
sensible RW reflecting the bank's superior information about its own
conduit's receivables and structural supports.

---------------------------------------------------------------------------
WHAT THE BANK GETS IN RETURN FOR PROVIDING A FACILITY
---------------------------------------------------------------------------
A facility position is contractually a *commitment to lend on demand*.
That looks superficially like a liability, but in regulatory and
economic substance it is an asset for the bank that provides it:

  Commercial side (income):
    - Commitment fee on undrawn balance (typically 15-50 bps p.a.)
    - Drawn-balance interest = SOFR + spread, when the facility is drawn
    - Sponsor / programme economics: the bank also earns servicing fees
      on the underlying receivables, captures hedging / FX / cash-management
      flow from the conduit's corporate sellers, and the relationship
      anchors broader transaction-banking revenue with those clients

  Accounting side (balance sheet):
    - Undrawn: off-balance-sheet commitment, disclosed as contractual
      obligation; income recognised as fee income
    - Drawn: on-balance-sheet loan to the SPV at the agreed rate

  Regulatory side (capital):
    - CRE40.19: for sec exposures, EAD = full commitment, no CCF.
      A $500m undrawn liquidity facility consumes the same capital as
      if it were fully drawn. That capital cost is the regulatory price
      of providing the facility - and the commitment fee is what
      compensates the bank for tying up that capital.

So while the FACILITY is the bank's commitment (a future obligation), the
RIGHT TO INTEREST + FEES + the eventual loan-on-draw is an asset position
from the bank's perspective. The conduit (SPV) holds the symmetric view:
a contingent liability to repay when the facility is drawn.

In this engine the facility position is modelled as a funded CSR-Sec
exposure with notional = commitment size, so the upstream sensitivity
engine and SA capital aggregation treat it consistently. The DRC sec
engine routes it to SEC-IAA via the (asset_class=ABCP, Rating=blank,
is_abcp_facility=True) signal pattern.
"""
from __future__ import annotations
import pandas as pd

from . import sec_erba


# ── CONFIG ACCESS ─────────────────────────────────────────────────────────────

def _load_iaa_mapping(config: dict) -> pd.DataFrame:
    """Return the SEC_IAA_Mapping sheet (internal grade -> ECAI rating)."""
    if "SEC_IAA_Mapping" not in config:
        raise KeyError(
            "sec_iaa requires config['SEC_IAA_Mapping']; rebuild "
            "FRTB_Sec_Config.xlsx via build_sec_config.py"
        )
    return config["SEC_IAA_Mapping"]


def map_internal_grade_to_ecai(mapping: pd.DataFrame, internal_grade: str) -> str:
    """Look up the ECAI-equivalent rating for an internal credit grade.

    CRE43.2(7): the bank's internal grades must be mapped to ECAI categories
    in advance, under supervisory approval. The mapping table is bank-specific
    and lives in SEC_IAA_Mapping. Returns the mapped ECAI rating string
    (e.g. 'AAA', 'BBB+', 'A-1/P-1'); raises if the grade is unknown.
    """
    idx = mapping.set_index("Internal_grade")
    if internal_grade not in idx.index:
        raise ValueError(
            f"sec_iaa.map_internal_grade_to_ecai: unknown internal grade "
            f"{internal_grade!r}; add a row to SEC_IAA_Mapping"
        )
    return str(idx.at[internal_grade, "ECAI_equivalent_rating"])


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def compute_rw(row: pd.Series, config: dict) -> tuple[float, dict]:
    """SEC-IAA risk weight per CRE43.

    Reads `internal_credit_grade` from the row, maps to ECAI via
    SEC_IAA_Mapping, then delegates to sec_erba.compute_rw with the
    Rating field overridden by the mapped ECAI rating.

    Required row inputs (in addition to those needed by sec_erba):
      internal_credit_grade
      is_abcp_facility (must be True; engine routing enforces this)
    """
    if not bool(row.get("is_abcp_facility", False)):
        return float("nan"), {
            "approach": "SEC-IAA",
            "error": "CRE43.1 op-req fail: is_abcp_facility=False "
                     "(IAA is for ABCP liquidity facilities / credit "
                     "enhancements, not for CP positions)",
        }

    internal_grade = row.get("internal_credit_grade")
    if internal_grade is None or (isinstance(internal_grade, float) and pd.isna(internal_grade)):
        return float("nan"), {
            "approach": "SEC-IAA",
            "error": "CRE43.2 op-req fail: no internal_credit_grade on the position",
        }
    internal_grade = str(internal_grade)

    mapping = _load_iaa_mapping(config)
    ecai_rating = map_internal_grade_to_ecai(mapping, internal_grade)

    # Delegate to sec_erba with Rating overridden by the mapped ECAI rating.
    erba_row = row.copy()
    erba_row["Rating"] = ecai_rating
    erba_rw, erba_det = sec_erba.compute_rw(erba_row, config)

    return erba_rw, {
        "approach": "SEC-IAA",
        "internal_grade": internal_grade,
        "mapped_ecai_rating": ecai_rating,
        "is_abcp_facility": True,
        "delegated_to": "SEC-ERBA",
        "erba_details": erba_det,
        "rw_pre_standard_floor": erba_rw,
    }
