"""
sec_irba.py - SEC-IRBA risk weight (CRE44).

Implements the full CRE44 pipeline for the SEC-IRBA path:

  - K_IRB (CRE44.2-13): per-pool input from Deal_Master.K_IRB. Pool-level
    capital charge under the IRB framework, expressed as a decimal of pool
    exposure. Treated as exogenous here - DRC consumes it; banking-book IRB
    is what produces it.

  - A, D (CRE44.14-15): tranche attachment/detachment as decimals.

  - Supervisory parameter p (CRE44.17 / 44.28):
        p = max(p_floor, A_p + B_p/N + C_p*K_IRB + D_p*LGD + E_p*M_T)
    where (A_p, B_p, C_p, D_p, E_p) are looked up by
        (asset_class, seniority, granularity)
    in SEC_IRBA_p_Table1 (non-STC) or SEC_IRBA_p_Table2_STC (STC). The two
    tables carry identical coefficients; STC only changes the downstream
    floor (10% senior / 15% non-senior in sec_caps).

  - SSFA core (CRE44.23): reuses sec_sa.compute_K_SSFA with K=K_IRB.
        a = -1 / (p * K_IRB)
        u = D - K_IRB
        l = max(A - K_IRB, 0)
        K_SSFA = (e^(au) - e^(al)) / (a * (u - l))

  - Risk-weight dispatch (CRE44.24):
        D <= K_IRB         -> RW = 1250%
        A >= K_IRB         -> RW = K_SSFA / rw_factor (i.e. * 12.5)
        A <  K_IRB < D     -> blended weighted average of 1250% and
                              K_SSFA/rw_factor

All regulatory scalars (p_floor=0.30, the 1250% cap, the 0.08 capital
factor, the N>=25 granularity threshold) are sourced from
FRTB_Sec_Config.xlsx -> SEC_Constants. Floors (15% standard, 10%/15% STC,
100% NPL) are applied downstream by sec_caps. This module returns the
pre-floor RW.

Mapping from pool to (asset_class, granularity):
  - Wholesale asset classes: CMBS, CLO, ABCP (any wholesale receivables /
    leveraged-loan / commercial-real-estate pool)
  - Retail asset classes: RMBS, ABS-CreditCard, ABS-Auto (any consumer pool)
  - Granularity is meaningful only for wholesale rows (CRE44.17 Table 1):
        Granular = N >= granularity_threshold_N (25)
        Non-granular = N < threshold
    Retail rows in the p-table carry Granularity="n/a".
"""
from __future__ import annotations
import pandas as pd

from .sec_sa import compute_K_SSFA


# ── CONFIG ACCESS ─────────────────────────────────────────────────────────────

def _load_irba_constants(config: dict) -> dict:
    """Pull CRE44 scalars from SEC_Constants."""
    sc = config["SEC_Constants"].set_index("Constant")
    return {
        "p_floor":              float(sc.at["p_floor_irba",                "Value"]),
        "rw_max":               float(sc.at["rw_d_lte_KA",                 "Value"]),
        "rw_factor_inv":        1.0 / float(sc.at["rw_factor_to_capital",  "Value"]),
        "granularity_threshold": float(sc.at["granularity_threshold_N",    "Value"]),
    }


def _select_p_table(config: dict, is_stc: bool) -> pd.DataFrame:
    return config["SEC_IRBA_p_Table2_STC" if is_stc else "SEC_IRBA_p_Table1"]


# ── BUILDING BLOCKS ───────────────────────────────────────────────────────────

# Pool-type to (asset_class) mapping per CRE44.17 Table 1 row taxonomy.
# Wholesale = corporates / CRE / SME / receivables; Retail = consumers.
_RETAIL_POOL_TYPES = {
    "RMBS_prime", "RMBS_subprime",
    "ABS_CreditCard", "ABS_Auto",
}
_WHOLESALE_POOL_TYPES = {
    "CMBS_conduit", "CMBS_SASB",
    "CLO_BSL",
    "ABCP_receivables",
}


def asset_class_for(pool_type: str) -> str:
    """Map pool_type string to the p-table Asset_class column.

    Returns 'Retail' or 'Wholesale'. Raises ValueError on unknown pool_type
    so the engine fails loudly rather than silently using a wrong row.
    """
    if pool_type in _RETAIL_POOL_TYPES:
        return "Retail"
    if pool_type in _WHOLESALE_POOL_TYPES:
        return "Wholesale"
    raise ValueError(
        f"sec_irba.asset_class_for: unknown pool_type {pool_type!r}; extend "
        f"_RETAIL_POOL_TYPES or _WHOLESALE_POOL_TYPES in sec_irba.py"
    )


def granularity_for(asset_class: str, N: float, threshold: float) -> str:
    """Map (asset_class, N) to the p-table Granularity column value."""
    if asset_class == "Retail":
        return "n/a"
    return "Granular (N>=25)" if N >= threshold else "Non-granular (N<25)"


def lookup_p_coefficients(
    p_table: pd.DataFrame, asset_class: str, seniority: str, granularity: str
) -> dict:
    """Look up (A_p, B_p, C_p, D_p, E_p) for one row of the CRE44.17 table.

    Returns a dict keyed by coefficient name. Raises if no row matches.
    NB: pandas read_excel may interpret the literal "n/a" cell as NaN, so
    the Granularity column is normalised to its string form before matching.
    """
    gran_col = p_table["Granularity"].fillna("n/a").astype(str)
    mask = (
        (p_table["Asset_class"] == asset_class)
        & (p_table["Seniority"] == seniority)
        & (gran_col == granularity)
    )
    rows = p_table[mask]
    if rows.empty:
        raise ValueError(
            f"sec_irba: no p-table row for "
            f"(Asset_class={asset_class!r}, Seniority={seniority!r}, "
            f"Granularity={granularity!r})"
        )
    r = rows.iloc[0]
    return {
        "A_p": float(r["A_p"]),
        "B_p": float(r["B_p"]),
        "C_p": float(r["C_p"]),
        "D_p": float(r["D_p"]),
        "E_p": float(r["E_p"]),
    }


def compute_p(
    coeffs: dict, N: float, K_IRB: float, LGD: float, M_T: float, p_floor: float
) -> float:
    """CRE44.17: p = max(p_floor, A_p + B_p/N + C_p*K_IRB + D_p*LGD + E_p*M_T)."""
    p_raw = (
        coeffs["A_p"]
        + coeffs["B_p"] / N
        + coeffs["C_p"] * K_IRB
        + coeffs["D_p"] * LGD
        + coeffs["E_p"] * M_T
    )
    return max(p_floor, p_raw)


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def compute_rw(row: pd.Series, config: dict) -> tuple[float, dict]:
    """SEC-IRBA risk weight per CRE44.

    Inputs read from row:
      K_IRB                   (decimal, pool-level IRB capital charge)
      N                       (effective number of exposures in pool)
      LGD                     (decimal, exposure-weighted average LGD)
      M_T_for_drc             MAR22.34(1) DRC overlay (= 1.0). The maturity
                              component in the banking-book sec framework
                              is set to zero (1y maturity assumed) to avoid
                              double-counting migration risk with CSR-Sec
                              capital. The actual cashflow-weighted
                              banking-book M_T (CRE40.22 / 40.23) is NOT
                              used inside CRE44.17 - it is used elsewhere
                              for MAR22.30 / MAR22.15-18 maturity scaling
                              of GROSS JTD (in sec_loader/sec_engine, not
                              here).
      Attachment Pt (%) / Detachment Pt (%)
      pool_type               (used for asset-class mapping)
      is_senior_derived
      is_stc_compliant

    Returns: (rw_decimal_pre_floor, details_dict)
    """
    # Required inputs
    K_IRB = row.get("K_IRB")
    if K_IRB is None or (isinstance(K_IRB, float) and pd.isna(K_IRB)):
        return float("nan"), {
            "approach": "SEC-IRBA",
            "error": "K_IRB missing - cannot compute SSFA",
        }
    K_IRB = float(K_IRB)

    N = row.get("N")
    if N is None or (isinstance(N, float) and pd.isna(N)):
        return float("nan"), {
            "approach": "SEC-IRBA",
            "error": "N (effective number of exposures) missing",
        }
    N = float(N)

    LGD = float(row.get("LGD", 0.0) or 0.0)
    # MAR22.34(1) overlay: the maturity component in the banking-book sec
    # framework is set to zero (i.e. 1y maturity assumed) for the RW calc,
    # to avoid double-counting migration risk with the trading-book CSR-Sec
    # capital requirement. The CRE44.17 E_p * M_T term therefore feeds 1.0,
    # not the actual cashflow-weighted maturity. The actual M_T flows
    # separately through the MAR22.30 / 22.15-18 GROSS JTD scaling at the
    # JTD/netting layer (sec_loader -> sec_engine step 9), not here.
    M_T = float(row.get("M_T_for_drc", 1.0))
    A = float(row.get("Attachment Pt (%)", 0.0)) / 100.0
    D = float(row.get("Detachment Pt (%)", 100.0)) / 100.0
    is_senior = bool(row.get("is_senior_derived", False))
    is_stc = bool(row.get("is_stc_compliant", False))
    pool_type = str(row.get("pool_type", ""))

    consts = _load_irba_constants(config)
    p_table = _select_p_table(config, is_stc)

    asset_class = asset_class_for(pool_type)
    seniority = "Senior" if is_senior else "Non-senior"
    granularity = granularity_for(asset_class, N, consts["granularity_threshold"])
    coeffs = lookup_p_coefficients(p_table, asset_class, seniority, granularity)
    p = compute_p(coeffs, N, K_IRB, LGD, M_T, consts["p_floor"])

    base_details = {
        "approach": "SEC-IRBA",
        "K_IRB": K_IRB,
        "N": N,
        "LGD": LGD,
        "M_T": M_T,
        "A": A,
        "D": D,
        "asset_class": asset_class,
        "seniority": seniority,
        "granularity": granularity,
        "p_coefficients": coeffs,
        "p_pre_floor": (
            coeffs["A_p"] + coeffs["B_p"] / N + coeffs["C_p"] * K_IRB
            + coeffs["D_p"] * LGD + coeffs["E_p"] * M_T
        ),
        "p_floor": consts["p_floor"],
        "p": p,
        "is_senior_derived": is_senior,
        "is_stc": is_stc,
        "p_table_used": "Table 2 (STC; same coeffs)" if is_stc else "Table 1",
    }

    rw_max = consts["rw_max"]
    rw_factor_inv = consts["rw_factor_inv"]

    # Case (1): D <= K_IRB - tranche entirely in loss zone (CRE44.24(1))
    if D <= K_IRB:
        return rw_max, {
            **base_details,
            "case": "CRE44.24(1) D<=K_IRB",
            "rw_pre_standard_floor": rw_max,
        }

    # Case (2): A >= K_IRB - tranche entirely above K_IRB (CRE44.24(2))
    if A >= K_IRB:
        K_SSFA = compute_K_SSFA(p, K_IRB, A, D)
        rw = rw_factor_inv * K_SSFA
        return rw, {
            **base_details,
            "case": "CRE44.24(2) A>=K_IRB",
            "K_SSFA": K_SSFA,
            "rw_pre_standard_floor": rw,
        }

    # Case (3): A < K_IRB < D - blended (CRE44.24(3))
    # K_SSFA is the same CRE44.23 quantity as case (2): compute_K_SSFA already
    # bakes in l = max(A-K_IRB, 0), which clips to 0 here because A < K_IRB.
    # The clip is part of the definition of K_SSFA(K_IRB) - there is no separate
    # "full" vs "upper" K_SSFA - so this is reported under the same canonical
    # key as case (2).
    K_SSFA = compute_K_SSFA(p, K_IRB, A, D)
    weight_loss = (K_IRB - A) / (D - A)
    weight_above = (D - K_IRB) / (D - A)
    rw = rw_factor_inv * (weight_loss + weight_above * K_SSFA)
    return rw, {
        **base_details,
        "case": "CRE44.24(3) A<K_IRB<D blended",
        "K_SSFA": K_SSFA,
        "weight_loss_portion": weight_loss,
        "weight_above_portion": weight_above,
        "rw_pre_standard_floor": rw,
    }
