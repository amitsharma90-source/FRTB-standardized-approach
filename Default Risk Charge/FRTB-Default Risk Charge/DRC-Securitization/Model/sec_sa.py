"""
sec_sa.py — SEC-SA risk weight (CRE41).

Implements:
  - K_A = (1-W) * K_SA + 0.5 * W                                    CRE41.8
  - W-unknown gates: stub for CRE41.9-10 (W is always known in v5 config)
  - Supervisory parameter p:
        non-resec:  p = 1.0                                          CRE41.12
        STC:        p = 0.5                                          CRE41.21
        resec:      p = 1.5                                          CRE41.16(3)
  - SSFA core: a = -1/(p*K_A), u = D - K_A, l = max(A - K_A, 0)     CRE41.11
        K_SSFA = (e^(a*u) - e^(a*l)) / (a * (u - l))
  - Risk weight by tranche position vs K_A:                         CRE41.13
        D <= K_A           ->  RW = 1250%
        A >= K_A           ->  RW = K_SSFA / capital_factor
        A <  K_A < D       ->  blended (weighted avg of 1250% and the above)
    where capital_factor = 0.08 (Basel 8% capital ratio), so 1/0.08 = 12.5.

All regulatory scalars (p values, the 1250% cap, the 8% capital factor) are
sourced from FRTB_Sec_Config.xlsx -> SEC_Constants. Floors (15% standard,
10%/15% STC, 100% resec) are applied DOWNSTREAM by sec_caps as part of the
floors-and-caps wrap. This module returns the pre-floor RW.
"""
from __future__ import annotations
import math
import pandas as pd


# ── CONFIG ACCESS ─────────────────────────────────────────────────────────────

def _load_sa_constants(config: dict) -> dict:
    """Pull SEC-SA scalars from SEC_Constants.

    Keys returned:
      p_non_resec, p_stc, p_resec   -- CRE41.12 / 41.21 / 41.16(3)
      rw_max                        -- 1250% cap, CRE41.13(1)
      rw_factor_inv                 -- 1/0.08 = 12.5 (Basel 8% capital ratio)
    """
    if "SEC_Constants" not in config:
        raise KeyError(
            "sec_sa requires config['SEC_Constants']; load the FRTB_Sec_Config "
            "workbook via sec_loader.load_sec_config() and pass the result."
        )
    sc = config["SEC_Constants"].set_index("Constant")
    return {
        "p_non_resec":    float(sc.at["p_sa_non_resec", "Value"]),
        "p_stc":          float(sc.at["p_sa_stc",       "Value"]),
        "p_resec":        float(sc.at["p_sa_resec",     "Value"]),
        "rw_max":         float(sc.at["rw_d_lte_KA",    "Value"]),
        "rw_factor_inv":  1.0 / float(sc.at["rw_factor_to_capital", "Value"]),
    }


# ── BUILDING BLOCKS ───────────────────────────────────────────────────────────

def compute_K_A(K_SA: float, W: float) -> float:
    """CRE41.8: K_A = (1-W) * K_SA + 0.5 * W."""
    return (1.0 - W) * K_SA + 0.5 * W


def select_p(is_resecuritisation: bool, is_stc: bool, consts: dict) -> float:
    """CRE41.12 / 41.16(3) / 41.21.

    `consts` is the dict returned by `_load_sa_constants(config)`.
    Resecuritisation precedence: a resec cannot also be STC (resec excluded
    from STC by definition), but if both flags are set the resec p (1.5) wins
    as the more conservative choice.
    """
    if is_resecuritisation:
        return consts["p_resec"]
    if is_stc:
        return consts["p_stc"]
    return consts["p_non_resec"]


def compute_K_SSFA(p: float, K_A: float, A: float, D: float) -> float:
    """CRE41.11: SSFA core formula.

    a = -1 / (p * K_A)
    u = D - K_A
    l = max(A - K_A, 0)
    K_SSFA = (e^(a*u) - e^(a*l)) / (a * (u - l))

    Returns K_SSFA as a decimal (capital per unit of exposure).
    Caller should multiply by 12.5 to convert to risk weight.
    """
    a = -1.0 / (p * K_A)
    u = D - K_A
    l = max(A - K_A, 0.0)
    if abs(u - l) < 1e-12:
        # Zero-thickness exposure: K_SSFA collapses; treat as zero capital.
        return 0.0
    return (math.exp(a * u) - math.exp(a * l)) / (a * (u - l))


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def compute_rw(row: pd.Series, config: dict) -> tuple[float, dict]:
    """SEC-SA risk weight per CRE41.

    Inputs read from row:
      K_SA               (decimal, pool-level SA capital charge)
      W                  (decimal, delinquency ratio per CRE41.6-7)
      Attachment Pt (%) / Detachment Pt (%)
      is_resecuritisation
      is_stc_compliant

    Returns: (rw_decimal_pre_floor, details_dict)
    """
    K_SA = row.get("K_SA")
    if K_SA is None or (isinstance(K_SA, float) and pd.isna(K_SA)):
        return float("nan"), {
            "approach": "SEC-SA",
            "error": "K_SA missing — cannot compute K_A",
        }
    K_SA = float(K_SA)

    W = float(row.get("W", 0.0) or 0.0)
    A = float(row.get("Attachment Pt (%)", 0.0)) / 100.0
    D = float(row.get("Detachment Pt (%)", 100.0)) / 100.0
    is_resec = bool(row.get("is_resecuritisation", False))
    is_stc = bool(row.get("is_stc_compliant", False))

    consts = _load_sa_constants(config)
    K_A = compute_K_A(K_SA, W)
    p = select_p(is_resec, is_stc, consts)
    rw_max = consts["rw_max"]
    rw_factor_inv = consts["rw_factor_inv"]

    base_details = {
        "approach": "SEC-SA",
        "K_SA": K_SA,
        "W": W,
        "K_A": K_A,
        "p": p,
        "A": A,
        "D": D,
        "is_resecuritisation": is_resec,
        "is_stc": is_stc,
    }

    # Case (1): D <= K_A — entire tranche in loss zone (CRE41.13(1))
    if D <= K_A:
        return rw_max, {
            **base_details,
            "case": "CRE41.13(1) D<=K_A",
            "rw_pre_standard_floor": rw_max,
        }

    # Case (2): A >= K_A — entire tranche above K_A (CRE41.13(2))
    if A >= K_A:
        K_SSFA = compute_K_SSFA(p, K_A, A, D)
        rw = rw_factor_inv * K_SSFA
        return rw, {
            **base_details,
            "case": "CRE41.13(2) A>=K_A",
            "K_SSFA": K_SSFA,
            "rw_pre_standard_floor": rw,
        }

    # Case (3): A < K_A < D — blended (CRE41.13(3))
    # K_SSFA is the same CRE41.11 quantity as case (2): compute_K_SSFA already
    # bakes in l = max(A-K_A, 0), which clips to 0 here because A < K_A. The
    # clip is part of the definition of K_SSFA(K_A) — there is no separate
    # "full" vs "upper" K_SSFA — so this is reported under the same canonical
    # key as case (2).
    K_SSFA = compute_K_SSFA(p, K_A, A, D)
    weight_loss = (K_A - A) / (D - A)
    weight_above = (D - K_A) / (D - A)
    rw = rw_factor_inv * (weight_loss + weight_above * K_SSFA)
    return rw, {
        **base_details,
        "case": "CRE41.13(3) A<K_A<D blended",
        "K_SSFA": K_SSFA,
        "weight_loss_portion": weight_loss,
        "weight_above_portion": weight_above,
        "rw_pre_standard_floor": rw,
    }
