"""
FRTB SA Capital Engine — Stage 2, Phase 4
curvature.py: curvature aggregation per MAR21.98-101.

Stage 1 already emits CVR_k^UP and CVR_k^DN for every curvature-eligible
position (MAR21.98), so this module only aggregates — no shock repricing.

Per-bucket (MAR21.100):
    Kb^UP = sqrt( max(0,
               sum_k max(CVR_k^UP, 0)^2
             + sum_{k != l} (rho_kl)^2 * CVR_k^UP * CVR_l^UP * psi(CVR_k^UP, CVR_l^UP)
             ))
    Kb^DN = similar on the DN side
    Kb    = max(Kb^UP, Kb^DN)
    side  = "UP" if Kb = Kb^UP else "DN"
    Sb    = sum_k CVR_k^{side}

Per risk class (MAR21.101):
    Curvature = sqrt( max(0,
                 sum_b Kb^2
               + sum_{b != c} (gamma_bc)^2 * Sb * Sc * psi(Sb, Sc)
                 ))

psi(a, b) = 0 if a <= 0 and b <= 0, else 1  [MAR21.100 footnote].

Correlations:
  - intra-bucket rho: same rho as delta (squared here) — reuse Phase 2 builders
  - inter-bucket gamma: same gamma as delta (squared here) — reuse Phase 3 builders

Scenario scaling (MAR21.6) applied to rho and gamma BEFORE squaring.

Portfolio scope note:
  Stage 1 emits curvature for SPX options (risk factor = SPX spot) and callable
  bonds (risk factor = parallel shift of USD yield curve). So each curvature
  bucket in this portfolio has exactly one distinct risk factor post-aggregation
  across positions, and both the intra rho^2 and inter gamma^2 terms collapse
  to zero. The general form is still coded for future portfolios.
"""
import os
import math
import numpy as np
import pandas as pd

from capital_loader import load_sensitivity_grid, load_config
from weighted_sensitivities import compute_weighted_sensitivities
from intra_bucket import (
    _girr_tenor_corr_matrix, _girr_rho,
    _csr_nonsec_rho_params, _csr_nonsec_rho,
    _csr_sec_nonctp_params, _csr_sec_nonctp_rho,
    _equity_cross_issuer_rho, _equity_rho,
    _commodity_rho,
    scenario_fn,
)
from inter_bucket import (
    _girr_gamma, _csr_nonsec_gamma, _csr_nonsec_gamma_matrix,
    _csr_sec_nonctp_gamma, _equity_gamma, _commodity_gamma, _fx_gamma,
)


def _psi(a: float, b: float) -> float:
    """psi(a, b) = 0 if a <= 0 AND b <= 0, else 1 [MAR21.100]."""
    return 0.0 if (a <= 0 and b <= 0) else 1.0


# ── RISK FACTOR IDENTITY ─────────────────────────────────────────────────────

def _risk_factor_key(row) -> tuple:
    """Identify the underlying risk factor for a curvature row, independent of
    which position generated it. CVRs on the same factor from different
    positions must be summed before bucket aggregation.
    """
    rc = row["risk_class"]
    if rc == "GIRR":
        return ("GIRR", row["ccy"])
    if rc == "EQUITY":
        return ("EQUITY", int(row["bucket"]), row["name"])
    if rc == "COMMODITY":
        return ("COMMODITY", int(row["bucket"]), row["name"])
    if rc == "FX":
        return ("FX", row["name"])
    # CSR curvature [MAR21.97 + MAR21.99]: parallel shift on the issuer credit
    # spread curve. Curvature risk factor identity = (risk_class, bucket, issuer)
    # where issuer comes from the position's `security` field.
    if rc.startswith("CSR"):
        return (rc, int(row["bucket"]), row.get("security"))
    return (rc, int(row["bucket"]) if row["bucket"] is not None else None,
            row.get("name"), row.get("ccy"))


# ── PAIRWISE RHO FOR CURVATURE ───────────────────────────────────────────────

def _curv_rho(a_key, b_key, cfg, girr_tenor_mat, csr_nonsec_params) -> float:
    """Return rho between two risk factors identified by keys from _risk_factor_key.
    For curvature, all factors are point factors — no tenor/basis sub-structure.
    Same factor returns 1.0 (only relevant if a key appears twice, which we
    prevent by aggregating CVR per key first).
    """
    if a_key == b_key:
        return 1.0
    rc_a, rc_b = a_key[0], b_key[0]
    if rc_a != rc_b:
        return 0.0
    rc = rc_a
    if rc == "GIRR":
        return 1.0   # parallel-shift factor, one per currency; within a bucket
    if rc == "EQUITY":
        if a_key[1] != b_key[1]:
            return 0.0   # different bucket
        cross = _equity_cross_issuer_rho(int(a_key[1]), cfg)
        if cross is None:
            return 0.0
        return cross   # same bucket, different name
    if rc == "COMMODITY":
        if a_key[1] != b_key[1]:
            return 0.0
        rw_df = cfg["COMMODITY_RW"].set_index("bucket")
        return float(rw_df.loc[int(a_key[1]), "intra_corr_commodity"])
    if rc == "CSR_NONSEC":
        # MAR21.100: for CSR non-sec curvature, only rho_name applies (rho_tenor
        # and rho_basis collapse because curvature is a single parallel-shift
        # factor per issuer). Same bucket + same issuer = 1; same bucket diff
        # issuer = rho_name from MAR21.54-55.
        if a_key[1] != b_key[1]:
            return 0.0   # different bucket
        if a_key[2] == b_key[2]:
            return 1.0   # same issuer
        bucket = int(a_key[1])
        bp = _csr_nonsec_rho_params(cfg, bucket)
        return bp["rho_name_diff"]
    if rc == "CSR_SEC_NONCTP":
        # MAR21.100 same logic; rho_tranche from MAR21.68 same/diff
        if a_key[1] != b_key[1]:
            return 0.0
        if a_key[2] == b_key[2]:
            return 1.0
        return _csr_sec_nonctp_params(cfg)["rho_tranche_diff"]
    return 0.0   # FX has single factor per pair


# ── BUCKET-LEVEL CURVATURE (MAR21.100) ───────────────────────────────────────

def _bucket_curvature(cvr_up: np.ndarray, cvr_dn: np.ndarray,
                       rho_matrix: np.ndarray) -> tuple[float, float, str]:
    """
    Given arrays of CVR^UP and CVR^DN for the distinct risk factors in a bucket
    and the (already scenario-scaled) correlation matrix rho, return (Kb, Sb, side).
    """
    def _side(cvr: np.ndarray) -> float:
        n = len(cvr)
        if n == 0:
            return 0.0
        pos_sq = float(((np.maximum(cvr, 0.0)) ** 2).sum())
        cross = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                cross += 2.0 * (rho_matrix[i, j] ** 2) * cvr[i] * cvr[j] * \
                         _psi(cvr[i], cvr[j])
        return math.sqrt(max(0.0, pos_sq + cross))

    k_up = _side(cvr_up)
    k_dn = _side(cvr_dn)
    if k_up >= k_dn:
        return k_up, float(cvr_up.sum()), "UP"
    return k_dn, float(cvr_dn.sum()), "DN"


# ── COMPUTE PER RISK CLASS ───────────────────────────────────────────────────

def _build_rho_matrix(keys: list, cfg: dict, scen) -> np.ndarray:
    """Build symmetric rho matrix (scenario-scaled) over the given factor keys."""
    n = len(keys)
    mat = np.eye(n)
    girr_tenor_mat = _girr_tenor_corr_matrix(cfg)
    for i in range(n):
        for j in range(i + 1, n):
            rho = _curv_rho(keys[i], keys[j], cfg, girr_tenor_mat, None)
            rho = scen(rho)
            mat[i, j] = mat[j, i] = rho
    return mat


def _rc_gamma_fn(rc: str, cfg: dict):
    """Return gamma function matching the delta inter-bucket rule for this class."""
    if rc == "GIRR":
        return lambda b1, b2: _girr_gamma(b1, b2, cfg)
    if rc == "CSR_NONSEC":
        mat = _csr_nonsec_gamma_matrix(cfg)
        return lambda b1, b2: _csr_nonsec_gamma(b1, b2, cfg, mat)
    if rc == "CSR_SEC_NONCTP":
        return lambda b1, b2: _csr_sec_nonctp_gamma(b1, b2, cfg)
    if rc == "EQUITY":
        return lambda b1, b2: _equity_gamma(b1, b2, cfg)
    if rc == "COMMODITY":
        return lambda b1, b2: _commodity_gamma(b1, b2, cfg)
    if rc == "FX":
        return lambda b1, b2: _fx_gamma(b1, b2, cfg)
    raise KeyError(rc)


def _bucket_id_for_curv(row) -> object:
    rc = row["risk_class"]
    if rc == "GIRR":   return row["ccy"]
    if rc == "FX":     return row["name"]
    return int(row["bucket"])


def compute_curvature(ws_df: pd.DataFrame, cfg: dict,
                      scenario: str = "medium") -> pd.DataFrame:
    """
    Aggregate curvature into per-risk-class charges (MAR21.100 + MAR21.101).

    Returns a DataFrame with one row per risk class:
        risk_class, n_buckets, charge, scenario.
    Also attaches per-bucket detail via the returned `bucket_detail` column (list
    of dicts) so the output Excel can show Kb, Sb, side per bucket.
    """
    scen = scenario_fn(scenario)

    curv = ws_df[ws_df["kind"] == "curvature"].copy()
    if curv.empty:
        empty_rc = pd.DataFrame(columns=["risk_class", "n_buckets",
                                          "charge", "scenario"])
        empty_detail = pd.DataFrame(columns=["risk_class", "bucket_id",
                                              "n_factors", "Kb", "Sb",
                                              "side", "scenario"])
        return empty_rc, empty_detail

    # Identity key per risk factor (for aggregation across positions)
    curv["rf_key"] = curv.apply(_risk_factor_key, axis=1)
    curv["bucket_id"] = curv.apply(_bucket_id_for_curv, axis=1)

    out_rows = []
    detail_rows = []

    for rc, rc_df in curv.groupby("risk_class"):
        # Per bucket, compute Kb / Sb
        bucket_Kb = []
        bucket_Sb = []
        bucket_ids = []
        bucket_sides = []

        for bid, bdf in rc_df.groupby("bucket_id"):
            # Sum CVR per risk factor within bucket, pair up UP and DN
            up = bdf[bdf["up_dn"] == "UP"].groupby("rf_key")["value"].sum()
            dn = bdf[bdf["up_dn"] == "DN"].groupby("rf_key")["value"].sum()
            # Align on the same key universe
            keys = sorted(set(up.index).union(set(dn.index)))
            cvr_up = np.array([up.get(k, 0.0) for k in keys], dtype=float)
            cvr_dn = np.array([dn.get(k, 0.0) for k in keys], dtype=float)
            rho_mat = _build_rho_matrix(list(keys), cfg, scen)
            Kb, Sb, side = _bucket_curvature(cvr_up, cvr_dn, rho_mat)
            bucket_Kb.append(Kb)
            bucket_Sb.append(Sb)
            bucket_ids.append(bid)
            bucket_sides.append(side)
            detail_rows.append(dict(risk_class=rc, bucket_id=bid,
                                    n_factors=len(keys),
                                    Kb=Kb, Sb=Sb, side=side,
                                    scenario=scenario))

        # Inter-bucket aggregation (MAR21.101) with gamma^2 and psi
        gamma_fn = _rc_gamma_fn(rc, cfg)
        n = len(bucket_ids)
        discr = float(sum(k * k for k in bucket_Kb))
        for i in range(n):
            for j in range(i + 1, n):
                g = scen(gamma_fn(bucket_ids[i], bucket_ids[j]))
                discr += 2.0 * (g ** 2) * bucket_Sb[i] * bucket_Sb[j] * \
                         _psi(bucket_Sb[i], bucket_Sb[j])
        charge = math.sqrt(max(0.0, discr))
        out_rows.append(dict(risk_class=rc, n_buckets=n,
                             charge=charge, scenario=scenario))

    result = pd.DataFrame(out_rows).sort_values("risk_class").reset_index(drop=True)
    detail = pd.DataFrame(detail_rows).sort_values(
        ["risk_class", "bucket_id"]).reset_index(drop=True)
    return result, detail


def compute_curvature_with_trace(ws_df: pd.DataFrame, cfg: dict,
                                 scenario: str = "medium"):
    """Like compute_curvature but additionally returns:
        bucket_detail_df  per (risk_class, bucket): Kb, Sb, side, n_factors
        distinct_factors_df  per distinct MAR21 risk factor: summed CVR_UP,
                             summed CVR_DN, sign flags, side that won, and
                             that factor's contribution to bucket Sb
        position_factors_df  per original tidy CURV row: position id, side,
                             raw CVR value, contributing to which factor key
    """
    scen = scenario_fn(scenario)
    curv = ws_df[ws_df["kind"] == "curvature"].copy()
    if curv.empty:
        return (pd.DataFrame(columns=["risk_class", "n_buckets", "charge", "scenario"]),
                pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    curv["rf_key"] = curv.apply(_risk_factor_key, axis=1)
    curv["bucket_id"] = curv.apply(_bucket_id_for_curv, axis=1)

    summary_rows = []
    bucket_rows = []
    distinct_factor_rows = []
    position_factor_rows = []

    for rc, rc_df in curv.groupby("risk_class"):
        bucket_Kb, bucket_Sb, bucket_ids, bucket_sides = [], [], [], []

        for bid, bdf in rc_df.groupby("bucket_id"):
            up = bdf[bdf["up_dn"] == "UP"].groupby("rf_key")["value"].sum()
            dn = bdf[bdf["up_dn"] == "DN"].groupby("rf_key")["value"].sum()
            keys = sorted(set(up.index).union(set(dn.index)))
            cvr_up = np.array([up.get(k, 0.0) for k in keys], dtype=float)
            cvr_dn = np.array([dn.get(k, 0.0) for k in keys], dtype=float)
            rho_mat = _build_rho_matrix(list(keys), cfg, scen)
            Kb, Sb, side = _bucket_curvature(cvr_up, cvr_dn, rho_mat)

            bucket_Kb.append(Kb)
            bucket_Sb.append(Sb)
            bucket_ids.append(bid)
            bucket_sides.append(side)

            n_position_rows_in_bucket = len(bdf)
            bucket_rows.append(dict(
                scenario=scenario, risk_class=rc, bucket_id=bid,
                n_distinct_factors=len(keys),
                n_position_rows=n_position_rows_in_bucket,
                Kb=Kb, Sb=Sb, side=side))

            for k_idx, k in enumerate(keys):
                distinct_factor_rows.append(dict(
                    scenario=scenario, risk_class=rc, bucket_id=bid,
                    risk_factor_key=str(k),
                    cvr_up_summed=float(cvr_up[k_idx]),
                    cvr_dn_summed=float(cvr_dn[k_idx]),
                    cvr_up_positive=cvr_up[k_idx] > 0,
                    cvr_dn_positive=cvr_dn[k_idx] > 0,
                    bucket_winning_side=side,
                    contribution_to_bucket_Sb=float(
                        cvr_up[k_idx] if side == "UP" else cvr_dn[k_idx]),
                    n_position_rows=int(((bdf["rf_key"] == k)).sum()),
                ))

            # Per-position trace: every CURV_*_UP / CURV_*_DN row
            for _, row in bdf.iterrows():
                position_factor_rows.append(dict(
                    scenario=scenario, risk_class=rc, bucket_id=bid,
                    risk_factor_key=str(row["rf_key"]),
                    position_id=row["id"], security=row["security"],
                    column=row["column"], up_dn=row["up_dn"],
                    cvr_value=float(row["value"]),
                ))

        gamma_fn = _rc_gamma_fn(rc, cfg)
        n = len(bucket_ids)
        discr = float(sum(k * k for k in bucket_Kb))
        for i in range(n):
            for j in range(i + 1, n):
                g = scen(gamma_fn(bucket_ids[i], bucket_ids[j]))
                discr += 2.0 * (g ** 2) * bucket_Sb[i] * bucket_Sb[j] * \
                         _psi(bucket_Sb[i], bucket_Sb[j])
        charge = math.sqrt(max(0.0, discr))
        summary_rows.append(dict(risk_class=rc, n_buckets=n,
                                 charge=charge, scenario=scenario))

    summary_df = pd.DataFrame(summary_rows).sort_values(
        "risk_class").reset_index(drop=True)
    buckets_df = pd.DataFrame(bucket_rows).sort_values(
        ["risk_class", "bucket_id"]).reset_index(drop=True)
    distinct_df = pd.DataFrame(distinct_factor_rows).sort_values(
        ["risk_class", "bucket_id", "risk_factor_key"]).reset_index(drop=True)
    position_df = pd.DataFrame(position_factor_rows).sort_values(
        ["risk_class", "bucket_id", "risk_factor_key", "position_id", "up_dn"]
    ).reset_index(drop=True)
    return summary_df, buckets_df, distinct_df, position_df


if __name__ == "__main__":
    print("Loading Phase 0 + Phase 1...")
    tidy = load_sensitivity_grid()
    cfg = load_config()
    ws = compute_weighted_sensitivities(tidy, cfg)

    for scen in ("medium", "high", "low"):
        print(f"\n=== PHASE 4 CURVATURE | scenario = {scen} ===")
        charges, detail = compute_curvature(ws, cfg, scenario=scen)
        print("Per-bucket detail:")
        print(detail.to_string(
            index=False,
            formatters={"Kb": "{:,.2f}".format, "Sb": "{:,.2f}".format}))
        print("\nPer risk class:")
        print(charges.to_string(
            index=False, formatters={"charge": "{:,.2f}".format}))
        print(f"Total curvature charge: {charges['charge'].sum():,.2f}")

    out_path = os.path.join(os.path.dirname(__file__), "output",
                            "phase4_curvature.xlsx")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    readme = pd.DataFrame([
        ("Sheet", "Purpose"),
        ("README", "This sheet."),
        ("factors_per_position",
         "Every original CURV_*_UP / CURV_*_DN tidy row with position id and "
         "security. Raw CVR values from Stage 1, SCENARIO-INDEPENDENT (CVR is "
         "a Stage 1 number, no rho involvement) so one sheet covers all three "
         "scenarios. Sum within (risk_factor_key, up_dn) equals the "
         "corresponding cvr_up_summed / cvr_dn_summed in factors_distinct."),
        ("{scen}_summary",
         "Per-risk-class curvature charge under MAR21.101: "
         "Curv_rc = sqrt( sum_b Kb^2 + sum_{b!=c} gamma_bc^2 * Sb * Sc * psi(Sb,Sc) )."),
        ("{scen}_buckets",
         "Per-bucket Kb (= max(Kb_UP, Kb_DN)), Sb (= sum of CVR on the winning "
         "side), side, and n_distinct_factors vs n_position_rows. Kb depends on "
         "rho^2 cross-terms when n_distinct_factors >= 2, hence per-scenario."),
        ("{scen}_factors_distinct",
         "One row per distinct MAR21 risk factor in each bucket. cvr_up_summed "
         "and cvr_dn_summed are scenario-independent sums of the per-position "
         "CVR cells; bucket_winning_side and contribution_to_bucket_Sb depend "
         "on Kb which depends on rho^2 cross-terms (per scenario when buckets "
         "have >= 2 factors)."),
        ("Notes",
         "psi(a,b) = 0 if a<=0 AND b<=0, else 1 (MAR21.100 footnote). "
         "Intra-bucket cross-terms use rho^2; inter-bucket cross-terms use "
         "gamma^2; both rho and gamma are scenario-scaled per MAR21.6 BEFORE "
         "squaring. In this portfolio every curvature bucket has a single "
         "post-aggregation factor, so rho^2 cross-terms collapse to zero, and "
         "every risk class has a single bucket with curvature, so gamma^2 "
         "cross-terms also collapse to zero -- charge = max(Kb_UP, Kb_DN) per "
         "risk class."),
    ], columns=["A", "B"])

    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        readme.to_excel(xw, sheet_name="README", index=False, header=False)

        # Scenario-independent per-position trace — write once
        _, _, _, pos_f = compute_curvature_with_trace(ws, cfg, scenario="medium")
        pos_f.drop(columns=["scenario"], errors="ignore").to_excel(
            xw, sheet_name="factors_per_position", index=False)

        for scen in ("medium", "high", "low"):
            summary, buckets, distinct_f, _ = compute_curvature_with_trace(
                ws, cfg, scenario=scen)
            summary.to_excel(xw, sheet_name=f"{scen}_summary", index=False)
            buckets.to_excel(xw, sheet_name=f"{scen}_buckets", index=False)
            distinct_f.to_excel(xw, sheet_name=f"{scen}_factors_distinct", index=False)
    print(f"\nWrote: {out_path}")
