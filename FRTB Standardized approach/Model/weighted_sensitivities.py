"""
FRTB SA Capital Engine — Stage 2, Phase 1
weighted_sensitivities.py: compute WS_k = RW_k * s_k for every row of the
Phase 0 tidy grid.

Spec: MAR21.4 (definition of WS_k), with per-risk-class risk weights from:
    GIRR delta            MAR21.42-44  (tenor-indexed + specified-ccy divisor)
    GIRR inflation/xccy   MAR21.43     (flat RW equal to GIRR max-curve RW)
    CSR non-sec delta     MAR21.53     (bucket-indexed)
    CSR sec non-CTP       MAR21.62-66  (bucket-indexed)
    CSR sec CTP           MAR21.70     (bucket-indexed; unused if portfolio empty)
    Equity delta          MAR21.77     (bucket-indexed, rw_spot column)
    Commodity delta       MAR21.83     (bucket-indexed)
    FX delta              MAR21.87-88  (flat RW + specified-pair divisor)
    Vega (all classes)    MAR21.92     (pre-computed in VEGA_RW.risk_weight)

Curvature rows are *not* weighted here — MAR21.98 defines CVR_k directly
(already computed in Stage 1) and aggregation uses CVR, not RW*s. Phase 4
handles curvature separately; Phase 1 passes curvature rows through with
rw = NaN and ws = NaN, so downstream code can filter by kind.

All constants read from cfg (MAR21_Config_RW_Corr.xlsx) — no hardcoded values.
"""
import os
import math
import pandas as pd

from capital_loader import load_sensitivity_grid, load_config


# ── RW LOOKUP BUILDERS ────────────────────────────────────────────────────────

def _build_girr_tenor_rw(cfg: dict) -> dict:
    """Map normalised GIRR tenor key -> risk weight from GIRR_RW sheet.
    Keys: float tenors (0.25, 0.5, 1.0, ...) and strings 'inflation', 'xccy_basis'.
    """
    df = cfg["GIRR_RW"][["tenor", "risk_weight"]].dropna(subset=["tenor"])
    out: dict = {}
    for _, r in df.iterrows():
        key = str(r["tenor"]).strip()
        if key in ("inflation", "xccy_basis"):
            out[key] = float(r["risk_weight"])
        else:
            try:
                out[float(key.rstrip("Y"))] = float(r["risk_weight"])
            except ValueError:
                pass
    return out


def _build_specified_ccys(cfg: dict) -> tuple[set, float]:
    """Return (set of specified currencies, divisor) from SPECIFIED_CCYS sheet."""
    df = cfg["SPECIFIED_CCYS"]
    ccys = set(df["currency"].astype(str).str.strip().str.upper()) - {"DOMESTIC"}
    divisor = float(df["girr_rw_divisor"].iloc[0])
    return ccys, divisor


def _build_bucket_rw(cfg: dict, sheet: str, rw_col: str = "risk_weight") -> dict:
    df = cfg[sheet][["bucket", rw_col]]
    return {int(r["bucket"]): float(r[rw_col]) for _, r in df.iterrows()}


def _build_fx_params(cfg: dict) -> tuple[float, set, float]:
    """Return (base FX RW, set of specified pairs (direction-agnostic), divisor)."""
    df = cfg["FX_PARAMS"].set_index("parameter")["value"]
    rw = float(df.loc["risk_weight"])
    divisor = float(df.loc["specified_pairs_rw_divisor"])
    raw_pairs = str(df.loc["specified_pairs"]).split(",")
    pairs: set = set()
    for p in raw_pairs:
        p = p.strip().upper()
        if not p or "/" not in p:
            continue
        a, b = p.split("/")
        pairs.add(frozenset({a, b}))
    return rw, pairs, divisor


def _build_vega_rw(cfg: dict) -> dict:
    """Map risk_class-tag -> pre-computed vega RW from VEGA_RW sheet.
    Tag mapping handles equity large vs small cap selection downstream.
    """
    df = cfg["VEGA_RW"].set_index("risk_class")["risk_weight"]
    return {str(k): float(v) for k, v in df.items()}


def build_lookups(cfg: dict) -> dict:
    """Bundle all RW lookups into a single dict consumed by _apply_rw."""
    girr_tenor_rw = _build_girr_tenor_rw(cfg)
    specified_ccys, girr_ccy_div = _build_specified_ccys(cfg)
    fx_rw, fx_specified_pairs, fx_pair_div = _build_fx_params(cfg)
    return dict(
        girr_tenor_rw=girr_tenor_rw,
        specified_ccys=specified_ccys,
        girr_ccy_div=girr_ccy_div,
        csr_nonsec_rw=_build_bucket_rw(cfg, "CSR_NONSEC_RW"),
        csr_sec_nonctp_rw=_build_bucket_rw(cfg, "CSR_SEC_NONCTP_RW"),
        csr_sec_ctp_rw=_build_bucket_rw(cfg, "CSR_SEC_CTP_RW"),
        equity_rw=_build_bucket_rw(cfg, "EQUITY_RW", rw_col="rw_spot"),
        commodity_rw=_build_bucket_rw(cfg, "COMMODITY_RW"),
        fx_rw=fx_rw,
        fx_specified_pairs=fx_specified_pairs,
        fx_pair_div=fx_pair_div,
        vega_rw=_build_vega_rw(cfg),
        equity_bucket_meta=cfg["EQUITY_RW"][["bucket", "market_cap", "economy"]]
            .set_index("bucket").to_dict("index"),
    )


# ── PER-RISK-CLASS RW RESOLUTION ──────────────────────────────────────────────

def _rw_girr_delta(row, lu) -> tuple[float, float, str]:
    """MAR21.42-44: tenor-indexed RW, divided by sqrt(2) for specified currencies.
    Returns (rw_base, divisor, divisor_reference)."""
    tenor_key = row["tenor"] if row["tenor"] in ("inflation", "xccy_basis") \
                else float(row["tenor"])
    rw_base = lu["girr_tenor_rw"][tenor_key]
    ccy = str(row["ccy"]).upper()
    if ccy in lu["specified_ccys"]:
        return rw_base, lu["girr_ccy_div"], "MAR21.44 specified ccy"
    return rw_base, 1.0, ""


def _rw_csr_delta(row, lu) -> tuple[float, float, str]:
    """MAR21.53 / MAR21.62-66 / MAR21.70: bucket-indexed RW; no divisor."""
    rc = row["risk_class"]
    bucket = int(row["bucket"])
    if rc == "CSR_NONSEC":       return lu["csr_nonsec_rw"][bucket], 1.0, ""
    if rc == "CSR_SEC_NONCTP":   return lu["csr_sec_nonctp_rw"][bucket], 1.0, ""
    if rc == "CSR_SEC_CTP":      return lu["csr_sec_ctp_rw"][bucket], 1.0, ""
    raise KeyError(f"Unknown CSR risk_class: {rc}")


def _rw_equity_delta(row, lu) -> tuple[float, float, str]:
    return lu["equity_rw"][int(row["bucket"])], 1.0, ""


def _rw_commodity_delta(row, lu) -> tuple[float, float, str]:
    return lu["commodity_rw"][int(row["bucket"])], 1.0, ""


def _rw_fx_delta(row, lu) -> tuple[float, float, str]:
    """MAR21.87-88: flat RW, divided by sqrt(2) for specified pairs."""
    rw_base = lu["fx_rw"]
    pair = str(row["name"]).upper()
    if "/" in pair:
        a, b = pair.split("/")
        if frozenset({a, b}) in lu["fx_specified_pairs"]:
            return rw_base, lu["fx_pair_div"], "MAR21.88 specified pair"
    return rw_base, 1.0, ""


def _rw_vega(row, lu) -> tuple[float, float, str]:
    """MAR21.92: pre-computed in VEGA_RW.risk_weight. No divisor."""
    rc = row["risk_class"]
    if rc == "EQUITY":
        bucket = int(row["bucket"])
        meta = lu["equity_bucket_meta"].get(bucket, {})
        mcap = str(meta.get("market_cap", "")).strip().lower()
        tag = "EQUITY_LARGE" if mcap == "large" else "EQUITY_SMALL"
        return lu["vega_rw"][tag], 1.0, ""
    tag_map = {"GIRR": "GIRR", "CSR_NONSEC": "CSR_NONSEC",
               "CSR_SEC_NONCTP": "CSR_SEC_NONCTP", "CSR_SEC_CTP": "CSR_SEC_CTP",
               "COMMODITY": "COMMODITY", "FX": "FX"}
    return lu["vega_rw"][tag_map[rc]], 1.0, ""


def _apply_rw(row, lu):
    """Dispatch to the right RW resolver.
    Returns (rw_base, divisor, rw, ws, divisor_reference) for the row.
    Curvature rows return NaNs for rw_base/rw/ws and divisor=1.0.
    """
    kind = row["kind"]
    if kind == "curvature":
        return (float("nan"), 1.0, float("nan"), float("nan"), "")
    rc = row["risk_class"]
    if kind == "vega":
        rw_base, divisor, ref = _rw_vega(row, lu)
    elif rc == "GIRR":
        rw_base, divisor, ref = _rw_girr_delta(row, lu)
    elif rc.startswith("CSR"):
        rw_base, divisor, ref = _rw_csr_delta(row, lu)
    elif rc == "EQUITY":
        rw_base, divisor, ref = _rw_equity_delta(row, lu)
    elif rc == "COMMODITY":
        rw_base, divisor, ref = _rw_commodity_delta(row, lu)
    elif rc == "FX":
        rw_base, divisor, ref = _rw_fx_delta(row, lu)
    else:
        raise KeyError(f"Unhandled risk_class: {rc}")
    rw = rw_base / divisor
    return (rw_base, divisor, rw, rw * float(row["value"]), ref)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def compute_weighted_sensitivities(tidy: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return tidy grid augmented with full RW decomposition so every weight is
    traceable to its MAR21 paragraph:
        rw_base            raw RW from config (MAR21.42 / .53 / .62 / .77 / .83 / .87 / .92)
        divisor            1.0, or sqrt(2) where MAR21.44 (GIRR specified ccy) or
                           MAR21.88 (FX specified pair) applies
        divisor_reference  MAR21 citation for a non-trivial divisor; "" otherwise
        rw                 rw_base / divisor  (the weight actually used in WS)
        ws                 rw * value
    Curvature rows have rw_base = rw = ws = NaN (handled by Phase 4).
    """
    lu = build_lookups(cfg)
    rw_base_list, div_list, rw_list, ws_list, ref_list = [], [], [], [], []
    for _, row in tidy.iterrows():
        rw_base, divisor, rw, ws, ref = _apply_rw(row, lu)
        rw_base_list.append(rw_base)
        div_list.append(divisor)
        rw_list.append(rw)
        ws_list.append(ws)
        ref_list.append(ref)
    out = tidy.copy()
    out["rw_base"] = rw_base_list
    out["divisor"] = div_list
    out["divisor_reference"] = ref_list
    out["rw"] = rw_list
    out["ws"] = ws_list
    return out


# ── SUMMARY / AUDIT ───────────────────────────────────────────────────────────

def summary(ws_df: pd.DataFrame) -> pd.DataFrame:
    """Per (kind, risk_class) summary: row count, sum |s|, sum |ws|."""
    g = ws_df.groupby(["kind", "risk_class"])
    out = pd.DataFrame({
        "rows": g.size(),
        "sum_abs_s": g["value"].apply(lambda x: x.abs().sum()),
        "sum_abs_ws": g["ws"].apply(lambda x: x.abs().sum(skipna=True)),
        "min_rw": g["rw"].min(),
        "max_rw": g["rw"].max(),
    }).reset_index()
    return out


def sanity_checks(ws_df: pd.DataFrame) -> None:
    issues = []
    # 1. Every non-curvature row must have a finite RW and WS
    non_curv = ws_df[ws_df["kind"] != "curvature"]
    miss_rw = non_curv[non_curv["rw"].isna()]
    if len(miss_rw):
        issues.append(f"{len(miss_rw)} non-curvature row(s) missing RW")
    miss_ws = non_curv[non_curv["ws"].isna()]
    if len(miss_ws):
        issues.append(f"{len(miss_ws)} non-curvature row(s) missing WS")
    # 2. Every curvature row must have NaN RW/WS (passed through)
    curv = ws_df[ws_df["kind"] == "curvature"]
    bad_curv = curv[curv["rw"].notna() | curv["ws"].notna()]
    if len(bad_curv):
        issues.append(f"{len(bad_curv)} curvature row(s) unexpectedly weighted")
    # 3. WS sign must match sensitivity sign (RW is always >= 0)
    wrong_sign = non_curv[(non_curv["value"] > 0) & (non_curv["ws"] < 0)]
    wrong_sign2 = non_curv[(non_curv["value"] < 0) & (non_curv["ws"] > 0)]
    if len(wrong_sign) or len(wrong_sign2):
        issues.append(f"{len(wrong_sign)+len(wrong_sign2)} row(s) with WS/s sign mismatch")
    # 4. RW ranges
    rw_max = non_curv["rw"].max()
    if rw_max > 1.0:
        issues.append(f"Max RW = {rw_max:.4f} > 1.0 — check config")

    if issues:
        print("[FAIL] Sanity issues:")
        for i in issues: print(f"  - {i}")
    else:
        print("[PASS] All Phase 1 sanity checks clear.")


if __name__ == "__main__":
    print("Loading Phase 0 tidy grid and config...")
    tidy = load_sensitivity_grid()
    cfg = load_config()

    print(f"Applying MAR21.4 weighting to {len(tidy)} risk factor rows...")
    ws_df = compute_weighted_sensitivities(tidy, cfg)

    print("\n=== PHASE 1 SUMMARY: WS_k = RW_k * s_k ===")
    print(summary(ws_df).to_string(index=False))

    print("\n=== SANITY CHECKS ===")
    sanity_checks(ws_df)

    out_path = os.path.join(os.path.dirname(__file__), "output",
                            "phase1_weighted_sensitivities.xlsx")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Build a "README" frame explaining each column so the file is self-describing
    readme = pd.DataFrame([
        ("id", "int", "Stage 1 position id"),
        ("security", "str", "Security name (issuer id for CSR same-name)"),
        ("instrument_type", "str", "Stage 1 instrument classification"),
        ("column", "str", "Original Stage 1 sensitivity column name"),
        ("kind", "str", "delta / vega / curvature"),
        ("risk_class", "str", "GIRR / CSR_NONSEC / CSR_SEC_NONCTP / EQUITY / COMMODITY / FX"),
        ("bucket", "int|None", "MAR21 bucket (CSR/EQ/COMM); None for GIRR/FX"),
        ("ccy", "str|None", "Currency for GIRR"),
        ("name", "str|None", "Issuer/ticker/commodity/FX-pair identifier"),
        ("tenor", "float|str", "FRTB tenor in years; 'inflation'/'xccy_basis' for sub-factors"),
        ("under_tenor", "float|None", "GIRR vega underlying-residual tenor"),
        ("up_dn", "str|None", "UP/DN for curvature rows"),
        ("value", "float", "Raw sensitivity in USD (Stage 1 output)"),
        ("rw_base", "float", "Raw MAR21 risk weight before any divisor (MAR21.42/.53/.62/.77/.83/.87/.92)"),
        ("divisor", "float", "1.0 by default; sqrt(2)=1.41421 where MAR21.44 (GIRR specified ccy) or MAR21.88 (FX specified pair) applies"),
        ("divisor_reference", "str", "MAR21 paragraph justifying a non-trivial divisor"),
        ("rw", "float", "Final RW used in WS: rw = rw_base / divisor"),
        ("ws", "float", "Weighted sensitivity: ws = rw * value  (MAR21.4 definition)"),
    ], columns=["column", "dtype", "description"])

    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        readme.to_excel(xw, sheet_name="README", index=False)
        ws_df.to_excel(xw, sheet_name="Weighted_Sensitivities", index=False)

        # Extra convenience: rows where the divisor is non-trivial, so the
        # reviewer can see at a glance which risk factors benefited from MAR21.44/88
        div_rows = ws_df[ws_df["divisor"] > 1.0]
        if len(div_rows):
            div_rows.to_excel(xw, sheet_name="Divisor_Applied", index=False)
    print(f"\nWrote: {out_path}")
    n_div = int((ws_df['divisor'] > 1.0).sum())
    print(f"  Rows where MAR21.44/.88 sqrt(2) divisor applied: {n_div}")
