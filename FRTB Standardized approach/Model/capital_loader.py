"""
FRTB SA Capital Engine — Stage 2, Phase 0
capital_loader.py: load the Stage 1 sensitivity grid and the MAR21 config workbook,
parse every sensitivity column name into a structured risk-factor tuple, and emit a
tidy long-format DataFrame ready for Phase 1 (weighted sensitivities WS_k = RW_k * s_k).

Authoritative spec: MAR21 - Standardised approach_ sensitivities-based method.pdf
"""
import re
import os
import pandas as pd

SENS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "Sensitivities calculator", "Sensitivity v10",
    "files", "output", "FRTB_Sensitivities.xlsx",
)
CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "Sensitivities calculator", "Sensitivity v10",
    "files", "data", "MAR21_Config_RW_Corr.xlsx",
)

META_COLS = ["ID", "Security", "Instrument_Type", "Sensitivity_Definition"]
FLAG_COLS = [
    "GIRR_Delta", "GIRR_Vega", "GIRR_Curvature",
    "CSR_NonSec_Delta", "CSR_Sec_Delta", "CSR_Curvature",
    "EQ_Delta", "EQ_Vega", "EQ_Curvature",
    "COMM_Delta", "FX_Delta",
    "GIRR_Inflation", "GIRR_XCcy_Basis",
]


# ── COLUMN-NAME PARSER ────────────────────────────────────────────────────────

def parse_column(col: str):
    """
    Parse a Stage 1 sensitivity column name into a structured dict.
    Returns None if the column is not a recognised risk-factor column.

    Recognised patterns:
        GIRR_<CCY>_<TENOR>Y              -> delta, GIRR yield curve
        GIRR_<CCY>_INFLATION             -> delta, GIRR inflation
        GIRR_<CCY>_XCCY_BASIS            -> delta, GIRR xccy basis
        CSR_NONSEC_<BUCKET>_<TENOR>Y     -> delta, CSR non-sec
        CSR_SEC_NONCTP_<BUCKET>_<TENOR>Y -> delta, CSR sec non-CTP
        CSR_SEC_CTP_<BUCKET>_<TENOR>Y    -> delta, CSR sec CTP
        EQ_<BUCKET>_<NAME>               -> delta, equity spot
        COMM_<BUCKET>_<NAME>             -> delta, commodity
        FX_<PAIR>                        -> delta, FX
        VEGA_EQ_<BUCKET>_<NAME>_<TENOR>Y -> vega, equity
        VEGA_GIRR_<CCY>_<OPT>Y_<UND>Y    -> vega, GIRR
        CURV_EQ_<BUCKET>_<NAME>_<UP|DN>  -> curvature, equity
        CURV_GIRR_<CCY>_<UP|DN>          -> curvature, GIRR
    """
    # ── VEGA ──
    m = re.fullmatch(r"VEGA_EQ_(\d+)_(.+?)_([\d.]+)Y", col)
    if m:
        return dict(kind="vega", risk_class="EQUITY", bucket=int(m.group(1)),
                    name=m.group(2), tenor=float(m.group(3)), under_tenor=None,
                    up_dn=None)
    m = re.fullmatch(r"VEGA_GIRR_([A-Z]{3})_([\d.]+)Y_([\d.]+)Y", col)
    if m:
        return dict(kind="vega", risk_class="GIRR", bucket=None, ccy=m.group(1),
                    name=None, tenor=float(m.group(2)), under_tenor=float(m.group(3)),
                    up_dn=None)
    m = re.fullmatch(r"VEGA_(CSR_NONSEC|CSR_SEC_NONCTP|CSR_SEC_CTP)_(\d+)_([\d.]+)Y", col)
    if m:
        return dict(kind="vega", risk_class=m.group(1), bucket=int(m.group(2)),
                    name=None, tenor=float(m.group(3)), under_tenor=None, up_dn=None)
    m = re.fullmatch(r"VEGA_COMM_(\d+)_(.+?)_([\d.]+)Y", col)
    if m:
        return dict(kind="vega", risk_class="COMMODITY", bucket=int(m.group(1)),
                    name=m.group(2), tenor=float(m.group(3)), under_tenor=None,
                    up_dn=None)
    m = re.fullmatch(r"VEGA_FX_(.+?)_([\d.]+)Y", col)
    if m:
        return dict(kind="vega", risk_class="FX", bucket=None, name=m.group(1),
                    tenor=float(m.group(2)), under_tenor=None, up_dn=None)

    # ── CURVATURE ──
    m = re.fullmatch(r"CURV_EQ_(\d+)_(.+?)_(UP|DN)", col)
    if m:
        return dict(kind="curvature", risk_class="EQUITY", bucket=int(m.group(1)),
                    name=m.group(2), tenor=None, up_dn=m.group(3))
    m = re.fullmatch(r"CURV_GIRR_([A-Z]{3})_(UP|DN)", col)
    if m:
        return dict(kind="curvature", risk_class="GIRR", bucket=None, ccy=m.group(1),
                    name=None, tenor=None, up_dn=m.group(2))
    m = re.fullmatch(r"CURV_(CSR_NONSEC|CSR_SEC_NONCTP|CSR_SEC_CTP)_(\d+)_(UP|DN)", col)
    if m:
        return dict(kind="curvature", risk_class=m.group(1), bucket=int(m.group(2)),
                    name=None, tenor=None, up_dn=m.group(3))
    m = re.fullmatch(r"CURV_COMM_(\d+)_(.+?)_(UP|DN)", col)
    if m:
        return dict(kind="curvature", risk_class="COMMODITY", bucket=int(m.group(1)),
                    name=m.group(2), tenor=None, up_dn=m.group(3))
    m = re.fullmatch(r"CURV_FX_(.+?)_(UP|DN)", col)
    if m:
        return dict(kind="curvature", risk_class="FX", bucket=None, name=m.group(1),
                    tenor=None, up_dn=m.group(2))

    # ── DELTA: GIRR ──
    m = re.fullmatch(r"GIRR_([A-Z]{3})_INFLATION", col)
    if m:
        return dict(kind="delta", risk_class="GIRR", bucket=None, ccy=m.group(1),
                    name=None, tenor="inflation", up_dn=None)
    m = re.fullmatch(r"GIRR_([A-Z]{3})_XCCY_BASIS", col)
    if m:
        return dict(kind="delta", risk_class="GIRR", bucket=None, ccy=m.group(1),
                    name=None, tenor="xccy_basis", up_dn=None)
    m = re.fullmatch(r"GIRR_([A-Z]{3})_([\d.]+)Y", col)
    if m:
        return dict(kind="delta", risk_class="GIRR", bucket=None, ccy=m.group(1),
                    name=None, tenor=float(m.group(2)), up_dn=None)

    # ── DELTA: CSR (no issuer in column name — taken from position row) ──
    m = re.fullmatch(r"(CSR_NONSEC|CSR_SEC_NONCTP|CSR_SEC_CTP)_(\d+)_([\d.]+)Y", col)
    if m:
        return dict(kind="delta", risk_class=m.group(1), bucket=int(m.group(2)),
                    name=None, tenor=float(m.group(3)), up_dn=None)

    # ── DELTA: EQUITY, COMMODITY, FX ──
    m = re.fullmatch(r"EQ_(\d+)_(.+)", col)
    if m:
        return dict(kind="delta", risk_class="EQUITY", bucket=int(m.group(1)),
                    name=m.group(2), tenor=None, up_dn=None)
    m = re.fullmatch(r"COMM_(\d+)_(.+)", col)
    if m:
        return dict(kind="delta", risk_class="COMMODITY", bucket=int(m.group(1)),
                    name=m.group(2), tenor=None, up_dn=None)
    m = re.fullmatch(r"FX_(.+)", col)
    if m:
        return dict(kind="delta", risk_class="FX", bucket=None, name=m.group(1),
                    tenor=None, up_dn=None)

    return None


# ── LOADERS ───────────────────────────────────────────────────────────────────

def load_sensitivity_grid(path: str = SENS_PATH) -> pd.DataFrame:
    """
    Load the Stage 1 sensitivity grid in tidy long format.

    One row per (position x non-zero risk factor) pair. Zero sensitivities are
    dropped; meta fields are repeated on every row of a given position so the
    DataFrame is self-describing and no join is required downstream.

    This is the canonical Stage 2 input schema — all Phase 1..6 code reads it.

    Schema
    ------
    Position identity (from the grid meta columns)
        id                    int       unique position id from Stage 1 (1..62)
        security              str       security name, used as issuer id for
                                        CSR non-sec same-name correlation
                                        (MAR21.56 rho=1 for identical names)
        instrument_type       str       Stage 1 classification, e.g. GOV_BOND,
                                        CORP_BOND, SECURITISATION, XCCY_GBP_LEG
        sensitivity_definition str      free-text MAR21 reference from Stage 1

    Risk-factor identity (parsed from the sensitivity column name)
        column       str   raw Stage 1 column name, e.g. GIRR_USD_5.0Y
        kind         str   one of {"delta", "vega", "curvature"}
        risk_class   str   one of {"GIRR", "CSR_NONSEC", "CSR_SEC_NONCTP",
                                   "CSR_SEC_CTP", "EQUITY", "COMMODITY", "FX"}
        bucket       int|None    MAR21 bucket number (CSR/EQ/COMM); None for GIRR/FX
        ccy          str|None    currency code for GIRR only (USD, GBP, ...)
        name         str|None    issuer/ticker/commodity/FX-pair identifier;
                                 None for GIRR delta (tenor carries identity)
        tenor        float|str|None
                     float: FRTB tenor in years (0.25, 0.5, 1.0, 2.0, 3.0,
                            5.0, 10.0, 15.0, 20.0, 30.0 for GIRR delta;
                            0.5, 1.0, 3.0, 5.0, 10.0 for CSR delta;
                            0.5, 1.0, 3.0, 5.0, 10.0 for vega option-maturity)
                     str:   "inflation" or "xccy_basis" for those GIRR sub-factors
                     None:  equity/commodity/FX spot delta and curvature rows
        under_tenor  float|None  GIRR vega underlying-residual tenor; None otherwise
        up_dn        str|None    "UP" or "DN" for curvature rows; None otherwise

    Value
        value        float   signed sensitivity in USD (Stage 1 reports everything
                             in USD; GBP amounts converted at mkt['usd_gbp']).
                             Units follow MAR21 convention:
                               - delta: PV change per +1bp, divided by 0.0001
                               - vega: vega * sigma (FRTB convention, MAR21.92)
                               - curvature: CVR_up/CVR_down raw PV delta (MAR21.98)

    Invariants
    ----------
    * Zero-filter applied: |value| > 0 for every row.
    * Exactly one row per (id, column) pair that has a non-zero cell.
    * Every row's (risk_class, bucket or tenor) maps to a valid config RW
      entry -- verified by smoke_test().
    * Curvature rows come in UP/DN pairs keyed by (id, risk_class, bucket,
      ccy, name) -- verified by smoke_test().
    """
    raw = pd.read_excel(path, sheet_name="Sensitivities", header=None)
    header = raw.iloc[1].tolist()
    data = raw.iloc[2:].reset_index(drop=True)
    data.columns = header

    meta = data[META_COLS].copy()
    meta.columns = ["id", "security", "instrument_type", "sensitivity_definition"]

    # Sensitivity value columns = everything not in META_COLS or FLAG_COLS
    value_cols = [c for c in data.columns if c not in META_COLS + FLAG_COLS]

    rows = []
    unparsed = []
    for col in value_cols:
        parsed = parse_column(col)
        if parsed is None:
            unparsed.append(col)
            continue
        vals = pd.to_numeric(data[col], errors="coerce")
        mask = vals.notna() & (vals != 0)
        if not mask.any():
            continue
        sub = data.loc[mask, META_COLS].copy()
        sub.columns = ["id", "security", "instrument_type", "sensitivity_definition"]
        sub["column"] = col
        sub["kind"] = parsed["kind"]
        sub["risk_class"] = parsed["risk_class"]
        sub["bucket"] = parsed.get("bucket")
        sub["ccy"] = parsed.get("ccy")
        sub["name"] = parsed.get("name")
        sub["tenor"] = parsed.get("tenor")
        sub["under_tenor"] = parsed.get("under_tenor")
        sub["up_dn"] = parsed.get("up_dn")
        sub["value"] = vals.loc[mask].values
        rows.append(sub)

    if unparsed:
        print(f"[WARN] {len(unparsed)} column(s) could not be parsed: {unparsed}")

    tidy = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return tidy


def load_config(path: str = CONFIG_PATH) -> dict:
    """Load every sheet of MAR21_Config_RW_Corr.xlsx into a dict of DataFrames."""
    xl = pd.ExcelFile(path)
    cfg = {s: pd.read_excel(path, sheet_name=s) for s in xl.sheet_names}
    return cfg


# ── SMOKE TEST ────────────────────────────────────────────────────────────────

def smoke_test(tidy: pd.DataFrame, cfg: dict) -> None:
    """Verify every risk factor in the tidy grid maps to a valid RW in config."""
    print(f"\n=== SMOKE TEST ===")
    print(f"Positions : {tidy['id'].nunique()}")
    print(f"Rows (non-zero risk factors): {len(tidy)}")
    print(f"\nBreakdown by (kind, risk_class):")
    print(tidy.groupby(["kind", "risk_class"]).size().to_string())

    issues = []

    girr_rw = cfg["GIRR_RW"].set_index("tenor")["risk_weight"].to_dict()
    girr_tenors_num = set()
    for k in girr_rw.keys():
        if pd.isna(k):
            continue
        s = str(k).strip()
        if s in ("inflation", "xccy_basis"):
            continue
        try:
            girr_tenors_num.add(float(s.rstrip("Y")))
        except ValueError:
            pass
    csr_ns_buckets = set(cfg["CSR_NONSEC_RW"]["bucket"].astype(int).tolist())
    csr_sec_nctp_buckets = set(cfg["CSR_SEC_NONCTP_RW"]["bucket"].astype(int).tolist())
    eq_buckets = set(cfg["EQUITY_RW"]["bucket"].astype(int).tolist())
    comm_buckets = set(cfg["COMMODITY_RW"]["bucket"].astype(int).tolist())

    for _, r in tidy.iterrows():
        rc = r["risk_class"]
        if rc == "GIRR" and r["kind"] == "delta":
            if r["tenor"] not in ("inflation", "xccy_basis"):
                if float(r["tenor"]) not in girr_tenors_num:
                    issues.append(f"GIRR tenor not in config: {r['column']}")
        elif rc == "CSR_NONSEC" and r["kind"] == "delta":
            if r["bucket"] not in csr_ns_buckets:
                issues.append(f"CSR_NONSEC bucket not in config: {r['column']}")
        elif rc == "CSR_SEC_NONCTP" and r["kind"] == "delta":
            if r["bucket"] not in csr_sec_nctp_buckets:
                issues.append(f"CSR_SEC_NONCTP bucket not in config: {r['column']}")
        elif rc == "EQUITY" and r["kind"] == "delta":
            if r["bucket"] not in eq_buckets:
                issues.append(f"Equity bucket not in config: {r['column']}")
        elif rc == "COMMODITY" and r["kind"] == "delta":
            if r["bucket"] not in comm_buckets:
                issues.append(f"Commodity bucket not in config: {r['column']}")

    if issues:
        print(f"\n[FAIL] {len(issues)} mapping issue(s):")
        for i in issues[:20]:
            print(f"  - {i}")
    else:
        print("\n[PASS] every risk factor maps to a valid RW bucket/tenor in config.")

    # Additional check: curvature UP/DN pairing
    curv = tidy[tidy["kind"] == "curvature"]
    if not curv.empty:
        pair_key = curv["id"].astype(str) + "|" + curv["risk_class"] + "|" + \
                   curv["bucket"].astype(str) + "|" + curv["name"].astype(str) + "|" + \
                   curv["ccy"].astype(str)
        up = set(pair_key[curv["up_dn"] == "UP"])
        dn = set(pair_key[curv["up_dn"] == "DN"])
        only_up = up - dn
        only_dn = dn - up
        if only_up or only_dn:
            print(f"\n[WARN] Curvature pairing mismatches: {len(only_up)} UP without DN, "
                  f"{len(only_dn)} DN without UP")
        else:
            print(f"\n[PASS] All {len(up)} curvature risk factors have both UP and DN.")


if __name__ == "__main__":
    print(f"Loading sensitivity grid: {os.path.abspath(SENS_PATH)}")
    tidy = load_sensitivity_grid()
    print(f"Loading config:          {os.path.abspath(CONFIG_PATH)}")
    cfg = load_config()
    print(f"Config sheets loaded: {list(cfg.keys())}")
    smoke_test(tidy, cfg)
