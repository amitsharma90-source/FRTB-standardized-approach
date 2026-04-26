"""
Audit FRTB_Sensitivities.xlsx: cross-check every flag column against the
corresponding sensitivity value columns, in both directions.

Type A mismatch: flag = TRUE but no matching sensitivity value is populated
                 (expected a risk to be present, but output is empty)
Type B mismatch: flag = FALSE but one or more matching sensitivity values are
                 populated (output has a risk the flag said was absent)

The flag → column mapping mirrors the MAR21 risk-class taxonomy used by Stage 1:
    GIRR_Delta        -> GIRR_<CCY>_<TENOR>Y           (excl. INFLATION, XCCY_BASIS)
    GIRR_Vega         -> VEGA_GIRR_*
    GIRR_Curvature    -> CURV_GIRR_*
    GIRR_Inflation    -> GIRR_<CCY>_INFLATION
    GIRR_XCcy_Basis   -> GIRR_<CCY>_XCCY_BASIS
    CSR_NonSec_Delta  -> CSR_NONSEC_*
    CSR_Sec_Delta     -> CSR_SEC_NONCTP_* / CSR_SEC_CTP_*
    EQ_Delta          -> EQ_<BUCKET>_<NAME>            (excl. VEGA_EQ_*, CURV_EQ_*)
    EQ_Vega           -> VEGA_EQ_*
    EQ_Curvature      -> CURV_EQ_*
    COMM_Delta        -> COMM_*
    FX_Delta          -> FX_<PAIR>
"""
import os
import re
import pandas as pd

SENS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "Sensitivities calculator", "Sensitivity v10",
    "files", "output", "FRTB_Sensitivities.xlsx",
)
META_COLS = ["ID", "Security", "Instrument_Type", "Sensitivity_Definition"]
FLAG_COLS = [
    "GIRR_Delta", "GIRR_Vega", "GIRR_Curvature",
    "CSR_NonSec_Delta", "CSR_Sec_Delta", "CSR_Curvature",
    "EQ_Delta", "EQ_Vega", "EQ_Curvature",
    "COMM_Delta", "FX_Delta",
    "GIRR_Inflation", "GIRR_XCcy_Basis",
]


def classify_value_column(col: str) -> str | None:
    """Return the flag column this sensitivity column belongs to, or None."""
    if re.fullmatch(r"VEGA_GIRR_[A-Z]{3}_[\d.]+Y_[\d.]+Y", col):
        return "GIRR_Vega"
    if re.fullmatch(r"VEGA_EQ_\d+_.+_[\d.]+Y", col):
        return "EQ_Vega"
    if re.fullmatch(r"CURV_GIRR_[A-Z]{3}_(UP|DN)", col):
        return "GIRR_Curvature"
    if re.fullmatch(r"CURV_EQ_\d+_.+_(UP|DN)", col):
        return "EQ_Curvature"
    if re.fullmatch(r"CURV_(CSR_NONSEC|CSR_SEC_NONCTP|CSR_SEC_CTP)_\d+_(UP|DN)", col):
        return "CSR_Curvature"
    if re.fullmatch(r"GIRR_[A-Z]{3}_INFLATION", col):
        return "GIRR_Inflation"
    if re.fullmatch(r"GIRR_[A-Z]{3}_XCCY_BASIS", col):
        return "GIRR_XCcy_Basis"
    if re.fullmatch(r"GIRR_[A-Z]{3}_[\d.]+Y", col):
        return "GIRR_Delta"
    if re.fullmatch(r"CSR_NONSEC_\d+_[\d.]+Y", col):
        return "CSR_NonSec_Delta"
    if re.fullmatch(r"(CSR_SEC_NONCTP|CSR_SEC_CTP)_\d+_[\d.]+Y", col):
        return "CSR_Sec_Delta"
    if re.fullmatch(r"EQ_\d+_.+", col):
        return "EQ_Delta"
    if re.fullmatch(r"COMM_\d+_.+", col):
        return "COMM_Delta"
    if re.fullmatch(r"FX_.+", col):
        return "FX_Delta"
    return None


def load_grid(path: str = SENS_PATH) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Sensitivities", header=None)
    data = raw.iloc[2:].reset_index(drop=True)
    data.columns = raw.iloc[1].tolist()
    return data


def audit(path: str = SENS_PATH):
    df = load_grid(path)

    value_cols = [c for c in df.columns if c not in META_COLS + FLAG_COLS]
    col_to_flag = {c: classify_value_column(c) for c in value_cols}
    unknown = [c for c, f in col_to_flag.items() if f is None]
    if unknown:
        print(f"[WARN] {len(unknown)} value column(s) could not be classified: {unknown}")

    flag_to_cols: dict[str, list[str]] = {f: [] for f in FLAG_COLS}
    for c, f in col_to_flag.items():
        if f is not None:
            flag_to_cols[f].append(c)

    print("Flag -> value-column coverage:")
    for f in FLAG_COLS:
        print(f"  {f:20s} -> {len(flag_to_cols[f]):3d} column(s)")
    print()

    # Per-row audit
    mismatches_a: list[dict] = []   # flag TRUE, no value
    mismatches_b: list[dict] = []   # flag FALSE, value present

    for _, row in df.iterrows():
        pid = row["ID"]
        sec = row["Security"]
        itype = row["Instrument_Type"]
        for flag in FLAG_COLS:
            flag_val = bool(row[flag]) if pd.notna(row[flag]) else False
            cols_for_flag = flag_to_cols[flag]
            if not cols_for_flag:
                continue
            numeric = pd.to_numeric(row[cols_for_flag], errors="coerce").fillna(0)
            has_value = (numeric != 0).any()
            populated = numeric[numeric != 0].index.tolist()

            if flag_val and not has_value:
                mismatches_a.append(dict(id=pid, security=sec, instrument_type=itype,
                                        flag=flag, populated_cols=""))
            if (not flag_val) and has_value:
                mismatches_b.append(dict(id=pid, security=sec, instrument_type=itype,
                                        flag=flag, populated_cols=", ".join(populated)))

    print(f"Positions audited: {len(df)}")
    print(f"Flags checked per position: {len(FLAG_COLS)}")
    print(f"Total flag × position checks: {len(df) * len(FLAG_COLS)}")
    print()
    print(f"Type A (flag TRUE, value empty): {len(mismatches_a)}")
    if mismatches_a:
        print(pd.DataFrame(mismatches_a).to_string(index=False))
    print()
    print(f"Type B (flag FALSE, value present): {len(mismatches_b)}")
    if mismatches_b:
        print(pd.DataFrame(mismatches_b).to_string(index=False))

    if not mismatches_a and not mismatches_b:
        print("\n[PASS] Every flag reconciles cleanly with its sensitivity values.")


if __name__ == "__main__":
    audit()
