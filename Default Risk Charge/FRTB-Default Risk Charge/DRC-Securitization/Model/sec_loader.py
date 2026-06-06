"""
sec_loader.py — Step 2 of the DRC sec build sequence.

Reads the v5 portfolio Combined Holdings sheet and the FRTB_Sec_Config.xlsx
config workbook, joins per-position rows with deal/pool/override metadata, and
emits a flat DataFrame ready for sec_classifier / sec_hierarchy / sec_engine.

NO RW logic lives here. The loader's job is data assembly + derivation of the
small set of fields that depend on portfolio-wide context (tranche key,
is_senior_derived, M_T, signed notional, NPL flag).

Authoritative spec for derived fields:
    is_senior_derived       CRE40.18 (waterfall position, NOT data label)
    M_T_for_drc             MAR22.34(1) DRC overlay (= 1 year, used by SEC-ERBA)
    M_T_banking_book_years  CRE40.22 / 40.23 assumed cashflow-weighted maturity
                            from Deal_Master (used by SEC-IRBA p-formula)
    is_npl_derived          CRE45.1 (W >= 0.90)
    signed_notional         derived from Long/Short + |Quantity/Notional|
    tranche_key             engine contract: (Pool ID, A, D, Position Sub-Type)

Loader precedence: portfolio file > Deal_Master > Pool_Defaults > hard fallback.
(Position_Overrides has been removed - it was always empty in practice;
per-position fields come from the portfolio file when needed, deal-level
fields from Deal_Master.)
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
    "Input data", "FRTB_Sec_Config.xlsx",
))
SENSITIVITIES_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..",
    "Output data", "Sensitivities output", "FRTB_Sensitivities.xlsx",
))
MV_SHEET = "Portfolio_MV_Decomposed"
OUTPUT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..",
    "Output data", "SA capital charge",
))
OUTPUT_SNAPSHOT = os.path.join(OUTPUT_DIR, "drc_sec_phase1_load.xlsx")

DEFAULT_ASOF = dt.date(2026, 2, 5)

# ── PORTFOLIO LOAD ────────────────────────────────────────────────────────────

PORTFOLIO_SHEET = "Combined Holdings"
SEC_POSITION_TYPE = "Securitisation"


def load_portfolio_sec_rows(path: str = PORTFOLIO_PATH) -> pd.DataFrame:
    """Read Combined Holdings, filter to securitisation rows.

    Securitisation rows are identified by Position Type == 'Securitisation'
    AND a non-null Underlying Pool Type (defence in depth).
    """
    df = pd.read_excel(path, sheet_name=PORTFOLIO_SHEET)
    mask = (df["Position Type"] == SEC_POSITION_TYPE) & df["Underlying Pool Type"].notna()
    sec = df.loc[mask].copy().reset_index(drop=True)
    return sec


def load_sec_mv_from_sheet2(path: str = SENSITIVITIES_PATH) -> pd.Series:
    """Read MV_USD from Sheet 2 of FRTB_Sensitivities.xlsx, indexed by ID.

    Sec positions are not decomposed (parent_position_id == ID), so the join
    against the loader frame is on ID. Holdings-file `Market Value ($)` is
    deliberately not consumed; the MAR22.34(3) fair-value cap uses this
    model-priced MV instead.
    """
    df = pd.read_excel(path, sheet_name=MV_SHEET)
    df["ID"] = df["ID"].astype(int)
    return df.set_index("ID")["MV_USD"].astype(float)


# ── CONFIG LOAD ───────────────────────────────────────────────────────────────

CONFIG_SHEETS = [
    "SEC_ERBA_LongTerm_Table2",
    "SEC_ERBA_LongTerm_Table4_STC",
    "SEC_ERBA_ShortTerm_Table1",
    "SEC_ERBA_ShortTerm_Table3_STC",
    "SEC_IRBA_p_Table1",
    "SEC_IRBA_p_Table2_STC",
    "SEC_IAA_Mapping",
    "SEC_Floors",
    "SEC_Constants",
    "NPL_Constants",
    "DRC_Overlay",
    "Bank_Capabilities",
    "Jurisdiction_Permissions",
    "Pool_Defaults",
    "Deal_Master",
]


def load_sec_config(path: str = CONFIG_PATH) -> dict[str, pd.DataFrame]:
    """Read every sheet of the sec config workbook; return dict keyed by sheet name."""
    cfg = {name: pd.read_excel(path, sheet_name=name) for name in CONFIG_SHEETS}
    return cfg


# ── DERIVATIONS ───────────────────────────────────────────────────────────────

def derive_signed_notional(sec: pd.DataFrame) -> pd.Series:
    """signed_notional = sign(Long/Short) * |Quantity/Notional|.

    Cross-checks the Long/Short label against the sign of the Quantity/Notional
    field; raises if they disagree (data integrity issue).
    """
    qn = sec["Quantity/Notional"]
    label = sec["Long/Short"].str.strip().str.lower()
    # Determine sign from label
    sign_from_label = label.map({"long": 1, "short": -1})
    if sign_from_label.isna().any():
        bad = sec.loc[sign_from_label.isna(), ["ID", "Long/Short"]]
        raise ValueError(f"Unrecognised Long/Short label(s):\n{bad}")
    signed = sign_from_label * qn.abs()
    # Sanity: if the original Quantity/Notional was already signed, both methods agree
    label_disagrees = (qn.abs() > 0) & ((qn > 0) != (sign_from_label > 0))
    if label_disagrees.any():
        bad = sec.loc[label_disagrees, ["ID", "Long/Short", "Quantity/Notional"]]
        raise ValueError(
            f"Long/Short label disagrees with sign of Quantity/Notional:\n{bad}\n"
            "Using sign from label."
        )
    return signed.astype(float)


def derive_is_senior(sec: pd.DataFrame) -> pd.Series:
    """is_senior_derived per CRE40.18.

    Operational rule: a tranche is senior iff its Detachment Pt == 100%.
    Rationale: 'first claim on the entire underlying pool' (CRE40.18) is
    mathematically equivalent to D = 100% in the deal's loss waterfall.

    For the v5 portfolio this correctly flips CMBS-2024-RE1-A2 (ID 41,
    A=20, D=25) from the 'Senior' data label to non-senior_derived.

    Edge cases NOT handled here (need future extensions; flagged as TODO):
      - synthetic unrated tranches inferring seniority from a lower
        rated tranche
      - ABCP liquidity facilities qualifying as senior under the special
        full-loss-coverage condition
      - tied highest rating with same effective seniority but different maturity
    When one of these cases lands in the portfolio, expose the override
    through a portfolio-file column (NOT a new config sheet).
    """
    return (sec["Detachment Pt (%)"] == 100.0)


def compute_mt_years(maturity: pd.Series, asof: dt.date) -> pd.Series:
    """M_T = (maturity - asof) in years, clipped to [1, 5] per CRE40.22."""
    asof_ts = pd.Timestamp(asof)
    delta_days = (pd.to_datetime(maturity) - asof_ts).dt.days
    years = delta_days / 365.25
    return years.clip(lower=1.0, upper=5.0)


def derive_tranche_key(df: pd.DataFrame) -> pd.Series:
    """tranche_key = canonical string for per-tranche netting.

    Engine contract: RW is keyed by
        (Pool ID, A, D, Position Sub-Type)
    Two positions sharing this key MUST get identical RW; signed notionals
    are summed within the group before multiplying by RW * 8%.

    Why these four fields and only these four:
      - (Pool ID, A, D) defines the tranche in the deal's loss waterfall
        per CRE40.18 - one slice, one rating, one approach by construction.
      - Position Sub-Type ('Tranche' default vs 'Liquidity_Facility') is the
        ONE legitimate discriminator within an (A, D) range: the ABCP CP
        and the ABCP Liquidity Facility on the same conduit can both attach
        at A=0/D=100 but represent economically distinct claims (different
        priority, different RW approach: SEC-ERBA short-term vs SEC-IAA).
      - Rating and is_senior_derived are deliberately EXCLUDED. A standard
        tranche has one rating by definition; is_senior is a deterministic
        function of D (D==100% -> senior). Including them in the key would
        silently split a single tranche into two groups when the underlying
        portfolio data carries a typo, hiding the defect. The engine asserts
        Rating / is_senior_derived consistency within each tranche key in
        sec_engine.assert_tranche_key_consistency() instead.
    """
    return (
        df["Pool ID"].astype(str)
        + "|A=" + df["Attachment Pt (%)"].round(4).astype(str)
        + "|D=" + df["Detachment Pt (%)"].round(4).astype(str)
        + "|T=" + df["Position Sub-Type"].fillna("Tranche").astype(str)
    )


# ── MERGE WITH CONFIG ─────────────────────────────────────────────────────────

DEAL_MASTER_FIELDS = [
    "pool_type", "is_traditional", "is_resecuritisation",
    "implicit_support_provided", "is_stc_compliant",
    "W", "K_IRB", "K_SA", "N", "LGD",
    "nrppd_amount", "pool_outstanding",
    "continuous_look_through_known", "jurisdiction", "granularity_override",
    "is_originator_or_sponsor_deal",
    "dd_status",  # CRE40.31-36; per-deal DD verdict (was on Position_Overrides
                  # which has been removed for sheet-count discipline)
]
# NOTE: M_T_banking_book_years was previously pulled from Deal_Master under
# the misreading that CRE44.17's p-formula should use the actual cashflow-
# weighted maturity. MAR22.34(1) is clear that the maturity component inside
# the banking-book RW formulas is set to zero (1y assumed) for sec DRC, to
# avoid double-counting CSR-Sec migration risk. So CRE44.17 reads M_T_for_drc.
# The actual position-level remaining maturity flows separately into the
# MAR22.30 / MAR22.15-18 GROSS JTD scaling at the JTD/netting layer
# (M_T_position_years + mat_scaling_factor below). Two distinct M_T uses;
# don't merge them.

# Per-position fields the engine consumes. Most come from the portfolio
# holdings file directly (Rating, Attachment Pt, Detachment Pt, Position
# Sub-Type, Internal_Credit_Grade). The two defaults below are derived /
# convenience fields filled in by apply_position_defaults when missing:
#   is_abcp_facility       - True for ABCP liquidity facility positions
#                            (drives SEC-IAA routing per CRE40.44)
#   internal_credit_grade  - CRE43.2(7) internal grade for unrated ABCP
#                            facilities (input to the IAA -> ECAI mapping)
# inferred_rating (CRE42.9-10) is consulted by sec_erba.compute_rw via
# row.get("inferred_rating") and tolerates the column being absent.
#
# Truly-dead position-level defaults (is_senior_override, protection_*) were
# removed when Position_Overrides was deleted - they had no live consumer
# and only added noise to phase outputs. Reintroduce per-field when a
# corresponding engine consumer (e.g. sec_crm) goes live.
POSITION_DEFAULTS = {
    "is_abcp_facility":            False,
    "internal_credit_grade":       None,
}


# Columns deliberately hidden from PHASE OUTPUT SNAPSHOTS (the 10_input /
# 20_output / 30_audit sheets) because they are constant defaults across the
# CURRENT portfolio - i.e. every cell carries the same regulatory-default
# value, so they add column-width to QA views without ever changing per row.
# The engine still consumes these columns from the in-memory frame; the
# filter applies only to what is written to disk. When a future portfolio
# carries non-default values, drop the column from this list and it will
# reappear in the snapshot views automatically.
# See FRTB SA/CLAUDE.md "No noise columns in phase outputs" rule.
PHASE_OUTPUT_HIDDEN_COLUMNS = [
    # === Portfolio columns inherited from the mixed-type holdings schema ===
    # Sec rows always carry these as NaN (they belong to bonds / options /
    # equity rows in the same Combined Holdings sheet). Kept on the
    # in-memory frame for downstream consumers (drc_main, sec_engine audit
    # views) that touch them generically; filtered from sec phase outputs
    # since they add zero information per row.
    "Issuer",
    "Ticker/Index",
    "Call Date",
    "Strike Price",
    # === Quantity/Notional == signed_notional (pure duplicate) ===
    # The portfolio already encodes Long/Short into the sign of Quantity/Notional,
    # so derive_signed_notional() reproduces the same series. signed_notional is
    # the canonical engine field; Quantity/Notional is the redundant portfolio
    # column. Hidden from output to satisfy the "no duplicate columns" rule.
    "Quantity/Notional",
    # === Market Value ($) is the portfolio-file MV; MV_USD is the canonical ===
    # The DRC engine consumes ONLY MV_USD (model-priced from Sheet 2 of
    # FRTB_Sensitivities.xlsx). The portfolio column 'Market Value ($)' is
    # carried through join but never read by any DRC math; left visible it
    # invites a reviewer to compare two MVs that disagree by design
    # (portfolio MV is an entered figure, model MV is bump-and-reprice).
    # MV_USD is the single source of truth - hide the portfolio column to
    # prevent confusion.
    "Market Value ($)",
    # === Spread (bps) is consumed only by the SOFR TRS sensitivity calc ===
    # The sensitivity engine reads this for the TRS-SOFR pay-leg contractual
    # spread. For securitisations the spread inputs come from the manufactured
    # 'Sec_Tranche_Curves' sheet in FRTB_Sec_Config.xlsx, not from this
    # portfolio column. Sec phase outputs should not show it - sparse,
    # narrative-only, no sec consumer.
    "Spread (bps)",
    # === granularity_override (pool-level metadata, never consumed) ===
    # Set on the frame via apply_pool_defaults_fallback (Pool_Defaults sheet
    # supplies 'retail_senior' / 'wholesale_senior_granular' / etc.), but the
    # IRBA p-table row is selected at runtime via (is_senior_derived,
    # asset_class, N) - no consumer reads granularity_override. The 'senior'
    # token in values like 'retail_senior' is unrelated to tranche seniority;
    # leaving the column visible in outputs invites confusion with
    # is_senior_derived and Seniority (DRC).
    "granularity_override",
    # === Portfolio metadata constant by filter or by single-currency demo ===
    # 'Position Type' / 'Issue Type' / 'FRTB Risk Measures' / 'FRTB_Risk_Class'
    # are filter criteria for the sec loader (rows that pass the filter are
    # all = 'Securitisation' / 'ABS' / 'Delta' / 'CSR_SEC_NONCTP' by definition).
    # 'Currency' / 'Payment Frequency' / 'Region' are demo-portfolio constants.
    "Position Type",
    "Issue Type",
    "FRTB Risk Measures",
    "FRTB_Risk_Class",
    "Currency",
    "Payment Frequency",
    "Region",
    # === Live regulatory inputs but always-default in current portfolio ===
    # When a deal takes a non-default value, drop the entry below and the
    # column reappears in snapshot views automatically.
    "inferred_rating",                       # CRE42.9-10 fallback for unrated tranches
    "is_traditional",                        # CRE45.5 NPL senior-concession gate
    "implicit_support_provided",             # CRE40.49 bypass flag
    "continuous_look_through_known",         # CRE40.51 look-through cap eligibility
    "is_resecuritisation",                   # CRE41.16 mandatory-SA routing input
    # === Step-output flags whose live consumer modules are still stubbed ===
    # Re-show when the underlying module goes live (sec_crm decomposition,
    # sec_dd substantive checks, sec_caps look-through with real pool data).
    "spe_op_req_pass",                       # sec_engine step 2 stub
    "sec_bypass_due_to_implicit_support",    # sec_engine step 3 derived (informational)
    "crm_decomposed",                        # sec_engine step 6 stub (sec_crm)
    "crm_subtranche_count",                  # sec_engine step 6 stub (sec_crm)
    "lookthrough_cap_eligible",              # CRE40.51 eligibility (no deal eligible today)
    "lookthrough_cap_rw",                    # CRE40.50 underlying-pool avg RW
    "lookthrough_cap_binds",                 # binding indicator
]


def filter_output_columns(df):
    """Drop noise columns from a phase output view per PHASE_OUTPUT_HIDDEN_COLUMNS.

    Engine code continues to read these columns from the in-memory frame -
    this helper only affects what lands in the Excel snapshot. Returns a
    copy with the noise columns removed; original frame is untouched.
    """
    if df is None:
        return None
    return df.drop(
        columns=[c for c in PHASE_OUTPUT_HIDDEN_COLUMNS if c in df.columns]
    )


def merge_deal_master(sec: pd.DataFrame, deal_master: pd.DataFrame) -> pd.DataFrame:
    """Left-join Deal_Master fields onto sec rows by Pool ID == pool_id.

    Raises if any sec row's Pool ID is not in Deal_Master (data integrity).
    """
    missing = set(sec["Pool ID"]) - set(deal_master["pool_id"])
    if missing:
        raise ValueError(
            f"Pool IDs in portfolio not found in Deal_Master: {sorted(missing)}. "
            f"Add rows to Deal_Master in FRTB_Sec_Config.xlsx (or extend "
            f"build_sec_config.py:sheet_deal_master())."
        )
    dm = deal_master.set_index("pool_id")[DEAL_MASTER_FIELDS]
    out = sec.merge(dm, left_on="Pool ID", right_index=True, how="left", validate="many_to_one")
    return out


def apply_position_defaults(sec: pd.DataFrame) -> pd.DataFrame:
    """Fill in position-level fields that the portfolio file may not carry,
    and translate portfolio-file column names to engine field names.

    Position_Overrides has been removed (it was always empty in practice).
    Position-level signals that the engine consumes now come from one of:
      - Deal_Master (e.g. dd_status, is_originator_or_sponsor_deal)
      - the portfolio holdings file directly:
          'Position Sub-Type'      -> drives is_abcp_facility (when value is
                                     'Liquidity_Facility')
          'Internal_Credit_Grade'  -> internal_credit_grade (SEC-IAA input)
    For any field listed in POSITION_DEFAULTS that the portfolio doesn't
    supply, install the default value.
    """
    sec = sec.copy()

    # Default the portfolio's Position Sub-Type column to 'Tranche' so it is
    # always present (it joins into the tranche_key in derive_tranche_key).
    if "Position Sub-Type" not in sec.columns:
        sec["Position Sub-Type"] = "Tranche"
    else:
        sec["Position Sub-Type"] = (
            sec["Position Sub-Type"].fillna("Tranche").astype(str).str.strip()
        )

    # Translate portfolio columns to engine field names; drop the portfolio-
    # capitalised columns so each quantity appears exactly once on the frame
    # (CLAUDE.md "no duplicate columns" rule).
    sec["is_abcp_facility"] = (sec["Position Sub-Type"] == "Liquidity_Facility")
    if "Internal_Credit_Grade" in sec.columns:
        sec["internal_credit_grade"] = sec["Internal_Credit_Grade"]
        sec = sec.drop(columns=["Internal_Credit_Grade"])

    # Install defaults for any still-missing fields
    for f, default in POSITION_DEFAULTS.items():
        if f not in sec.columns:
            sec[f] = default
    return sec


def apply_pool_defaults_fallback(
    sec: pd.DataFrame, pool_defaults: pd.DataFrame,
) -> pd.DataFrame:
    """For any Deal_Master field that came back null, fall back to Pool_Defaults
    by pool_type. Currently this is a safety net — Deal_Master has all 8 v5
    deals fully populated.
    """
    sec = sec.copy()
    pd_idx = pool_defaults.set_index("Pool_type")
    pool_default_field_map = {
        "K_SA": "seed_K_SA",
        "N": "seed_N",
        "LGD": "seed_LGD",
        "granularity_override": "granularity_class",
    }
    for pool_field, default_field in pool_default_field_map.items():
        null_mask = sec[pool_field].isna()
        if null_mask.any():
            for idx in sec.index[null_mask]:
                pt = sec.at[idx, "pool_type"]
                if pt in pd_idx.index:
                    sec.at[idx, pool_field] = pd_idx.at[pt, default_field]
    return sec


# ── ORCHESTRATOR ──────────────────────────────────────────────────────────────

def build_sec_loader_frame(
    portfolio_path: str = PORTFOLIO_PATH,
    config_path: str = CONFIG_PATH,
    asof: dt.date = DEFAULT_ASOF,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Top-level loader: returns (positions_frame, full_config_dict).

    positions_frame columns (in approximate order):
      identity:    ID, Issuer, Security, Pool ID, pool_type
      tranche:     Attachment Pt (%), Detachment Pt (%), Rating, Seniority (DRC),
                   is_senior_derived, is_senior_label_override_applied,
                   tranche_key
      sizing:      Quantity/Notional, signed_notional, MV_USD (model-priced
                   from Sheet 2 of FRTB_Sensitivities.xlsx), Long/Short
      maturity:    Maturity, M_T_banking_book_years, M_T_for_drc
      deal facts:  W, K_IRB, K_SA, N, LGD, nrppd_amount, pool_outstanding,
                   is_traditional, is_resecuritisation,
                   implicit_support_provided, is_stc_compliant,
                   continuous_look_through_known, jurisdiction,
                   granularity_override, is_originator_or_sponsor_deal,
                   is_npl_derived
      overrides:   dd_status, bank_role_in_deal, inferred_rating,
                   protection_present, protection_type,
                   protection_attachment_pct, protection_detachment_pct,
                   protection_provider_id
    """
    sec = load_portfolio_sec_rows(portfolio_path)
    cfg = load_sec_config(config_path)

    sec["signed_notional"] = derive_signed_notional(sec)

    # Position-level defaults / portfolio-column translation. Position_Overrides
    # sheet has been removed; per-position fields come from the portfolio file
    # directly (Position Sub-Type, Internal_Credit_Grade, etc.).
    sec = apply_position_defaults(sec)

    # Derive is_senior from the waterfall (CRE40.18 - D == 100% means first
    # claim on the entire underlying pool). Position_Overrides has been
    # removed, so the override path that previously layered on top of this
    # is gone; if it's ever needed back, drive it from a portfolio column,
    # not a separate sheet.
    sec["is_senior_derived"] = derive_is_senior(sec)

    # Two distinct M_T values flow through the sec engine, each driving a
    # different part of the calculation:
    #
    #   M_T_for_drc            - MAR22.34(1) DRC overlay value (= 1.0). Used
    #                            INSIDE the banking-book RW formulas
    #                            (SEC-ERBA 1y/5y interpolation, SEC-IRBA
    #                            CRE44.17 p-formula). Suppresses the
    #                            maturity component so migration risk is not
    #                            double-counted with CSR-Sec. Sourced from
    #                            DRC_Overlay.mt_for_drc.
    #
    #   M_T_position_years     - Actual remaining time-to-maturity per
    #                            position, computed from the portfolio's
    #                            Maturity column. Used to derive
    #                            mat_scaling_factor below, which scales the
    #                            GROSS JTD per position per MAR22.30 +
    #                            MAR22.15-18 BEFORE the per-tranche netting
    #                            in step 9.
    sec_constants = cfg["SEC_Constants"].set_index("Constant")
    mt_for_drc = float(
        cfg["DRC_Overlay"].set_index("Constant").at["mt_for_drc", "Value"]
    )
    sec["M_T_for_drc"] = mt_for_drc

    # MAR22.30 + 22.15-18 per-position JTD scaling factor:
    #     mat_scaling_factor = max(floor, min(M_T_position_years, cap))
    # Floor 0.25, cap 1.0 per MAR22.16-17 (= sub-1y scaling, 3-month minimum).
    # M_T_position_years is the actual remaining maturity of the tranche.
    delta_days = (pd.to_datetime(sec["Maturity"]) - pd.Timestamp(asof)).dt.days
    sec["M_T_position_years"] = (delta_days / 365.25).astype(float)
    jtd_scaling_floor = float(sec_constants.at["jtd_scaling_floor_years", "Value"])
    jtd_scaling_cap = float(sec_constants.at["jtd_scaling_cap_years", "Value"])
    sec["mat_scaling_factor"] = (
        sec["M_T_position_years"].clip(lower=jtd_scaling_floor, upper=jtd_scaling_cap)
    )

    # Merge Deal_Master facts
    sec = merge_deal_master(sec, cfg["Deal_Master"])

    # Pool_Defaults safety net
    sec = apply_pool_defaults_fallback(sec, cfg["Pool_Defaults"])

    # NPL gate (CRE45.1) — derived from W
    npl_w_threshold = cfg["NPL_Constants"].set_index("Constant").at[
        "npl_w_threshold", "Value"
    ]
    sec["is_npl_derived"] = sec["W"] >= float(npl_w_threshold)

    # Tranche key (engine contract for per-tranche netting)
    sec["tranche_key"] = derive_tranche_key(sec)

    # DRC-SEC bucket (MAR22.31/32) for the bucket-level HBR aggregation in
    # MAR22.33. The bucket is (asset class x region); both dimensions already
    # exist on the holdings file — `Underlying Pool Type` is the asset class
    # (RMBS / CMBS / CLO / ABCP / ABS-Auto / ABS-CreditCard …) and `Region` is
    # the region. No new column/config is introduced; the bucket is just their
    # composite key. MAR22.31(1) "Corporates (excl SME)" is a single all-region
    # bucket — a corporate-underlying securitisation would carry an Underlying
    # Pool Type that signals it; none do in v5, so every exposure lands in an
    # asset-class x region bucket. (Region normalisation to the four MAR22.31
    # regions — Asia / Europe / North America / other — is a holdings-data
    # responsibility; the demo carries a single region.)
    sec["drc_sec_bucket"] = (
        sec["Region"].astype(str).str.strip()
        + " | " + sec["Underlying Pool Type"].astype(str).str.strip()
    )

    # Model-priced MV from Sheet 2 — used by MAR22.34(3) FV cap downstream.
    # The holdings-file `Market Value ($)` column is intentionally not consumed.
    mv_by_id = load_sec_mv_from_sheet2()
    sec["MV_USD"] = sec["ID"].map(mv_by_id).astype(float)
    missing_mv = sec["MV_USD"].isna()
    if missing_mv.any():
        bad = sec.loc[missing_mv, "ID"].tolist()
        raise ValueError(
            f"Sheet 2 (Portfolio_MV_Decomposed) missing rows for sec position IDs: "
            f"{bad}. Re-run the sensitivity engine to refresh "
            f"{os.path.normpath(SENSITIVITIES_PATH)}."
        )

    return sec, cfg


# ── VALIDATION ────────────────────────────────────────────────────────────────

def validate_sec_frame(sec: pd.DataFrame) -> list[str]:
    """Returns list of integrity issue messages; empty list means clean."""
    issues = []

    # Required columns
    required = [
        "ID", "Pool ID", "Attachment Pt (%)", "Detachment Pt (%)", "Rating",
        "signed_notional", "is_senior_derived", "tranche_key",
        "M_T_for_drc", "W", "K_SA", "pool_type",
    ]
    missing = [c for c in required if c not in sec.columns]
    if missing:
        issues.append(f"Missing required columns: {missing}")
        return issues

    # Attachment < Detachment
    bad_ad = sec[sec["Attachment Pt (%)"] >= sec["Detachment Pt (%)"]]
    if not bad_ad.empty:
        issues.append(f"Attachment >= Detachment for IDs: {bad_ad['ID'].tolist()}")

    # A and D in [0, 100]
    bad_range = sec[
        (sec["Attachment Pt (%)"] < 0) | (sec["Attachment Pt (%)"] > 100)
        | (sec["Detachment Pt (%)"] < 0) | (sec["Detachment Pt (%)"] > 100)
    ]
    if not bad_range.empty:
        issues.append(f"A or D out of [0,100] for IDs: {bad_range['ID'].tolist()}")

    # Notional non-zero
    bad_zero = sec[sec["signed_notional"] == 0]
    if not bad_zero.empty:
        issues.append(f"Zero signed_notional for IDs: {bad_zero['ID'].tolist()}")

    # Per-tranche RW invariant: positions sharing tranche_key must share
    # all RW-determining attributes (this is by construction since the key
    # encodes them, but we cross-check Rating for safety).
    by_key = sec.groupby("tranche_key")["Rating"].nunique()
    bad_keys = by_key[by_key > 1]
    if not bad_keys.empty:
        issues.append(
            f"tranche_key collision: same key, different Rating for keys {bad_keys.index.tolist()}"
        )

    # M_T_for_drc must be exactly 1
    if not (sec["M_T_for_drc"] == 1.0).all():
        issues.append("M_T_for_drc != 1.0 for some rows (MAR22.34(1) violation)")

    # NPL flag consistency
    inconsistent_npl = sec[
        (sec["is_npl_derived"]) & (sec["W"] < 0.90)
    ]
    if not inconsistent_npl.empty:
        issues.append(f"is_npl_derived=True but W<0.90 for IDs: {inconsistent_npl['ID'].tolist()}")

    return issues


# ── SUMMARY / SNAPSHOT ────────────────────────────────────────────────────────

def summarise(sec: pd.DataFrame) -> str:
    lines = []
    lines.append(f"Loaded {len(sec)} securitisation positions from "
                 f"{sec['Pool ID'].nunique()} deals.")
    lines.append("")
    lines.append("Pool type breakdown:")
    lines.append(sec["pool_type"].value_counts().to_string())
    lines.append("")
    lines.append("Approach hint (NOT yet routed - for engine in step 4):")
    lines.append(f"  rated, investor -> SEC-ERBA: "
                 f"{(sec['bank_role_in_deal'] == 'investor').sum()}")
    lines.append(f"  NPL deals (W >= 0.90): "
                 f"{sec['is_npl_derived'].sum()}")
    lines.append(f"  protected positions (CRM): "
                 f"{sec['protection_present'].astype(bool).sum()}")
    lines.append(f"  DD failures: "
                 f"{(sec['dd_status'] != 'pass').sum()}")
    lines.append("")
    lines.append("Senior derivation audit (label vs derived):")
    sec_lbl = sec["Seniority (DRC)"].astype(str).str.lower() == "senior"
    label_vs_derived = pd.crosstab(sec_lbl, sec["is_senior_derived"],
                                    rownames=["label='Senior'"],
                                    colnames=["is_senior_derived"])
    lines.append(label_vs_derived.to_string())
    flipped = sec[sec_lbl & ~sec["is_senior_derived"]]
    if not flipped.empty:
        lines.append("")
        lines.append("Senior LABEL but NOT senior_derived (regulatory correction applied):")
        lines.append(flipped[["ID", "Pool ID", "Security",
                              "Attachment Pt (%)", "Detachment Pt (%)"]].to_string(index=False))
    lines.append("")
    lines.append("Tranche-netting groups (perfect hedges should net to 0):")
    netting = sec.groupby("tranche_key").agg(
        n_positions=("ID", "count"),
        net_notional=("signed_notional", "sum"),
        gross_notional=("signed_notional", lambda s: s.abs().sum()),
    )
    netting["net_pct_of_gross"] = (
        netting["net_notional"].abs() / netting["gross_notional"] * 100
    ).round(1)
    multi = netting[netting["n_positions"] > 1]
    if not multi.empty:
        lines.append(multi.to_string())
    else:
        lines.append("  (no multi-position tranches in current portfolio)")
    return "\n".join(lines)


def write_snapshot(sec: pd.DataFrame, path: str = OUTPUT_SNAPSHOT) -> None:
    """Phase 1 — Sec position load + Deal_Master / Position_Overrides join.

    `output_df` is the per-row sec frame (one row per portfolio position) with
    derived fields ready for phase 2 (classification + hierarchy routing).
    """
    import sys
    here = os.path.dirname(__file__)
    parent = os.path.abspath(os.path.join(here, ".."))
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from phase_snapshot import write_phase_snapshot

    netting = sec.groupby("tranche_key").agg(
        n_positions=("ID", "count"),
        net_notional=("signed_notional", "sum"),
        gross_notional=("signed_notional", lambda s: s.abs().sum()),
        example_pool_id=("Pool ID", "first"),
        example_rating=("Rating", "first"),
        example_is_senior_derived=("is_senior_derived", "first"),
    ).reset_index()
    by_pool = sec["Pool ID"].value_counts().rename_axis("Pool ID").reset_index(name="n")
    senior_flips = sec[sec.get("seniority_label_disagrees_with_derived", False) == True][
        ["ID", "Pool ID", "Security", "Seniority (DRC)", "is_senior_derived"]
    ] if "seniority_label_disagrees_with_derived" in sec.columns else pd.DataFrame()

    write_phase_snapshot(
        path,
        phase_num=1,
        phase_name="Sec position load + Deal_Master join",
        mar_ref="MAR22.34 inherits CRE40 conventions; CRE40.18 senior derivation; M_T=1 forced",
        source_module="drc_securitisation/sec_loader.py",
        input_df=None,
        output_df=filter_output_columns(sec),
        input_files=[
            os.path.normpath(PORTFOLIO_PATH),
            os.path.normpath(CONFIG_PATH),
        ],
        audit={
            "tranche_key_netting_view": netting,
            "positions_by_pool": by_pool,
            "label_vs_derived_senior_disagreements": senior_flips,
        },
        reconciliation=[
            ("rows_in (raw sec rows from Combined Holdings)", int(len(sec))),
            ("rows_out (enriched positions)", int(len(sec))),
            ("unique_tranche_keys", int(sec["tranche_key"].nunique())),
            ("unique_pool_ids", int(sec["Pool ID"].nunique())),
            ("sum_signed_notional", float(sec["signed_notional"].sum())),
            ("sum_abs_notional (gross)",
                float(sec["signed_notional"].abs().sum())),
        ],
        notes="Loader has NO RW logic. Tranche key = (Pool ID, A, D, "
              "Position Sub-Type) per engine contract. RW-determining "
              "attributes (Rating, is_senior_derived, approach, dd_status, "
              "Deal_Master fields) are asserted consistent within each "
              "tranche-key group in sec_engine.assert_tranche_key_consistency().",
    )


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

def main() -> None:
    sec, cfg = build_sec_loader_frame()
    issues = validate_sec_frame(sec)
    print(summarise(sec))
    print()
    if issues:
        print("VALIDATION ISSUES:")
        for i in issues:
            print(f"  - {i}")
    else:
        print("Validation: clean.")
    write_snapshot(sec)
    print(f"\nSnapshot written: {OUTPUT_SNAPSHOT}")


if __name__ == "__main__":
    main()
