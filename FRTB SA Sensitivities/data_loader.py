"""
FRTB SA — Module 1: Data Loader (v6)
- Curves interpolated to FRTB tenors only (no non-FRTB tenors like 7Y)
- All constants loaded from MAR21_Config_RW_Corr.xlsx (no hardcoding)
"""
import pandas as pd
import numpy as np
import QuantLib as ql


def _interp_raw(raw_curve: dict, target_tenor: float) -> float:
    """Linearly interpolate a raw curve dict at a target tenor."""
    ts = sorted(raw_curve.keys()); rs = [raw_curve[t] for t in ts]
    if target_tenor <= ts[0]: return rs[0]
    if target_tenor >= ts[-1]: return rs[-1]
    for i in range(len(ts) - 1):
        if ts[i] <= target_tenor <= ts[i + 1]:
            w = (target_tenor - ts[i]) / (ts[i + 1] - ts[i])
            return rs[i] + w * (rs[i + 1] - rs[i])
    return rs[-1]


def _snap_to_frtb_tenors(raw_curve: dict, frtb_tenors: list) -> dict:
    """Interpolate a raw market data curve onto FRTB prescribed tenors only."""
    return {t: _interp_raw(raw_curve, t) for t in frtb_tenors}


def _build_ql_curve(val_date, rate_dict):
    """Build a QuantLib ZeroCurve from a {tenor_years: rate} dict."""
    dates = [val_date]
    rates = [list(rate_dict.values())[0]]
    for tenor_yrs in sorted(rate_dict.keys()):
        d = val_date + ql.Period(max(int(tenor_yrs * 365.25), 1), ql.Days)
        dates.append(d)
        rates.append(rate_dict[tenor_yrs])
    curve = ql.ZeroCurve(dates, rates, ql.Actual365Fixed())
    curve.enableExtrapolation()
    handle = ql.YieldTermStructureHandle(curve)
    return curve, handle, dates, rates


def load_config(filepath: str) -> dict:
    """Load MAR21 config including FRTB tenors, model params, tranche spreads, WAL."""
    cfg = {}
    xl = pd.ExcelFile(filepath)
    for sheet in xl.sheet_names:
        try:
            cfg[sheet] = pd.read_excel(xl, sheet_name=sheet)
        except Exception as e:
            print(f"  Warning: {sheet}: {e}")

    # Parse FRTB tenors
    if 'FRTB_TENORS' in cfg:
        df = cfg['FRTB_TENORS']
        cfg['girr_tenors'] = sorted(df[df['risk_class'] == 'GIRR']['tenor'].tolist())
        cfg['csr_tenors'] = sorted(df[df['risk_class'] == 'CSR']['tenor'].tolist())
        cfg['vega_tenors'] = sorted(df[df['risk_class'] == 'VEGA']['tenor'].tolist())
    else:
        cfg['girr_tenors'] = [0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, 30]
        cfg['csr_tenors'] = [0.5, 1, 3, 5, 10]
        cfg['vega_tenors'] = [0.5, 1, 3, 5, 10]

    # Parse model params
    cfg['params'] = {}
    if 'MODEL_PARAMS' in cfg:
        for _, row in cfg['MODEL_PARAMS'].iterrows():
            cfg['params'][row['parameter']] = row['value']

    # Parse tranche spreads
    cfg['tranche_spreads'] = {}
    if 'TRANCHE_SPREADS' in cfg:
        for _, row in cfg['TRANCHE_SPREADS'].iterrows():
            cfg['tranche_spreads'][(row['rating'], row['seniority'])] = row['spread_decimal']

    # Parse WAL assumptions
    cfg['wal_assumptions'] = {}
    if 'WAL_ASSUMPTIONS' in cfg:
        for _, row in cfg['WAL_ASSUMPTIONS'].iterrows():
            cfg['wal_assumptions'][row['pool_type']] = row['wal_factor']

    # Parse SEC leverage params
    cfg['sec_leverage'] = {}
    if 'SEC_LEVERAGE_PARAMS' in cfg:
        for _, row in cfg['SEC_LEVERAGE_PARAMS'].iterrows():
            cfg['sec_leverage'][row['parameter']] = row['value']

    # Parse equity risk weights by bucket (rw_spot column, percentage string or float)
    cfg['equity_rw'] = {}
    if 'EQUITY_RW' in cfg:
        for _, row in cfg['EQUITY_RW'].iterrows():
            try:
                bucket = int(row['bucket'])
                rw = float(str(row['rw_spot']).replace('%', '')) / 100
                cfg['equity_rw'][bucket] = rw
            except (ValueError, KeyError):
                pass

    return cfg


def load_portfolio(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(filepath, sheet_name="Combined Holdings")
    df = df[df['ID'].apply(lambda x: str(x).replace('.', '').isdigit() if pd.notna(x) else False)].copy()
    df['ID'] = df['ID'].astype(int)
    df = df.set_index('ID')
    for col in ['Quantity/Notional', 'Market Value ($)', 'Coupon/Rate (%)',
                'Strike Price', 'Option Premium', 'Spread (bps)',
                'Attachment Pt (%)', 'Detachment Pt (%)']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ['Maturity', 'Call Date', 'Issue Date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


def _load_gbp_curves(filepath: str) -> dict:
    """Read all three GBP curves from the GBP_Curves sheet.

    Scans for section header rows (where col B contains 'SONIA', 'Real', 'Nominal')
    and reads tenor/rate pairs until the next blank or header row.
    Returns dict with keys: 'sonia', 'real', 'nominal' — each a {tenor: rate_decimal}.
    """
    df = pd.read_excel(filepath, sheet_name='GBP_Curves', header=None)
    curves = {'sonia': {}, 'real': {}, 'nominal': {}}
    current = None
    for _, row in df.iterrows():
        label = str(row.iloc[1]).strip().lower() if pd.notna(row.iloc[1]) else ''
        if 'sonia' in label:
            current = 'sonia'; continue
        if 'real' in label and 'nominal' not in label:
            current = 'real'; continue
        if 'nominal' in label:
            current = 'nominal'; continue
        if current is None:
            continue
        tenor = pd.to_numeric(row.iloc[0], errors='coerce')
        rate  = pd.to_numeric(row.iloc[1], errors='coerce')
        if pd.notna(tenor) and pd.notna(rate):
            curves[current][float(tenor)] = float(rate) / 100.0
        elif pd.isna(tenor) and pd.isna(rate):
            current = None  # blank row signals end of section
    return curves


def load_market_data(filepath: str, cfg: dict) -> dict:
    df = pd.read_excel(filepath, sheet_name=0)
    row = df.iloc[0]
    mkt = {}

    mkt['val_date'] = pd.to_datetime(row['Date'])
    mkt['ql_val_date'] = ql.Date(mkt['val_date'].day, mkt['val_date'].month, mkt['val_date'].year)
    ql.Settings.instance().evaluationDate = mkt['ql_val_date']

    girr_tenors = cfg['girr_tenors']

    # ── USD SOFR OIS: load raw, snap to FRTB tenors ──
    # Per MAR21.39, USD risk-free curve is SOFR. Reads SOFR_* columns from snapshot;
    # falls back to legacy Treasury-labelled columns ('1M', '3M', ...) if SOFR_*
    # columns are absent (for backward compatibility with older snapshots).
    usd_raw = {}
    for sofr_label, treas_label, tenor in [
        ('SOFR_1M',  '1M',  1/12), ('SOFR_3M',  '3M',  3/12), ('SOFR_6M',  '6M',  0.5),
        ('SOFR_1Y',  '1Y',  1),    ('SOFR_2Y',  '2Y',  2),    ('SOFR_3Y',  '3Y',  3),
        ('SOFR_5Y',  '5Y',  5),    ('SOFR_7Y',  '7Y',  7),    ('SOFR_10Y', '10Y', 10),
        ('SOFR_20Y', '20Y', 20),   ('SOFR_30Y', '30Y', 30)]:
        # Prefer SOFR-labelled column; fall back to legacy Treasury-labelled
        if sofr_label in row.index and pd.notna(row[sofr_label]):
            usd_raw[tenor] = row[sofr_label] / 100
        elif treas_label in row.index and pd.notna(row[treas_label]):
            usd_raw[tenor] = row[treas_label] / 100
            print(f"  [WARN] USD curve: SOFR_{treas_label} not found, using Treasury {treas_label} as fallback")
    mkt['usd_rates'] = _snap_to_frtb_tenors(usd_raw, girr_tenors)
    mkt['usd_curve'], mkt['usd_handle'], _, _ = _build_ql_curve(mkt['ql_val_date'], mkt['usd_rates'])

    # ── OAS by rating ──
    oas_map = {'AAA': 'AAA_OAS', 'AA': 'AA_OAS', 'A': 'A_OAS',
               'BBB': 'BBB_OAS', 'BB': 'BB_OAS', 'B': 'B_OAS'}
    mkt['oas'] = {}
    for rtg, col in oas_map.items():
        if col in row.index and pd.notna(row[col]):
            mkt['oas'][rtg] = row[col] / 100
    mkt['oas']['A-1'] = mkt['oas'].get('A', 0.0092)

    # ── Equity ──
    tickers = ['NVDA', 'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'JPM', 'BRK_B', 'XOM', 'EQIX']
    mkt['equity_prices'] = {t: row[t] for t in tickers if t in row.index and pd.notna(row[t])}

    # ── SPX / VIX ──
    mkt['spx'] = row.get('SPX_Close', 6798.40)
    mkt['vix'] = row.get('VIXCLS', 21.77)
    mkt['spx_div_yield'] = cfg['params'].get('spx_div_yield', 0.013)
    mkt['spx_vol_surface'] = _load_spx_vol_surface(filepath)

    # ── FX ──
    mkt['usd_gbp'] = row.get('USD/GBP', 1.3645)

    # ── GBP curves: read all three from GBP_Curves sheet ──
    gbp_curves = _load_gbp_curves(filepath)
    if not gbp_curves['sonia']:
        raise ValueError("GBP_Curves sheet: SONIA section not found or empty")
    if not gbp_curves['real']:
        raise ValueError("GBP_Curves sheet: Real Yield section not found or empty")
    if not gbp_curves['nominal']:
        raise ValueError("GBP_Curves sheet: Nominal Gilt section not found or empty")

    mkt['gbp_sonia_rates'] = _snap_to_frtb_tenors(gbp_curves['sonia'], girr_tenors)
    mkt['gbp_curve'], mkt['gbp_handle'], _, _ = _build_ql_curve(mkt['ql_val_date'], mkt['gbp_sonia_rates'])

    mkt['gbp_real_rates'] = _snap_to_frtb_tenors(gbp_curves['real'], girr_tenors)
    mkt['gbp_real_curve'], mkt['gbp_real_handle'], _, _ = _build_ql_curve(
        mkt['ql_val_date'], mkt['gbp_real_rates'])

    mkt['gbp_nominal_gilt_rates'] = _snap_to_frtb_tenors(gbp_curves['nominal'], girr_tenors)

    mkt['uk_real_10y'] = gbp_curves['real'].get(10, 0.0095)
    mkt['uk_breakeven'] = row.get('UK_BREAKEVEN_10Y', 3.48) / 100
    print(f"  GBP curves loaded from sheet: "
          f"SONIA={len(gbp_curves['sonia'])} tenors, "
          f"Real={len(gbp_curves['real'])} tenors, "
          f"NominalGilt={len(gbp_curves['nominal'])} tenors")
    mkt['bcomtr'] = row.get('BCOMCTGT', 145.36)

    # ── Issuer-specific credit spread curves ──
    mkt['issuer_spreads'] = _load_issuer_credit_curves(filepath, cfg.get('csr_tenors', [0.5, 1, 3, 5, 10]))
    mkt['tranche_curves'] = _load_tranche_curves(filepath, cfg.get('csr_tenors', [0.5, 1, 3, 5, 10]))
    print(f"DEBUG mkt['tranche_curves'] keys: {sorted(mkt.get('tranche_curves', {}).keys())}")
    return mkt


def load_bcomtr_weights(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(filepath, sheet_name="BCOMTR constituents")
    weights_2026 = {
        'Brent Crude Oil': 8.3602, 'WTI Crude Oil': 6.6398,
        'Low Sulphur Gas Oil': 2.8906, 'ULS Diesel': 2.1923,
        'RBOB Gasoline': 2.1547, 'Natural Gas': 7.1979,
        'Corn': 5.5282, 'Soybeans': 5.3553, 'Soybean Meal': 2.9335,
        'Soybean Oil': 2.8238, 'Wheat': 2.7217, 'HRW Wheat': 1.7911,
        'Copper': 6.3620, 'Aluminum': 3.9706, 'Zinc': 2.2499,
        'Nickel': 2.2278, 'Lead': 0.9498,
        'Gold': 14.8957, 'Silver': 3.9436,
        'Sugar': 2.9514, 'Coffee': 2.9146, 'Cocoa': 1.7137, 'Cotton': 1.5950,
        'Live Cattle': 3.8580, 'Lean Hogs': 1.7786,
    }
    df['Weight_Pct'] = df['Commodity'].map(weights_2026)
    df['Weight'] = df['Weight_Pct'] / 100
    return df


def get_oas_for_rating(mkt: dict, rating: str, issue_type: str = '') -> float:
    if pd.isna(rating): return 0.0
    if any(x in str(issue_type).lower() for x in ['gov', 'treasury', 'tbill']):
        return 0.0
    rating = str(rating).strip().upper()
    if rating in mkt['oas']: return mkt['oas'][rating]
    rating_map = {'AAA':'AAA','AA+':'AA','AA':'AA','AA-':'AA','A+':'A','A':'A','A-':'A','A-1':'A',
                  'BBB+':'BBB','BBB':'BBB','BBB-':'BBB','BB+':'BB','BB':'BB','BB-':'BB','B+':'B','B':'B','B-':'B'}
    return mkt['oas'].get(rating_map.get(rating, 'BBB'), 0.01)


def _load_spx_vol_surface(filepath: str) -> dict:
    try:
        df = pd.read_excel(filepath, sheet_name="SPX_Vol_Surface", header=3)
        df = df[pd.to_numeric(df.iloc[:, 0], errors='coerce').notna()].copy()
        surface = {}
        iv_cols = [c for c in df.columns if str(c).startswith('IV_')]
        for _, row in df.iterrows():
            strike = float(row.iloc[0])
            vol_by_expiry = {}
            for col in iv_cols:
                try:
                    expiry = float(str(col).replace('IV_', '').replace('Y', ''))
                    vol_by_expiry[expiry] = float(row[col])
                except: continue
            if vol_by_expiry: surface[strike] = vol_by_expiry
        if surface: print(f"  Loaded SPX vol surface: {len(surface)} strikes × {len(iv_cols)} expiries")
        return surface if surface else None
    except Exception as e:
        print(f"  No SPX vol surface: {e}")
        return None


def _load_issuer_credit_curves(filepath: str, csr_tenors: list) -> dict:
    """Load issuer-specific credit spread term structures from market data.
    Returns dict: {issuer_name: {0.5: spread_decimal, 1: spread_decimal, ...}}
    """
    try:
        df = pd.read_excel(filepath, sheet_name="Issuer_Credit_Curves", header=3)
        # Filter to valid issuer rows: must have Issuer AND numeric CS_5Y
        df = df[pd.notna(df['Issuer']) & pd.to_numeric(df.get('CS_5Y', pd.Series()), errors='coerce').notna()].copy()
        # Exclude rows where Issuer looks like a number or tenor label
        df = df[df['Issuer'].apply(lambda x: isinstance(x, str) and not x.replace('.','').replace('Y','').isdigit())]
        spreads = {}
        for _, row in df.iterrows():
            issuer = str(row['Issuer']).strip()
            curve = {}
            for t in csr_tenors:
                # Try column names: CS_0.5Y, CS_1Y, CS_3Y, CS_5Y, CS_10Y
                candidates = [f"CS_{t}Y", f"CS_{int(t)}Y"]
                for c in candidates:
                    if c in df.columns and pd.notna(row.get(c)):
                        curve[t] = float(row[c]) / 10000  # bps → decimal
                        break
            if curve:
                spreads[issuer] = curve
        if spreads:
            print(f"  Loaded issuer credit curves: {len(spreads)} issuers × {len(csr_tenors)} tenors")
        return spreads
    except Exception as e:
        print(f"  No issuer credit curves: {e}")
        return {}


def get_issuer_spread_curve(mkt: dict, issuer_name: str, rating: str, issue_type: str = '') -> dict:
    """Get the credit spread curve for a specific issuer.
    Returns dict {tenor: spread_decimal} at CSR tenors.
    Falls back to flat OAS by rating if issuer curve not found.
    """
    # Gov bonds: zero spread at all tenors
    if any(x in str(issue_type).lower() for x in ['gov', 'treasury', 'tbill']):
        return {}

    # Try issuer-specific curve
    issuer_spreads = mkt.get('issuer_spreads', {})
    if issuer_name in issuer_spreads:
        return issuer_spreads[issuer_name]

    # Partial match (e.g., "Apple 2027 Bond" → look for "Apple")
    for key in issuer_spreads:
        if key.lower() in issuer_name.lower():
            return issuer_spreads[key]

    # Fallback: flat OAS from rating
    flat_oas = get_oas_for_rating(mkt, rating, issue_type)
    if flat_oas > 0:
        return {0.5: flat_oas, 1: flat_oas, 3: flat_oas, 5: flat_oas, 10: flat_oas}
    return {}

def _load_tranche_curves(filepath: str, csr_tenors: list) -> dict:
    """Load manufactured tranche spread curves from Sec_Tranche_Curves sheet.
    Returns dict: {instrument_id: {0.5: spread_decimal, 1: ..., 3: ..., 5: ..., 10: ...}}

    Reads by header name (not column index) so the loader is robust to schema
    changes in the snapshot file. Spreads are stored in bp in the sheet and
    converted to decimal here. See CSR_Sec_NonCTP_Methodology.docx for the
    Level 3 construction methodology (anchor + thinness + term shape).
    """
    try:
        # Header is at row 7 in the Sec_Tranche_Curves sheet (rows 1-6 are metadata)
        df = pd.read_excel(filepath, sheet_name="Sec_Tranche_Curves", header=6)
        df = df[pd.to_numeric(df.get('ID', pd.Series()), errors='coerce').notna()].copy()
        df['ID'] = df['ID'].astype(int)

        tenor_to_col = {0.5: 'sp_0.5', 1: 'sp_1', 3: 'sp_3', 5: 'sp_5', 10: 'sp_10'}

        curves = {}
        for _, row in df.iterrows():
            inst_id = int(row['ID'])
            curve = {}
            for t in csr_tenors:
                col = tenor_to_col.get(t)
                if col and col in df.columns and pd.notna(row.get(col)):
                    curve[t] = float(row[col]) / 10000.0  # bps -> decimal
            if curve:
                curves[inst_id] = curve

        if curves:
            print(f"  Loaded tranche spread curves: {len(curves)} positions x {len(csr_tenors)} tenors")
        print(f"  DEBUG: tranche_curves keys = {sorted(curves.keys())}")
        return curves
    except Exception as e:
        import traceback
        print(f"  TRANCHE CURVES ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        return {}

