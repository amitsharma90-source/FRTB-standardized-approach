"""
FRTB SA — Module 3: Sensitivity Orchestrator
Routes each instrument to the correct sensitivity calculator,
collects all results, and outputs a single Excel file.
"""
import pandas as pd
import numpy as np
from data_loader import (
    load_portfolio, load_market_data, load_bcomtr_weights,
    load_config
)
from sensitivity_calc import (
    calc_bond_girr_delta, calc_bond_csr_delta,
    calc_callable_bond_girr_delta, calc_callable_bond_csr_delta,
    calc_callable_bond_girr_vega, calc_callable_bond_girr_curvature,
    calc_callable_bond_csr_curvature,
    calc_equity_delta,
    calc_spx_option_delta, calc_spx_option_vega, calc_spx_option_curvature,
    calc_securitisation_csr_delta, calc_securitisation_girr_delta,
    calc_commodity_trs_delta, calc_commodity_trs_girr_delta,
    calc_xccy_usd_leg_girr, calc_xccy_gbp_leg_girr,
    calc_il_gilt_sensitivities, calc_fx_delta,
    years_between
)
from mv_decompose import (
    single_leg_row, decompose_callable_legs, decompose_spx_legs, sort_rows,
)

def build_instrument_dict(row, mkt) -> dict:
    """Convert a portfolio DataFrame row to a clean dict for calculators."""
    def _num(field, default=0):
        """NaN-safe numeric field reader. pandas NaN is truthy in Python so
        the common 'row.get(x, 0) or 0' pattern silently propagates NaN."""
        v = row.get(field, default)
        return default if pd.isna(v) else v

    inst = {
        'id': int(row.name) if isinstance(row.name, (int, float)) else row.name,
        'security': str(row.get('Security', '')),
        'notional': _num('Quantity/Notional', 0),
        'market_value': _num('Market Value ($)', 0),
        'coupon': _num('Coupon/Rate (%)', 0),
        'maturity': row.get('Maturity', None),
        'call_date': row.get('Call Date', None),
        'currency': str(row.get('Currency', 'USD')),
        'rating': str(row.get('Rating', '')) if pd.notna(row.get('Rating')) else '',
        'strike': _num('Strike Price', 0),
        'spread_bps': _num('Spread (bps)', 0),
        'long_short': str(row.get('Long/Short', 'Long')),
        'risk_class': str(row.get('FRTB_Risk_Class', '')),
        'bucket': str(row.get('FRTB_Bucket', '')),
        'ticker': str(row.get('Ticker/Index', '')) if pd.notna(row.get('Ticker/Index')) else '',
        'position_type': str(row.get('Position Type', '')),
        'issue_type': str(row.get('Issue Type', '')),
        'asset_class': str(row.get('Asset Class', '')),
        'risk_measures': str(row.get('FRTB Risk Measures', '')),
        'underlying': str(row.get('Underlying', '')) if pd.notna(row.get('Underlying')) else '',
        'pool_id': str(row.get('Pool ID', '')) if pd.notna(row.get('Pool ID')) else '',
        'pool_type': str(row.get('Underlying Pool Type', '')) if pd.notna(row.get('Underlying Pool Type')) else 'RMBS',
        'attach_pt': _num('Attachment Pt (%)', 0),
        'detach_pt': _num('Detachment Pt (%)', 100),
        'freq': 2,  # Default semi-annual
        'issue_date': row.get('Issue Date', None),
    }
    
    # Parse payment frequency
    freq_str = str(row.get('Payment Frequency', ''))
    if 'Quarter' in freq_str: inst['freq'] = 4
    elif 'Month' in freq_str: inst['freq'] = 12
    elif 'Annual' in freq_str and 'Semi' not in freq_str: inst['freq'] = 1
    
    # Extract bucket numbers from bucket string
    bucket_str = str(row.get('FRTB_Bucket', ''))
    if 'CSR:' in bucket_str:
        try:
            inst['csr_bucket'] = int(bucket_str.split('CSR:')[1].strip().split()[0].split('|')[0])
        except: inst['csr_bucket'] = 1
    if 'CSRSec:' in bucket_str:
        try:
            inst['csr_sec_bucket'] = int(bucket_str.split('CSRSec:')[1].strip().split()[0])
        except: inst['csr_sec_bucket'] = 1
    if 'EQ:' in bucket_str:
        try:
            inst['eq_bucket'] = int(bucket_str.split('EQ:')[1].strip().split()[0])
        except: inst['eq_bucket'] = 8
    
    return inst


# Equity index keywords for disambiguating index options from single-name options.
# Mirrors drc_nonsecuritisation/nonsec_decompose._INDEX_KEYWORDS so the two phases
# detect index-vs-single-name on the same rule.
_EQUITY_INDEX_KEYWORDS = ("index", "spx", "s&p", "nasdaq", "russell", "dow")


def classify_instrument(inst: dict) -> str:
    """Route a portfolio row to a sensitivity calculator using the canonical
    (Position Type, Issue Type) schema. Disambiguators are limited to other
    structured fields (Currency, Underlying); Security-name substring matches
    are not used."""
    pt = (inst['position_type'] or '').strip()
    it = (inst['issue_type'] or '').strip()
    underlying = (inst['underlying'] or '').lower()
    ccy = (inst['currency'] or '').strip().upper()

    if pt == 'Bond':
        if it == 'Gov':
            return 'GOV_BOND'
        if it == 'IL Gilt':
            return 'IL_GILT'
        if it == 'Callable Bond':
            return 'CALLABLE_BOND'
        if it == 'Corp':
            return 'CORP_BOND'

    if pt == 'Equity':
        if it == 'Corp':
            return 'EQUITY_SPOT'
        if it == 'Call Option':
            if any(k in underlying for k in _EQUITY_INDEX_KEYWORDS):
                return 'SPX_OPTION'
            return 'EQUITY_SINGLE_NAME_OPTION'

    if pt == 'Securitisation':
        return 'SECURITISATION'

    if pt == 'TRS':
        if 'sofr' in underlying:
            return 'COMMODITY_TRS_SOFR'
        if 'bcom' in underlying or 'commodity' in underlying:
            return 'COMMODITY_TRS_RECEIVE'

    if pt == 'FXSwap':
        if ccy == 'USD':
            return 'XCCY_USD_LEG'
        if ccy == 'GBP':
            return 'XCCY_GBP_LEG'

    return 'UNKNOWN'


def compute_all_sensitivities(portfolio_path: str, market_data_path: str,
                               config_path: str) -> pd.DataFrame:
    """Main entry point. Computes sensitivities for all instruments.
    Returns DataFrame: one row per instrument, columns for each risk factor.
    """
    # Load data
    cfg = load_config(config_path)
    port = load_portfolio(portfolio_path)
    mkt = load_market_data(market_data_path, cfg)
    bcomtr = load_bcomtr_weights(market_data_path)
    
    print(f"Loaded {len(port)} instruments, val date {mkt['val_date'].date()}")
    print(f"BCOMTR: {len(bcomtr)} constituents")
    print(f"USD curve tenors: {sorted(mkt['usd_rates'].keys())}")
    
    # Process each instrument
    all_results = []
    mv_rows = []
    parent_totals: dict[int, float] = {}

    for idx, row in port.iterrows():
        inst = build_instrument_dict(row, mkt)
        inst_type = classify_instrument(inst)
        
        sensitivities = {}
        risk_flags = {
            'GIRR_Delta': False, 'GIRR_Vega': False, 'GIRR_Curvature': False,
            'CSR_NonSec_Delta': False, 'CSR_Sec_Delta': False,
            'EQ_Delta': False, 'EQ_Vega': False, 'EQ_Curvature': False,
            'COMM_Delta': False, 'FX_Delta': False,
            'GIRR_Inflation': False, 'GIRR_XCcy_Basis': False,
        }
        sens_definition = ""
        
        if inst_type == 'GOV_BOND':
            girr = calc_bond_girr_delta(inst, mkt, cfg)
            sensitivities.update(girr)
            risk_flags['GIRR_Delta'] = True
            sens_definition = "PV01: bump risk-free rate +1bp per tenor [MAR21.19]"
        
        elif inst_type == 'CORP_BOND':
            girr = calc_bond_girr_delta(inst, mkt, cfg)
            csr = calc_bond_csr_delta(inst, mkt, cfg)
            sensitivities.update(girr)
            sensitivities.update(csr)
            risk_flags['GIRR_Delta'] = True
            risk_flags['CSR_NonSec_Delta'] = True
            sens_definition = "GIRR PV01 [MAR21.19] + CSR CS01 [MAR21.20]"
        
        elif inst_type == 'CALLABLE_BOND':
            girr = calc_callable_bond_girr_delta(inst, mkt, cfg)
            csr = calc_callable_bond_csr_delta(inst, mkt, cfg)
            vega = calc_callable_bond_girr_vega(inst, mkt, cfg)
            curv = calc_callable_bond_girr_curvature(inst, mkt, cfg)
            csr_curv = calc_callable_bond_csr_curvature(inst, mkt, cfg)
            sensitivities.update(girr)
            sensitivities.update(csr)
            sensitivities.update(vega)
            sensitivities.update(curv)
            sensitivities.update(csr_curv)
            risk_flags['GIRR_Delta'] = True
            risk_flags['GIRR_Vega'] = True
            risk_flags['GIRR_Curvature'] = True
            risk_flags['CSR_NonSec_Delta'] = True
            risk_flags['CSR_Curvature'] = True
            sens_definition = "Callable: GIRR PV01 + CSR CS01 + Vega(HW) + GIRR Curv(±1.7%) + CSR Curv(parallel spread shift) [MAR21.19,20,25,5,97-99]"
        
        elif inst_type == 'EQUITY_SPOT':
            eq = calc_equity_delta(inst, mkt, cfg)
            sensitivities.update(eq)
            risk_flags['EQ_Delta'] = True
            sens_definition = "Equity delta = MV (1% relative bump) [MAR21.21]"
        
        elif inst_type == 'SPX_OPTION':
            eq_d = calc_spx_option_delta(inst, mkt, cfg)
            eq_v = calc_spx_option_vega(inst, mkt, cfg)
            eq_c = calc_spx_option_curvature(inst, mkt, cfg)
            sensitivities.update(eq_d)
            sensitivities.update(eq_v)
            sensitivities.update(eq_c)
            risk_flags['EQ_Delta'] = True
            risk_flags['EQ_Vega'] = True
            risk_flags['EQ_Curvature'] = True
            sens_definition = "BSM delta/vega/curvature [MAR21.21,25,5] RW=15%"
        
        elif inst_type == 'SECURITISATION':
            csr_sec = calc_securitisation_csr_delta(inst, mkt, cfg)
            girr_sec = calc_securitisation_girr_delta(inst, mkt, cfg)
            sensitivities.update(csr_sec)
            sensitivities.update(girr_sec)
            risk_flags['CSR_Sec_Delta'] = True
            if girr_sec:
                risk_flags['GIRR_Delta'] = True
            sens_definition = "Tranche CS01 + GIRR PV01 (SOFR bumped, spread fixed) [MAR21.10(1),19,20]"
        
        elif inst_type == 'COMMODITY_TRS_RECEIVE':
            comm = calc_commodity_trs_delta(inst, mkt, cfg, bcomtr)
            sensitivities.update(comm)
            risk_flags['COMM_Delta'] = True
            sens_definition = "Look-through 25 constituents, delta = weight × notional [MAR21.23,34]"
        
        elif inst_type == 'COMMODITY_TRS_SOFR':
            girr = calc_commodity_trs_girr_delta(inst, mkt, cfg)
            sensitivities.update(girr)
            risk_flags['GIRR_Delta'] = True
            sens_definition = "SOFR floating leg PV01 at 0.25Y tenor [MAR21.19]"
        
        elif inst_type == 'XCCY_USD_LEG':
            girr = calc_xccy_usd_leg_girr(inst, mkt, cfg)
            sensitivities.update(girr)
            risk_flags['GIRR_Delta'] = True
            sens_definition = "USD SOFR curve PV01 across tenors [MAR21.19]"
        
        elif inst_type == 'XCCY_GBP_LEG':
            all_sens = calc_xccy_gbp_leg_girr(inst, mkt, cfg)
            sensitivities.update(all_sens)
            risk_flags['GIRR_Delta'] = True
            risk_flags['FX_Delta'] = True
            risk_flags['GIRR_XCcy_Basis'] = True
            sens_definition = "GBP SONIA PV01 + XCcy basis(flat,+8bps) + FX delta [MAR21.19,8(3),24]"
        
        elif inst_type == 'IL_GILT':
            all_sens = calc_il_gilt_sensitivities(inst, mkt, cfg)
            sensitivities.update(all_sens)
            risk_flags['GIRR_Delta'] = True
            risk_flags['GIRR_Inflation'] = True
            risk_flags['FX_Delta'] = True
            sens_definition = "Real rate PV01 + inflation(flat) + FX delta [MAR21.19,8(2),24]"
        
        else:
            sens_definition = f"UNKNOWN TYPE: {inst_type}"
        
        # ── Sheet 2 MV emit (Portfolio_MV_Decomposed) ─────────────────────────
        # Each calculator that ran has stashed `_computed_mv` (signed model NPV
        # in USD) and `_pricing_model` on `inst`. For decomposed instruments
        # (callable bonds, equity index options) we expand to multi-leg rows
        # here using the prices already computed during sensitivity calc.
        if inst_type == 'CALLABLE_BOND':
            legs, parent_total = decompose_callable_legs(row, inst, mkt)
            mv_rows.extend(legs)
            parent_totals[inst['id']] = parent_total
        elif inst_type == 'SPX_OPTION':
            legs, parent_total = decompose_spx_legs(row, inst)
            mv_rows.extend(legs)
            parent_totals[inst['id']] = parent_total
        elif inst_type != 'UNKNOWN' and '_computed_mv' in inst:
            mv_signed = float(inst['_computed_mv'])
            mv_rows.append(single_leg_row(
                row, inst, mv_signed,
                inst.get('_pricing_model', 'unknown')))
            parent_totals[inst['id']] = mv_signed

        # Build result row
        result = {
            'ID': inst['id'],
            'Security': inst['security'],
            'Instrument_Type': inst_type,
            'Sensitivity_Definition': sens_definition,
        }
        result.update(risk_flags)
        result.update(sensitivities)
        all_results.append(result)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    df = df.set_index('ID')

    # Fill NaN sensitivities with 0
    sens_cols = [c for c in df.columns if c not in
                 ['Security', 'Instrument_Type', 'Sensitivity_Definition']
                 and c not in risk_flags.keys()]
    df[sens_cols] = df[sens_cols].fillna(0)

    # Sheet 2 rows in deterministic order
    mv_rows = sort_rows(mv_rows)

    return df, mv_rows, parent_totals
