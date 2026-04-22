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
    calc_equity_delta,
    calc_spx_option_delta, calc_spx_option_vega, calc_spx_option_curvature,
    calc_securitisation_csr_delta, calc_securitisation_girr_delta,
    calc_commodity_trs_delta, calc_commodity_trs_girr_delta,
    calc_xccy_usd_leg_girr, calc_xccy_gbp_leg_girr,
    calc_il_gilt_sensitivities, calc_fx_delta,
    years_between
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
        'tranche_type': str(row.get('Tranche Type', '')) if pd.notna(row.get('Tranche Type')) else '',
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


def classify_instrument(inst: dict) -> str:
    """Determine instrument type from portfolio fields."""
    rc = inst['risk_class'].upper()
    sec = inst['security'].upper()
    issue = inst['issue_type'].upper() if inst['issue_type'] else ''
    asset = inst['asset_class'].upper() if inst['asset_class'] else ''
    
    # Government bonds
    if 'TBILL' in sec or 'TREASURY' in sec:
        return 'GOV_BOND'
    
    # Callable corporate bonds
    if pd.notna(inst['call_date']) and inst['call_date'] is not None:
        try:
            if not pd.isna(inst['call_date']):
                return 'CALLABLE_BOND'
        except:
            pass
    
    # Plain corporate bonds
    if 'CSR_NONSEC' in rc and ('BOND' in sec or 'GIRR' in rc):
        return 'CORP_BOND'
    
    # Equities
    if 'EQUITY' in rc and 'SPX' not in sec and 'CALL' not in sec.upper():
        return 'EQUITY_SPOT'
    
    # SPX options
    if 'SPX' in sec or ('EQUITY' in rc and ('CALL' in sec or 'PUT' in sec)):
        return 'SPX_OPTION'
    
    # Securitisations
    if 'CSR_SEC' in rc:
        return 'SECURITISATION'
    
    # Commodity TRS receive legs
    if 'COMM' in rc and 'BCOMTR' in rc:
        return 'COMMODITY_TRS_RECEIVE'
    
    # SOFR pay legs
    if 'GIRR:USD' in rc and inst['underlying'] and 'SOFR' in inst['underlying'].upper():
        return 'COMMODITY_TRS_SOFR'
    
    # XCcy swap USD leg
    if 'GIRR:USD' in rc and 'XCCY' in sec.upper():
        return 'XCCY_USD_LEG'
    
    # XCcy swap GBP leg
    if 'GIRR:GBP' in rc and ('XCCY' in sec.upper() or 'XCCY_BASIS' in rc):
        return 'XCCY_GBP_LEG'
    
    # IL Gilts
    if 'INFLATION' in rc or 'IL GILT' in sec or 'UKTI' in sec:
        return 'IL_GILT'
    
    # Fallback
    if 'GIRR' in rc:
        return 'CORP_BOND'
    
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
            sensitivities.update(girr)
            sensitivities.update(csr)
            sensitivities.update(vega)
            sensitivities.update(curv)
            risk_flags['GIRR_Delta'] = True
            risk_flags['GIRR_Vega'] = True
            risk_flags['GIRR_Curvature'] = True
            risk_flags['CSR_NonSec_Delta'] = True
            sens_definition = "Callable: tree-based GIRR PV01 + CSR CS01 + Vega(HW) + Curvature(±1.7%) [MAR21.19,20,25,5]"
        
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
    
    return df
