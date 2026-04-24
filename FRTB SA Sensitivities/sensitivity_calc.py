"""
FRTB SA — Module 2: Sensitivity Calculators (v7)
Changes from v6:
- calc_securitisation_csr_delta: 
- calc_il_gilt_sensitivities: real rate PV01 via QuantLib (unchanged, correct);
  inflation sensitivity now a separate parallel bump on the real curve (flat inflation
  assumption → single GIRR_GBP_INFLATION scalar); FX delta preserved
- calc_xccy_usd_leg_girr: replaced manual discount loop with QuantLib FixedRateBond
  pricing — consistent with bond modules, more accurate term structure
- calc_xccy_gbp_leg_girr: replaced manual discount loop with QuantLib FixedRateBond
  on GBP SONIA curve; XCcy basis modelled as flat spread bump on full curve;
  FX delta preserved
"""
import numpy as np
import QuantLib as ql
from scipy.stats import norm
from datetime import datetime
from sec_cashflows import project_tranche_bond
import sec_cashflows
print(f"[IMPORT CHECK] sec_cashflows from: {sec_cashflows.__file__}")
print(f"[IMPORT CHECK] sec_cashflows mtime: {__import__('os').path.getmtime(sec_cashflows.__file__)}")
import pandas as pd
import numpy as np


def years_between(d1, d2):
    if isinstance(d1, str): d1 = datetime.strptime(d1, '%Y-%m-%d')
    if isinstance(d2, str): d2 = datetime.strptime(d2, '%Y-%m-%d')
    return (d2 - d1).days / 365.25

def allocate_to_tenors(exact, tenors):
    if exact <= tenors[0]: return {tenors[0]: 1.0}
    if exact >= tenors[-1]: return {tenors[-1]: 1.0}
    for i in range(len(tenors) - 1):
        lo, hi = tenors[i], tenors[i + 1]
        if lo <= exact <= hi:
            w_lo = (hi - exact) / (hi - lo)
            w_hi = (exact - lo) / (hi - lo)
            r = {}
            if w_lo > 1e-6: r[lo] = w_lo
            if w_hi > 1e-6: r[hi] = w_hi
            return r
    return {tenors[-1]: 1.0}

def _interp_rate(curve, tenor):
    if not curve: return 0.04
    ts = sorted(curve.keys()); rs = [curve[t] for t in ts]
    if tenor <= ts[0]: return rs[0]
    if tenor >= ts[-1]: return rs[-1]
    for i in range(len(ts) - 1):
        if ts[i] <= tenor <= ts[i + 1]:
            w = (tenor - ts[i]) / (ts[i + 1] - ts[i])
            return rs[i] + w * (rs[i + 1] - rs[i])
    return rs[-1]

def _to_ql_date(d):
    if isinstance(d, ql.Date): return d
    if hasattr(d, 'day'): return ql.Date(d.day, d.month, d.year)
    return ql.DateParser.parseISO(str(d)[:10])


# ── CURVE BUMPING ──────────────────────────────────────────────────────────────

def _build_bumped_curve(val_date, rates_dict, bump_tenor, bump_size=0.0001):
    """Bump a single FRTB tenor by bump_size. Curve already on FRTB tenors only."""
    bumped = dict(rates_dict)
    if bump_tenor in bumped:
        bumped[bump_tenor] += bump_size
    from data_loader import _build_ql_curve
    _, handle, _, _ = _build_ql_curve(val_date, bumped)
    return handle

def _build_parallel_shifted_curve(val_date, rates_dict, shift):
    """Shift all tenors by a flat amount. Used for curvature and inflation bump."""
    shifted = {t: r + shift for t, r in rates_dict.items()}
    from data_loader import _build_ql_curve
    _, handle, _, _ = _build_ql_curve(val_date, shifted)
    return handle


# ── QUANTLIB BOND BUILDER ──────────────────────────────────────────────────────

def _build_ql_bond(notional, coupon, mat_date, freq=2, issue_date=None):
    face = 100.0
    scale = notional / face
    issue_ql = _to_ql_date(issue_date) if issue_date else ql.Date(1, 1, 2020)
    mat_ql = _to_ql_date(mat_date)
    period = {1: ql.Annual, 2: ql.Semiannual, 4: ql.Quarterly, 12: ql.Monthly}.get(freq, ql.Semiannual)
    cal = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    schedule = ql.Schedule(issue_ql, mat_ql, ql.Period(period), cal,
        ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Backward, False)
    bond = ql.FixedRateBond(0, face, schedule, [coupon], ql.Actual365Fixed())
    return bond, scale

def _price_bond_ql(bond, curve_handle, spread=0):
    if spread > 0:
        sq = ql.SimpleQuote(spread)
        sc = ql.ZeroSpreadedTermStructure(curve_handle, ql.QuoteHandle(sq))
        sc.enableExtrapolation()
        engine = ql.DiscountingBondEngine(ql.YieldTermStructureHandle(sc))
    else:
        engine = ql.DiscountingBondEngine(curve_handle)
    bond.setPricingEngine(engine)
    return bond.NPV()


# ── BOND GIRR DELTA [MAR21.19] ────────────────────────────────────────────────

def calc_bond_girr_delta(inst, mkt, cfg):
    notional = inst['notional']
    coupon = inst.get('coupon', 0) / 100
    mat_yrs = years_between(mkt['val_date'], inst['maturity'])
    ccy = inst.get('currency', 'USD')
    freq = inst.get('freq', 2)
    if mat_yrs <= 0: return {}

    from data_loader import get_oas_for_rating
    spread = get_oas_for_rating(mkt, inst.get('rating', ''), inst.get('issue_type', ''))
    rates_dict = mkt['usd_rates'] if ccy == 'USD' else mkt['gbp_sonia_rates']
    base_handle = mkt['usd_handle'] if ccy == 'USD' else mkt['gbp_handle']
    girr_tenors = cfg['girr_tenors']

    bond, scale = _build_ql_bond(notional, coupon, inst['maturity'], freq, inst.get('issue_date'))
    pv_base = _price_bond_ql(bond, base_handle, spread)
    inst['_computed_mv'] = pv_base * scale

    sens = {}
    for bt in girr_tenors:
        if bt > mat_yrs + 5: continue
        bumped_handle = _build_bumped_curve(mkt['ql_val_date'], rates_dict, bt)
        pv01 = (_price_bond_ql(bond, bumped_handle, spread) - pv_base) * scale / 0.0001
        if abs(pv01) > 0.01:
            sens[f"GIRR_{ccy}_{bt}Y"] = pv01
    return sens


# ── BOND CSR DELTA [MAR21.20] ─────────────────────────────────────────────────

def _build_ql_spread_curve(val_date, rf_handle, spread_curve_dict):
    """Build combined yield curve: risk-free + issuer credit spread term structure."""
    combined_dates = [val_date]
    combined_rates = [
        rf_handle.currentLink().zeroRate(0.01, ql.Continuous, ql.Annual).rate()
        + list(spread_curve_dict.values())[0]
    ]
    for t in sorted(spread_curve_dict.keys()):
        d = val_date + ql.Period(max(int(t * 365.25), 1), ql.Days)
        rf_rate = rf_handle.currentLink().zeroRate(t, ql.Continuous, ql.Annual).rate()
        combined_dates.append(d)
        combined_rates.append(rf_rate + spread_curve_dict[t])
    combined = ql.ZeroCurve(combined_dates, combined_rates, ql.Actual365Fixed())
    combined.enableExtrapolation()
    return ql.YieldTermStructureHandle(combined)

def calc_bond_csr_delta(inst, mkt, cfg):
    """CSR CS01: bump issuer credit spread at each CSR tenor by 1bp, reprice. [MAR21.20]"""
    notional = inst['notional']
    coupon = inst.get('coupon', 0) / 100
    mat_yrs = years_between(mkt['val_date'], inst['maturity'])
    bucket = inst.get('csr_bucket', 1)
    freq = inst.get('freq', 2)
    ccy = inst.get('currency', 'USD')
    if mat_yrs <= 0: return {}

    from data_loader import get_issuer_spread_curve
    issuer_name = inst.get('security', '')
    spread_curve = get_issuer_spread_curve(mkt, issuer_name, inst.get('rating', 'A'), inst.get('issue_type', ''))
    if not spread_curve: return {}

    base_handle = mkt['usd_handle'] if ccy == 'USD' else mkt['gbp_handle']
    csr_tenors = cfg['csr_tenors']
    bond, scale = _build_ql_bond(notional, coupon, inst['maturity'], freq, inst.get('issue_date'))

    base_spread_handle = _build_ql_spread_curve(mkt['ql_val_date'], base_handle, spread_curve)
    bond.setPricingEngine(ql.DiscountingBondEngine(base_spread_handle))
    pv_base = bond.NPV()

    sens = {}
    for bump_tenor in csr_tenors:
        if bump_tenor > mat_yrs + 3: continue
        bumped_spread = dict(spread_curve)
        bumped_spread[bump_tenor] = bumped_spread.get(bump_tenor, 0) + 0.0001
        bumped_handle = _build_ql_spread_curve(mkt['ql_val_date'], base_handle, bumped_spread)
        bond.setPricingEngine(ql.DiscountingBondEngine(bumped_handle))
        cs01 = (bond.NPV() - pv_base) * scale / 0.0001   # MAR21.20: (V_bumped - V_base) / 0.0001
        if abs(cs01) > 0.01:
            sens[f"CSR_NONSEC_{bucket}_{bump_tenor}Y"] = cs01
    return sens


# ── CALLABLE BOND ─────────────────────────────────────────────────────────────

def _build_ql_callable(notional, coupon, mat_date, call_date, freq, crv_handle, cfg, issue_date=None):
    face = 100.0
    scale = notional / face
    hw_a = cfg['params'].get('hw_mean_reversion', 0.03)
    hw_sigma = cfg['params'].get('hw_volatility', 0.015)
    tree_steps = int(cfg['params'].get('hw_tree_steps', 200))
    mat_ql = _to_ql_date(mat_date)
    call_ql = _to_ql_date(call_date)
    issue_ql = _to_ql_date(issue_date) if issue_date else ql.Date(1, 1, 2025)
    period = {2: ql.Semiannual, 4: ql.Quarterly}.get(freq, ql.Semiannual)
    cal = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    schedule = ql.Schedule(issue_ql, mat_ql, ql.Period(period), cal,
        ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Backward, False)
    cs = ql.CallabilitySchedule()
    for d in schedule:
        if d >= call_ql and d < mat_ql:
            cs.append(ql.Callability(ql.BondPrice(face, ql.BondPrice.Clean),
                                     ql.Callability.Call, d))
    bond = ql.CallableFixedRateBond(0, face, schedule, [coupon],
        ql.Actual365Fixed(), ql.Following, face, issue_ql, cs)
    model = ql.HullWhite(crv_handle, hw_a, hw_sigma)
    engine = ql.TreeCallableFixedRateBondEngine(model, tree_steps)
    bond.setPricingEngine(engine)
    return bond, scale, hw_a, hw_sigma

def calc_callable_bond_girr_delta(inst, mkt, cfg):
    mat_yrs = years_between(mkt['val_date'], inst['maturity'])
    if mat_yrs <= 0: return {}
    coupon = inst.get('coupon', 0) / 100
    ccy = inst.get('currency', 'USD')
    rates_dict = mkt['usd_rates'] if ccy == 'USD' else mkt['gbp_sonia_rates']
    base_handle = mkt['usd_handle'] if ccy == 'USD' else mkt['gbp_handle']
    girr_tenors = cfg['girr_tenors']

    bond_base, scale, _, _ = _build_ql_callable(
        inst['notional'], coupon, inst['maturity'], inst['call_date'],
        inst.get('freq', 2), base_handle, cfg, inst.get('issue_date'))
    pv_base = bond_base.NPV()
    inst['_computed_mv'] = pv_base * scale

    sens = {}
    for bt in girr_tenors:
        if bt > mat_yrs + 5: continue
        bumped_rates = dict(rates_dict)
        if bt in bumped_rates:
            bumped_rates[bt] += 0.0001
        from data_loader import _build_ql_curve
        _, bumped_handle, _, _ = _build_ql_curve(mkt['ql_val_date'], bumped_rates)
        bond_bumped, _, _, _ = _build_ql_callable(
            inst['notional'], coupon, inst['maturity'], inst['call_date'],
            inst.get('freq', 2), bumped_handle, cfg, inst.get('issue_date'))
        pv01 = (bond_bumped.NPV() - pv_base) * scale / 0.0001   # MAR21.19: divide by 0.0001
        if abs(pv01) > 0.01:
            sens[f"GIRR_{ccy}_{bt}Y"] = pv01
    return sens

def calc_callable_bond_csr_delta(inst, mkt, cfg):
    mat_yrs = years_between(mkt['val_date'], inst['maturity'])
    if mat_yrs <= 0: return {}
    coupon = inst.get('coupon', 0) / 100
    ccy = inst.get('currency', 'USD')
    bucket = inst.get('csr_bucket', 1)
    base_handle = mkt['usd_handle'] if ccy == 'USD' else mkt['gbp_handle']
    csr_tenors = cfg['csr_tenors']

    from data_loader import get_issuer_spread_curve
    spread_curve = get_issuer_spread_curve(mkt, inst.get('security', ''),
                                           inst.get('rating', 'A'), inst.get('issue_type', ''))
    if not spread_curve: return {}

    base_spread_handle = _build_ql_spread_curve(mkt['ql_val_date'], base_handle, spread_curve)
    bond_base, scale, _, _ = _build_ql_callable(
        inst['notional'], coupon, inst['maturity'], inst['call_date'],
        inst.get('freq', 2), base_spread_handle, cfg, inst.get('issue_date'))
    pv_base = bond_base.NPV()

    sens = {}
    for bump_tenor in csr_tenors:
        if bump_tenor > mat_yrs + 3: continue
        bumped_spread = dict(spread_curve)
        bumped_spread[bump_tenor] = bumped_spread.get(bump_tenor, 0) + 0.0001
        bumped_handle = _build_ql_spread_curve(mkt['ql_val_date'], base_handle, bumped_spread)
        bond_bumped, _, _, _ = _build_ql_callable(
            inst['notional'], coupon, inst['maturity'], inst['call_date'],
            inst.get('freq', 2), bumped_handle, cfg, inst.get('issue_date'))
        cs01 = (bond_bumped.NPV() - pv_base) * scale / 0.0001   # MAR21.20: divide by 0.0001
        if abs(cs01) > 0.01:
            sens[f"CSR_NONSEC_{bucket}_{bump_tenor}Y"] = cs01
    return sens

def calc_callable_bond_girr_vega(inst, mkt, cfg):
    opt_mat = years_between(mkt['val_date'], inst['call_date'])
    und_res = years_between(inst['call_date'], inst['maturity'])
    if opt_mat <= 0 or und_res <= 0: return {}
    coupon = inst.get('coupon', 0) / 100
    ccy = inst.get('currency', 'USD')
    base_handle = mkt['usd_handle'] if ccy == 'USD' else mkt['gbp_handle']
    vega_bump = cfg['params'].get('hw_vega_bump_pct', 0.01)

    bond, scale, hw_a, hw_sigma = _build_ql_callable(
        inst['notional'], coupon, inst['maturity'], inst['call_date'],
        inst.get('freq', 2), base_handle, cfg, inst.get('issue_date'))
    npv_base = bond.NPV()
    model_up = ql.HullWhite(base_handle, hw_a, hw_sigma * (1 + vega_bump))
    tree_steps = int(cfg['params'].get('hw_tree_steps', 200))
    bond.setPricingEngine(ql.TreeCallableFixedRateBondEngine(model_up, tree_steps))
    npv_up = bond.NPV()

    vega_dollar = ((npv_up - npv_base) / vega_bump) * scale
    frtb_sens = vega_dollar * hw_sigma
    vega_tenors = cfg['vega_tenors']
    opt_alloc = allocate_to_tenors(opt_mat, vega_tenors)
    und_alloc = allocate_to_tenors(und_res, vega_tenors)
    sens = {}
    for ot, ow in opt_alloc.items():
        for ut, uw in und_alloc.items():
            sens[f"VEGA_GIRR_{ccy}_{ot}Y_{ut}Y"] = frtb_sens * ow * uw
    return {k: v for k, v in sens.items() if abs(v) > 0.0001}

def calc_callable_bond_girr_curvature(inst, mkt, cfg):
    mat_yrs = years_between(mkt['val_date'], inst['maturity'])
    if mat_yrs <= 0: return {}
    coupon = inst.get('coupon', 0) / 100
    ccy = inst.get('currency', 'USD')
    curv_rw = cfg['params'].get('girr_curvature_rw', 0.017)
    rates_dict = mkt['usd_rates'] if ccy == 'USD' else mkt['gbp_sonia_rates']

    def _callable_npv(shift):
        if shift == 0:
            h = mkt['usd_handle'] if ccy == 'USD' else mkt['gbp_handle']
        else:
            shifted = {t: r + shift for t, r in rates_dict.items()}
            from data_loader import _build_ql_curve
            _, h, _, _ = _build_ql_curve(mkt['ql_val_date'], shifted)
        bond, scale, _, _ = _build_ql_callable(
            inst['notional'], coupon, inst['maturity'], inst['call_date'],
            inst.get('freq', 2), h, cfg, inst.get('issue_date'))
        return bond.NPV() * scale

    pv_base = _callable_npv(0)
    pv_up = _callable_npv(curv_rw)
    pv_dn = _callable_npv(-curv_rw)
    sum_d = sum(calc_bond_girr_delta(inst, mkt, cfg).values())
    return {
        f"CURV_GIRR_{ccy}_UP": -(pv_up - pv_base) + curv_rw * sum_d,
        f"CURV_GIRR_{ccy}_DN": -(pv_dn - pv_base) - curv_rw * sum_d,
    }


# ── EQUITY SPOT + SPX OPTIONS ─────────────────────────────────────────────────

def calc_equity_delta(inst, mkt, cfg):
    mv = inst.get('market_value', 0)
    bucket = inst.get('eq_bucket', 8)
    ticker = inst.get('ticker', '')
    if mv == 0 or not ticker: return {}
    return {f"EQ_{bucket}_{ticker}": mv}

def _spx_iv(S, K, T, mkt):
    vs = mkt.get('spx_vol_surface')
    if vs and len(vs) > 0:
        strikes = sorted(vs.keys())
        K_lo = max([k for k in strikes if k <= K], default=strikes[0])
        K_hi = min([k for k in strikes if k >= K], default=strikes[-1])
        def _v(Kx): return _interp_rate(vs[Kx], T) / 100
        if K_lo == K_hi: return _v(K_lo)
        return _v(K_lo) + (_v(K_hi) - _v(K_lo)) * (K - K_lo) / (K_hi - K_lo)
    atm_ts = {0.08:22.5,0.17:21.8,0.25:21.5,0.50:20.8,0.75:20.2,1.0:19.8,1.5:19.2,2.0:18.8}
    atm = _interp_rate(atm_ts, T) / 100
    log_m = np.log(K / S)
    return max(atm + (-0.15) * log_m * min(1/max(np.sqrt(T),0.2),3) + 0.02*log_m**2, 0.08)

def _bsm_call(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0: return max(S*np.exp(-q*T)-K*np.exp(-r*T), 0)
    d1 = (np.log(S/K)+(r-q+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1-sigma*np.sqrt(T)
    return S*np.exp(-q*T)*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)

def _bsm_greeks(S, K, T, r, q, sigma):
    if T<=0 or sigma<=0: return (1.0 if S>K else 0.0), 0.0, max(S-K,0)
    d1=(np.log(S/K)+(r-q+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    return np.exp(-q*T)*norm.cdf(d1), S*np.exp(-q*T)*norm.pdf(d1)*np.sqrt(T), _bsm_call(S,K,T,r,q,sigma)

def calc_spx_option_delta(inst, mkt, cfg):
    S, K = mkt['spx'], inst['strike']
    T = years_between(mkt['val_date'], inst['maturity'])
    r = _interp_rate(mkt['usd_rates'], T)
    q = cfg['params'].get('spx_div_yield', 0.013)
    sigma = _spx_iv(S, K, T, mkt)
    mult = int(cfg['params'].get('spx_multiplier', 100))
    bucket = inst.get('eq_bucket', 12)
    if T <= 0: return {}
    bsm_delta, _, _ = _bsm_greeks(S, K, T, r, q, sigma)
    return {f"EQ_{bucket}_SPX": inst['notional'] * mult * bsm_delta}

def calc_spx_option_vega(inst, mkt, cfg):
    S, K = mkt['spx'], inst['strike']
    T = years_between(mkt['val_date'], inst['maturity'])
    r = _interp_rate(mkt['usd_rates'], T)
    q = cfg['params'].get('spx_div_yield', 0.013)
    sigma = _spx_iv(S, K, T, mkt)
    mult = int(cfg['params'].get('spx_multiplier', 100))
    bucket = inst.get('eq_bucket', 12)
    if T <= 0: return {}
    _, bv, _ = _bsm_greeks(S, K, T, r, q, sigma)
    total = inst['notional'] * mult * bv * sigma
    alloc = allocate_to_tenors(T, cfg['vega_tenors'])
    return {f"VEGA_EQ_{bucket}_SPX_{t}Y": total*w for t,w in alloc.items()}

def calc_spx_option_curvature(inst, mkt, cfg):
    S, K = mkt['spx'], inst['strike']
    T = years_between(mkt['val_date'], inst['maturity'])
    r = _interp_rate(mkt['usd_rates'], T)
    q = cfg['params'].get('spx_div_yield', 0.013)
    sigma = _spx_iv(S, K, T, mkt)
    c, m = inst['notional'], int(cfg['params'].get('spx_multiplier', 100))
    bucket = inst.get('eq_bucket', 12)
    curv_rw = cfg.get('equity_rw', {}).get(bucket, 0.15)
    if T <= 0: return {}
    pb   = _bsm_call(S, K, T, r, q, sigma) * c * m
    pu   = _bsm_call(S*(1+curv_rw), K, T, r, q, sigma) * c * m
    pd_v = _bsm_call(S*(1-curv_rw), K, T, r, q, sigma) * c * m
    bsm_delta, _, _ = _bsm_greeks(S, K, T, r, q, sigma)
    sd = bsm_delta * c * m
    return {f"CURV_EQ_{bucket}_SPX_UP": -(pu-pb)+curv_rw*sd,
            f"CURV_EQ_{bucket}_SPX_DN": -(pd_v-pb)-curv_rw*sd}


# ── SECURITISATION CSR DELTA [MAR21.20, MAR22.60] ─────────────────────────────

def calc_securitisation_csr_delta(inst, mkt, cfg):
    """
    CSR Sec Non-CTP CS01 — MAR21 compliant.

    For each of the 5 prescribed CSR tenors {0.5, 1, 3, 5, 10}, computes an
    INDEPENDENT CS01 by bumping that pillar of the manufactured tranche spread
    curve by +1bp and repricing the asset-class-specific cashflow schedule
    under (USD SOFR + bumped tranche spread). Five sensitivities, not one
    interpolated value.

    Methodology references:
      - MAR21: CS01 definition, 5 prescribed tenors
      - Sec_Tranche_Curves sheet: manufactured tranche spread curve (Level 3)
      - sec_cashflows.project_tranche_bond: amortisation profile per pool type
      - QuantLib SpreadedLinearZeroInterpolatedTermStructure: pillar-based
            spread curve with linear interpolation in zero-rate space. Bumping a
            single pillar produces a triangular shock that peaks at the bumped
            tenor and decays linearly to zero at adjacent pillars — the
            triangular kernel of Ho (1992) / Reitano (1992), consistent with
            MAR21.8 footnote 3.

    Critical inputs (raises ValueError if missing): inst['id'], inst['notional'],
    inst['issue_date'], inst['maturity'], inst['pool_type'], and a tranche
    spread curve in mkt['tranche_curves'][inst['id']].
    """

    # ── DIAGNOSTIC (temporary) ──
    _diag_id = inst.get('id', '?')
    print(f"  [SEC DIAG5] id={_diag_id} notional={inst.get('notional')!r} type={type(inst.get('notional')).__name__}")
    _diag_pool = inst.get('pool_type', '?')
    _diag_iss = inst.get('issue_date', '?')
    _diag_curves_count = len(mkt.get('tranche_curves', {}))
    _diag_my_curve = mkt.get('tranche_curves', {}).get(inst.get('id'))
    print(f"  [SEC DIAG] id={_diag_id} pool={_diag_pool} iss={_diag_iss} "
          f"total_curves_loaded={_diag_curves_count} my_curve={'YES' if _diag_my_curve else 'NO'}")
    # ── END DIAGNOSTIC ──

    try:
    # ── Critical input validation ──
        inst_id = inst.get('id')
        if inst_id is None:
            raise ValueError("Sec position missing 'id'")

        notional = inst.get('notional')
        if not notional:
            raise ValueError(f"Sec position {inst_id} missing or zero notional")

        issue_date = inst.get('issue_date') or inst.get('Issue Date')
        if pd.isna(issue_date) or issue_date is None:
            raise ValueError(f"Sec position {inst_id} missing critical input: issue_date")

        maturity = inst.get('maturity')
        if pd.isna(maturity) or maturity is None:
            raise ValueError(f"Sec position {inst_id} missing critical input: maturity")

        pool_type = inst.get('pool_type')
        if not pool_type or pool_type == 'nan':
            raise ValueError(f"Sec position {inst_id} missing critical input: pool_type")

        tranche_curves = mkt.get('tranche_curves', {})
        tranche_curve = tranche_curves.get(inst_id)
        if not tranche_curve:
            raise ValueError(
                f"Sec position {inst_id} has no manufactured tranche spread curve "
                f"in mkt['tranche_curves']. Check Sec_Tranche_Curves sheet."
            )

        bucket = inst.get('csr_sec_bucket', 1)
        # coupon_rate = (inst.get('coupon', 0) or 0) / 100.0
        import math
        _raw_coupon = inst.get('coupon', 0)
        if _raw_coupon is None or (isinstance(_raw_coupon, float) and math.isnan(_raw_coupon)):
            _raw_coupon = 0.0  # treat as zero-coupon when data unavailable
        coupon_rate = float(_raw_coupon) / 100.0
        freq = inst.get('freq', 4)

        # ── Convert pandas Timestamps to ql.Date ──
        iss_ql = ql.Date(issue_date.day, issue_date.month, issue_date.year)
        mat_ql = ql.Date(maturity.day, maturity.month, maturity.year)
        val_ql = mkt['ql_val_date']
        
        print(f"  [SEC DIAG6] id={_diag_id} freq={freq} freq_type={type(freq).__name__} "
          f"coupon_rate={coupon_rate} iss_ql={iss_ql} mat_ql={mat_ql}")
        # ── Build the bond via the asset-class-specific profile (sec_cashflows) ──
        bond = project_tranche_bond(
            pool_type=pool_type,
            notional=notional,
            coupon_rate=coupon_rate,
            issue_date=iss_ql,
            maturity_date=mat_ql,
            val_date=val_ql,
            frequency=freq,
        )

        import math
        _cf_list = list(bond.cashflows())
        _nan_count = sum(1 for cf in _cf_list if math.isnan(cf.amount()))
        _first_amounts = [cf.amount() for cf in _cf_list[:3]]
        print(f"  [SEC DIAG7] id={_diag_id} cf_total={len(_cf_list)} "
          f"nan_amounts={_nan_count} first_3_amounts={_first_amounts}")

        sign = getattr(bond, 'frtb_sign', 1.0)

        # ── Helper: build a combined (USD SOFR + tranche spread) discount handle ──
        csr_tenors = cfg.get('csr_tenors', [0.5, 1, 3, 5, 10])

    # Standard QuantLib pattern: use SimpleQuote handles so bumps are applied
    # via .setValue() and the bond automatically reprices via the observer
    # pattern (no curve rebuild required). Anchor pillars at val_date itself
    # plus the 5 CSR tenors so cashflows in the next few weeks are covered.
        spread_quotes = {t: ql.SimpleQuote(tranche_curve[t]) for t in csr_tenors}
        anchor_quote = ql.SimpleQuote(tranche_curve[csr_tenors[0]])   # ← NEW: independent quote
        spread_dates = [val_ql] + [
            val_ql + ql.Period(int(t * 365.25), ql.Days) for t in csr_tenors
        ]
        spread_handles = [
            ql.QuoteHandle(anchor_quote)  # anchor at val_date uses 0.5y spread
        ] + [ql.QuoteHandle(spread_quotes[t]) for t in csr_tenors]

        combined_curve = ql.SpreadedLinearZeroInterpolatedTermStructure(
            mkt['usd_handle'], spread_handles, spread_dates
        )
        combined_curve.enableExtrapolation()
        combined_handle = ql.YieldTermStructureHandle(combined_curve)

        bond.setPricingEngine(ql.DiscountingBondEngine(combined_handle))

        # ── DIAGNOSTIC 4: price against bare USD curve (no spread) ──
        bond.setPricingEngine(ql.DiscountingBondEngine(mkt['usd_handle']))
        _diag_pv_usd_only = bond.NPV() * sign
        # Restore the combined engine before pv_base is computed
        bond.setPricingEngine(ql.DiscountingBondEngine(combined_handle))
        print(f"  [SEC DIAG4] id={_diag_id} pv_usd_only={_diag_pv_usd_only}")
        # ── END DIAGNOSTIC 4 ──

        # ── Base PV ──
        pv_base = bond.NPV() * sign
        # ── DIAGNOSTIC 2 (temporary) ──
        _diag_n_cf = len(bond.cashflows())
        _diag_future_cf = sum(1 for cf in bond.cashflows() if cf.date() > val_ql)
        print(f"  [SEC DIAG2] id={_diag_id} pv_base={pv_base:,.2f} "
              f"total_cf={_diag_n_cf} future_cf={_diag_future_cf} "
              f"sign={sign} curve_at_5y={tranche_curve.get(5, '?')}")
        # ── END DIAGNOSTIC 2 ──

        # ── Bump each CSR pillar by +1bp via SimpleQuote.setValue ──
        # The observer pattern reprices the bond automatically — no curve rebuild.
        BUMP = 0.0001
        cs01_by_tenor = {}
        for target in csr_tenors:
            original = spread_quotes[target].value()
            spread_quotes[target].setValue(original + BUMP)
            pv_bumped = bond.NPV() * sign
            cs01_by_tenor[target] = (pv_bumped - pv_base) / BUMP
            spread_quotes[target].setValue(original)  # restore for next iteration

        # ── Output ──
        return {f"CSR_SEC_NONCTP_{bucket}_{t}Y": v
                for t, v in cs01_by_tenor.items()
                if abs(v) > 0.001}
    except Exception as e:
        import traceback
        print(f"  ❌ SEC FUNCTION ERROR for id={_diag_id}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return {}

# ── SECURITISATION GIRR DELTA [MAR21.19] ──────────────────────────────────────

def calc_securitisation_girr_delta(inst, mkt, cfg):
    """
    GIRR PV01 for a securitisation tranche: bump the USD SOFR curve one tenor
    at a time by +1bp, hold the tranche spread term structure fixed, reprice.

    Mirrors calc_securitisation_csr_delta in cashflow construction and curve
    setup, but the bump target is the risk-free base curve (GIRR risk) rather
    than the tranche credit spread (CSR risk). [MAR21.19]
    """
    _diag_id = inst.get('id', '?')
    try:
        # ── Input validation (same critical fields as CSR function) ──
        inst_id = inst.get('id')
        if inst_id is None:
            raise ValueError("Sec position missing 'id'")

        notional = inst.get('notional')
        if not notional:
            raise ValueError(f"Sec position {inst_id} missing or zero notional")

        issue_date = inst.get('issue_date') or inst.get('Issue Date')
        if pd.isna(issue_date) or issue_date is None:
            raise ValueError(f"Sec position {inst_id} missing critical input: issue_date")

        maturity = inst.get('maturity')
        if pd.isna(maturity) or maturity is None:
            raise ValueError(f"Sec position {inst_id} missing critical input: maturity")

        pool_type = inst.get('pool_type')
        if not pool_type or pool_type == 'nan':
            raise ValueError(f"Sec position {inst_id} missing critical input: pool_type")

        tranche_curve = mkt.get('tranche_curves', {}).get(inst_id)
        if not tranche_curve:
            raise ValueError(
                f"Sec position {inst_id} has no tranche spread curve in mkt['tranche_curves']."
            )

        mat_yrs = years_between(mkt['val_date'], maturity)
        if mat_yrs <= 0:
            return {}

        import math
        _raw_coupon = inst.get('coupon', 0)
        if _raw_coupon is None or (isinstance(_raw_coupon, float) and math.isnan(_raw_coupon)):
            _raw_coupon = 0.0
        coupon_rate = float(_raw_coupon) / 100.0
        freq = inst.get('freq', 4)

        iss_ql = ql.Date(issue_date.day, issue_date.month, issue_date.year)
        mat_ql = ql.Date(maturity.day, maturity.month, maturity.year)
        val_ql = mkt['ql_val_date']

        # ── Build bond cashflow schedule (pool-type amortisation profile) ──
        bond = project_tranche_bond(
            pool_type=pool_type,
            notional=notional,
            coupon_rate=coupon_rate,
            issue_date=iss_ql,
            maturity_date=mat_ql,
            val_date=val_ql,
            frequency=freq,
        )
        sign = getattr(bond, 'frtb_sign', 1.0)

        csr_tenors = cfg.get('csr_tenors', [0.5, 1, 3, 5, 10])
        girr_tenors = cfg['girr_tenors']

        # ── Helper: build combined (SOFR base + fixed tranche spread) handle ──
        # The tranche spread quotes are plain floats (not SimpleQuotes) because
        # we never bump them here — they stay fixed while we vary the SOFR base.
        def _combined_handle(sofr_handle):
            spread_dates = [val_ql] + [
                val_ql + ql.Period(int(t * 365.25), ql.Days) for t in csr_tenors
            ]
            spread_handles = (
                [ql.QuoteHandle(ql.SimpleQuote(tranche_curve[csr_tenors[0]]))]
                + [ql.QuoteHandle(ql.SimpleQuote(tranche_curve[t])) for t in csr_tenors]
            )
            cc = ql.SpreadedLinearZeroInterpolatedTermStructure(
                sofr_handle, spread_handles, spread_dates
            )
            cc.enableExtrapolation()
            return ql.YieldTermStructureHandle(cc)

        # ── Base PV (SOFR + tranche spread, unshocked) ──
        bond.setPricingEngine(ql.DiscountingBondEngine(_combined_handle(mkt['usd_handle'])))
        pv_base = bond.NPV() * sign

        # ── Bump each GIRR tenor on the SOFR curve, hold tranche spread fixed ──
        BUMP = 0.0001
        sens = {}
        for bt in girr_tenors:
            if bt > mat_yrs + 5:
                continue
            bumped_sofr = _build_bumped_curve(mkt['ql_val_date'], mkt['usd_rates'], bt)
            bond.setPricingEngine(ql.DiscountingBondEngine(_combined_handle(bumped_sofr)))
            pv01 = (bond.NPV() * sign - pv_base) / BUMP
            if abs(pv01) > 0.01:
                sens[f"GIRR_USD_{bt}Y"] = pv01

        return sens

    except Exception as e:
        import traceback
        print(f"  ❌ SEC GIRR ERROR for id={_diag_id}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return {}


# ── COMMODITY TRS ─────────────────────────────────────────────────────────────

def calc_commodity_trs_delta(inst, mkt, cfg, bcomtr_weights):
    n = inst['notional']
    sens = {}
    for _, row in bcomtr_weights.iterrows():
        c = row['Commodity']; b = int(row['FRTB Bucket'])
        sens[f"COMM_{b}_{c.replace(' ','_')}"] = row['Weight'] * n
    return sens

def calc_commodity_trs_girr_delta(inst, mkt, cfg):
    """
    GIRR PV01 on the SOFR floating pay-leg of a commodity TRS. [MAR21.19, MAR21.20]

    The pay leg pays SOFR + contractual spread (from 'spread_bps' in holdings).
    Modelled as a fixed-rate bond at (current SOFR_reset + spread), discounted
    on the SOFR curve. Bump-and-reprice across all GIRR tenors per MAR21.19.

    Floater approximation: coupon fixed at current SOFR interpolated at the
    reset tenor + contractual spread. This is exact for the next coupon and a
    good approximation for short-dated TRS (max 1.5y remaining here).

    Sign: negative — bank pays this leg, so rising rates increase cost → negative PV01.
    Pre-divided by 0.0001 per MAR21.20 convention.
    """
    notional = inst['notional']
    mat_yrs = years_between(mkt['val_date'], inst['maturity'])
    if mat_yrs <= 0:
        return {}

    freq         = inst.get('freq', 4)
    spread_bps   = inst.get('spread_bps', 0) or 0
    spread       = spread_bps / 10000.0                   # contractual SOFR+ spread, decimal

    # Use SOFR at the reset tenor as the projected floating rate
    reset_tenor  = 1.0 / freq
    sofr_reset   = _interp_rate(mkt['usd_rates'], reset_tenor)
    total_coupon = sofr_reset + spread                    # SOFR_current + contractual spread

    bond, scale  = _build_ql_bond(notional, total_coupon, inst['maturity'], freq, inst.get('issue_date'))
    pv_base      = _price_bond_ql(bond, mkt['usd_handle'], 0)

    BUMP         = 0.0001
    girr_tenors  = cfg['girr_tenors']
    sens         = {}
    for bt in girr_tenors:
        if bt > mat_yrs + 2:
            continue
        bumped_handle = _build_bumped_curve(mkt['ql_val_date'], mkt['usd_rates'], bt)
        pv01 = (_price_bond_ql(bond, bumped_handle, 0) - pv_base) * scale / BUMP
        
        if abs(pv01) > 0.01:
            sens[f"GIRR_USD_{bt}Y"] = pv01

    return sens


# ── FX ────────────────────────────────────────────────────────────────────────

def calc_fx_delta(inst, mkt, cfg):
    mv = inst.get('market_value', 0)
    if mv == 0: return {}
    return {f"FX_{inst.get('fx_pair','GBP/USD')}": mv}


# ── XCCY BASIS SWAP — USD LEG GIRR [MAR21.19] ─────────────────────────────────

def calc_xccy_usd_leg_girr(inst, mkt, cfg):
    """
    GIRR PV01 for USD leg of XCcy swap, modelled as FixedRateBond on USD curve.

    The USD leg of a cross-currency swap pays fixed USD cash flows and returns
    principal at maturity. This is economically identical to a fixed-rate bond:
    coupon cash flows discounted on the USD risk-free curve. Using QuantLib
    FixedRateBond is consistent with the bond modules and more accurate than a
    manual discount factor loop. [MAR21.19, MAR21.8(3)]

    Coupon assumed equal to par swap rate at maturity tenor (approx), or
    use inst['coupon'] if provided. For a floating leg, use 0 coupon and
    principal only (approximation consistent with SA treatment).
    """
    notional = inst['notional']
    mat_yrs = years_between(mkt['val_date'], inst['maturity'])
    if mat_yrs <= 0: return {}

    coupon = inst.get('coupon', 0) / 100   # 0 if floating leg
    freq = inst.get('freq', 4)             # quarterly typical for swaps
    rates_dict = mkt['usd_rates']
    base_handle = mkt['usd_handle']
    girr_tenors = cfg['girr_tenors']

    bond, scale = _build_ql_bond(notional, coupon, inst['maturity'], freq, inst.get('issue_date'))
    pv_base = _price_bond_ql(bond, base_handle, 0)

    sens = {}
    for bt in girr_tenors:
        if bt > mat_yrs + 2: continue
        bumped_handle = _build_bumped_curve(mkt['ql_val_date'], rates_dict, bt)
        pv01 = (_price_bond_ql(bond, bumped_handle, 0) - pv_base) * scale / 0.0001   # MAR21.19: divide by 0.0001
        if abs(pv01) > 0.01:
            sens[f"GIRR_USD_{bt}Y"] = pv01
    return sens


# ── XCCY BASIS SWAP — GBP LEG GIRR + BASIS + FX [MAR21.19, MAR21.8(3), MAR21.24] ──

def calc_xccy_gbp_leg_girr(inst, mkt, cfg):
    """
    GIRR PV01, XCcy basis sensitivity, and FX delta for the GBP leg.

    GBP leg modelled as FixedRateBond on GBP SONIA curve (consistent with
    IL gilt and bond modules). Using QuantLib is more accurate than the
    prior manual discount loop.

    Three risk components:
    1. GIRR GBP: tenor-by-tenor bumps on GBP SONIA curve, QuantLib repricing
    2. XCcy basis: flat +1bp parallel shift on GBP curve (basis is quoted as
       flat spread over SONIA; MAR21.8(3) — treated as a separate GIRR
       sub-risk factor). Basis sensitivity ≈ PV01 of the GBP leg.
    3. FX delta: USD equivalent of GBP notional [MAR21.24]

    Note: FX rate mkt['usd_gbp'] is USD per GBP (number of USD in 1 GBP).
    GBP notional converted to USD for reporting in USD reporting currency.
    """
    notional = inst['notional']           # GBP notional
    mat_yrs = years_between(mkt['val_date'], inst['maturity'])
    if mat_yrs <= 0: return {}

    fx = mkt['usd_gbp']                   # USD per GBP
    notional_usd = notional * fx          # USD equivalent for reporting

    coupon = inst.get('coupon', 0) / 100
    freq = inst.get('freq', 4)
    rates_dict = mkt['gbp_sonia_rates']
    base_handle = mkt['gbp_handle']
    girr_tenors = cfg['girr_tenors']

    # Build GBP leg as FixedRateBond on SONIA curve (GBP notional; convert to USD at end)
    bond, scale = _build_ql_bond(notional, coupon, inst['maturity'], freq, inst.get('issue_date'))
    pv_base = _price_bond_ql(bond, base_handle, 0)

    sens = {}

    # 1. GIRR GBP: tenor-by-tenor bumps [MAR21.19]
    for bt in girr_tenors:
        if bt > mat_yrs + 2: continue
        bumped_handle = _build_bumped_curve(mkt['ql_val_date'], rates_dict, bt)
        pv01 = (_price_bond_ql(bond, bumped_handle, 0) - pv_base) * scale / 0.0001   # MAR21.19: divide by 0.0001
        if abs(pv01) > 0.01:
            sens[f"GIRR_GBP_{bt}Y"] = pv01

    # 2. XCcy basis sensitivity: flat +1bp shift on entire GBP curve [MAR21.8(3)]
    # Basis is a flat spread over SONIA; sensitivity is the BPV of the full GBP leg.
    # Modelled as parallel shift to isolate the basis sub-risk factor.
    bumped_basis_handle = _build_parallel_shifted_curve(
        mkt['ql_val_date'], rates_dict, 0.0001)
    pv_basis_bumped = _price_bond_ql(bond, bumped_basis_handle, 0) * scale
    basis_sens = (pv_basis_bumped - pv_base * scale) / 0.0001   # MAR21.8(3): divide by 0.0001
    if abs(basis_sens) > 0.01:
        sens["GIRR_GBP_XCCY_BASIS"] = basis_sens

    # 3. FX delta: USD equivalent of GBP notional [MAR21.24]
    sens["FX_GBP/USD"] = notional_usd   # short USD (pay USD to receive GBP)

    return {k: v for k, v in sens.items() if abs(v) > 0.01}


# ── IL GILT [MAR21.19, MAR21.8(2), MAR21.24] ─────────────────────────────────

def calc_il_gilt_sensitivities(inst, mkt, cfg):
    """
    IL Gilt sensitivities: nominal GIRR PV01 + inflation sensitivity + FX delta.

    IL gilts pay coupons and principal linked to RPI/CPI. Pricing uses Method B
    (real-rate discounting of fixed real cashflows), which is mathematically
    equivalent to Method A (inflation-projected nominal cashflows discounted
    at nominal rates) via the Fisher relation: (1+nominal) = (1+real)(1+inflation).

    Risk components (MAR21.20 convention: returned value = ΔPV / 0.0001):

    1. Nominal GIRR PV01 (GIRR_GBP_xY): tenor-by-tenor bumps. Under Fisher with
       inflation expectation held fixed, bumping the real curve by +1bp at tenor t
       is equivalent to bumping the nominal SONIA curve by +1bp at the same tenor.
       This is therefore the MAR21.19 nominal rate sensitivity. [MAR21.19]

    2. Inflation sensitivity (GIRR_GBP_INFLATION): per MAR21.8(b), inflation is
       a separate flat GIRR sub-risk factor (one number per currency, no term
       structure). The bump is +1bp on breakeven, holding nominal fixed; by
       Fisher this equals a -1bp parallel shift on the real curve. Result
       naturally captures long-inflation exposure of the IL holder.
       Fixed-nominal convention is industry practice, not MAR21-mandated text;
       the MAR21.8(b)(iii) separation requirement supports this interpretation.

    3. FX delta (FX_GBP/USD): USD equivalent of GBP notional. [MAR21.24]

    No curvature charge — MAR21 excludes inflation and basis from curvature.
    """
    notional = inst['notional']
    coupon = inst.get('coupon', 0) / 100
    mat_yrs = years_between(mkt['val_date'], inst['maturity'])
    fx = mkt['usd_gbp']
    if mat_yrs <= 0: return {}

    notional_usd = notional * fx          # convert GBP notional to USD reporting ccy
    real_rates = mkt['gbp_real_rates']
    girr_tenors = cfg['girr_tenors']

    bond, scale = _build_ql_bond(notional, coupon, inst['maturity'], 2, inst.get('issue_date'))
    pv_base = _price_bond_ql(bond, mkt['gbp_real_handle'], 0)

    sens = {}

    # 1. Real rate GIRR PV01: tenor-by-tenor bumps [MAR21.19]
    for bt in girr_tenors:
        if bt > mat_yrs + 5: continue
        bumped_handle = _build_bumped_curve(mkt['ql_val_date'], real_rates, bt)
        pv01 = (_price_bond_ql(bond, bumped_handle, 0) - pv_base) * scale / 0.0001   # MAR21.19: divide by 0.0001
        if abs(pv01) > 0.01:
            sens[f"GIRR_GBP_{bt}Y"] = pv01 * fx

    # 2. Inflation sensitivity [MAR21.8(b), MAR21.20]
    # Method A: project real CFs to nominal via breakeven π(t) = nominal gilt(t) − real gilt(t),
    # discount at nominal SONIA rates, bump breakeven by +1bp (cash flow indexation effect).
    # Nominal discount curve held fixed — orthogonal to the real-rate GIRR above.
    # Breakeven uses nominal gilt yields (not SONIA) per standard DMO/BoE methodology.
    BUMP = 0.0001
    gbp_nominal_gilt_rates = mkt['gbp_nominal_gilt_rates']
    nominal_curve          = mkt['gbp_handle'].currentLink()
    val_ql_loc             = mkt['ql_val_date']
    pv_infl_base     = 0.0
    pv_infl_bumped   = 0.0
    for cf in bond.cashflows():
        cf_date = cf.date()
        if cf_date <= val_ql_loc:
            continue
        t        = (cf_date - val_ql_loc) / 365.0          # years to cash flow
        real_cf  = cf.amount() * scale                      # GBP real cash flow
        pi_t     = (_interp_rate(gbp_nominal_gilt_rates, t)
                    - _interp_rate(real_rates, t))          # breakeven: nominal gilt − real gilt
        df_t     = nominal_curve.discount(cf_date)          # nominal SONIA DF
        pv_infl_base   += real_cf * (1.0 + pi_t)         ** t * df_t
        pv_infl_bumped += real_cf * (1.0 + pi_t + BUMP)  ** t * df_t
    infl_sensitivity = (pv_infl_bumped - pv_infl_base) / BUMP
    if abs(infl_sensitivity) > 0.01:
        sens["GIRR_GBP_INFLATION"] = infl_sensitivity * fx

    # 3. FX delta: USD equivalent of GBP notional [MAR21.24]
    sens["FX_GBP/USD"] = notional_usd

    return {k: v for k, v in sens.items() if abs(v) > 0.01}
