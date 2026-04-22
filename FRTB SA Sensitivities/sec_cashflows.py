"""
FRTB SA — Securitisation Cashflow Projection Module
====================================================

Self-contained module that builds asset-class-specific QuantLib bond objects
representing the remaining cashflows of a securitisation tranche, used as the
input to CSR Sec Non-CTP CS01 calculation in sensitivity_calc.py.

Design principles
-----------------
1. QuantLib-first: every bond returned is a `ql.AmortizingFixedRateBond` or
   `ql.FixedRateBond`. Cashflow generation, day-count handling, accrual,
   schedule rolling, and NPV are delegated entirely to QuantLib. This module
   contributes ONLY the asset-class-specific notional vector and the
   schedule parameters — every other piece of bond mathematics is library code.
2. One public function: `project_tranche_bond(...)`. The caller attaches a
   pricing engine and calls .NPV() — no manual cashflow handling required.
3. Critical-input validation: missing or malformed inputs raise ValueError
   with a descriptive message. No silent fallbacks.
4. Asset-class-specific amortisation profiles, documented per builder.
5. Default assumptions exposed as a module-level constant so the methodology
   document can cite them and the smoke test can reference them.

Asset class profiles (see CSR_Sec_NonCTP_Methodology.docx Section 5)
--------------------------------------------------------------------
  CMBS              → bullet (coupon-only, full principal at maturity)
  CLO               → no paydown for `reinvestment_period_years` post-issue,
                      then linear sequential amortisation to maturity
  RMBS              → constant CPR on declining balance (default 8% annual)
  ABS-Auto          → linear amortisation over remaining life from issue
  ABS-CreditCard    → revolving (coupon-only) until `controlled_amort_years`
                      before maturity, then linear paydown
  ABCP              → bullet at remaining maturity

Industry references for default assumptions
-------------------------------------------
  CLO 5y reinvestment period: Western Asset / Citi (Jan 2021); Invesco
    "Understanding CLOs" (Sep 2024); LSEG Yield Book primer; Deutsche Bank
    2026 CLO outlook. The 4–5y norm has been stable across vintages 2.0/3.0.
  RMBS 8% CPR: long-run prime US RMBS conditional prepayment rate proxy.
    Configurable via the `assumptions` dict; sensitivity disclosed in the
    methodology document.

Author: Amit  |  Module version: 1.0  |  Last updated: 11 Apr 2026
"""

import QuantLib as ql


# ──────────────────────────────────────────────────────────────────────────
# Default assumptions per asset class (cited in CSR_Sec_NonCTP_Methodology.docx)
# ──────────────────────────────────────────────────────────────────────────
DEFAULT_ASSUMPTIONS = {
    'RMBS':           {'cpr_annual': 0.08},
    'CLO':            {'reinvestment_period_years': 5.0},
    'ABS-CreditCard': {'controlled_amort_years': 1.0},
    # CMBS, ABS-Auto, ABCP have no tunable assumption — fully determined by
    # issue/maturity/notional/coupon under their structural profile.
}

# Pool types this module recognises. Anything else raises ValueError.
SUPPORTED_POOL_TYPES = {'CMBS', 'CLO', 'RMBS', 'ABS-Auto', 'ABS-CreditCard', 'ABCP'}

# QuantLib conventions used consistently across all builders. Aligned with
# the rest of the FRTB engine (Actual/365 Fixed in data_loader.py).
DAY_COUNT = ql.Actual365Fixed()
CALENDAR = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
BUSINESS_CONVENTION = ql.ModifiedFollowing
# DATE_GENERATION = ql.DateGeneration.Forward
DATE_GENERATION = ql.DateGeneration.Backward


# ──────────────────────────────────────────────────────────────────────────
# Public interface
# ──────────────────────────────────────────────────────────────────────────
def project_tranche_bond(
    pool_type: str,
    notional: float,
    coupon_rate: float,
    issue_date: ql.Date,
    maturity_date: ql.Date,
    val_date: ql.Date,
    frequency: int = 4,
    assumptions: dict = None,
) -> ql.Bond:
    """
    Build a QuantLib bond representing the tranche's full cashflow schedule
    from issue to maturity under its asset-class-specific amortisation profile.

    Parameters
    ----------
    pool_type : str
        One of SUPPORTED_POOL_TYPES.
    notional : float
        Original tranche notional at issue (positive for longs, negative for
        shorts — the calling sensitivity function handles sign correctly via
        the resulting NPV).
    coupon_rate : float
        Decimal coupon (e.g. 0.045 for 4.5%). Used as a fixed coupon for
        cashflow construction; the actual sec leg in real life may be floating,
        but for SA CS01 the spread sensitivity is what matters and the fixed
        cashflow proxy is sufficient and standard.
    issue_date : ql.Date
        Original issue date of the tranche.
    maturity_date : ql.Date
        Legal final maturity date.
    val_date : ql.Date
        Valuation date (used for input validation only — QuantLib's pricing
        engine handles cashflow filtering automatically when NPV is called).
    frequency : int, default 4
        Coupon payments per year. 4 = quarterly (most common for secs).
    assumptions : dict, optional
        Per-asset-class tunables. If None, uses DEFAULT_ASSUMPTIONS for the
        pool type. Recognised keys per pool:
            RMBS:           {'cpr_annual': float}
            CLO:            {'reinvestment_period_years': float}
            ABS-CreditCard: {'controlled_amort_years': float}

    Returns
    -------
    ql.Bond
        A QuantLib `AmortizingFixedRateBond` (for amortising profiles) or
        `FixedRateBond` (for bullet profiles). Caller attaches a pricing
        engine via `bond.setPricingEngine(...)` and calls `bond.NPV()`.

    Raises
    ------
    ValueError
        If pool_type is unsupported, or if any critical input is missing,
        malformed, or inconsistent (e.g. maturity ≤ issue, notional == 0).
    """
    # ── Critical input validation (raise on missing data, no silent fallback) ──
    if pool_type not in SUPPORTED_POOL_TYPES:
        raise ValueError(
            f"Unsupported pool_type '{pool_type}'. "
            f"Must be one of: {sorted(SUPPORTED_POOL_TYPES)}"
        )
    if not isinstance(issue_date, ql.Date) or not isinstance(maturity_date, ql.Date):
        raise ValueError(
            f"issue_date and maturity_date must be ql.Date instances. "
            f"Got issue_date={type(issue_date).__name__}, "
            f"maturity_date={type(maturity_date).__name__}"
        )
    if not isinstance(val_date, ql.Date):
        raise ValueError(f"val_date must be ql.Date, got {type(val_date).__name__}")
    if maturity_date <= issue_date:
        raise ValueError(
            f"maturity_date ({maturity_date}) must be strictly after "
            f"issue_date ({issue_date})"
        )
    if maturity_date <= val_date:
        raise ValueError(
            f"maturity_date ({maturity_date}) must be strictly after "
            f"val_date ({val_date}) — position has already matured"
        )
    if notional == 0:
        raise ValueError("notional must be non-zero")
    if coupon_rate is None or coupon_rate < 0:
        raise ValueError(f"coupon_rate must be non-negative, got {coupon_rate}")
    if frequency not in (1, 2, 4, 12):
        raise ValueError(
            f"frequency must be 1, 2, 4, or 12 payments per year, got {frequency}"
        )

    # Use defaults where caller didn't supply assumptions
    asm = dict(DEFAULT_ASSUMPTIONS.get(pool_type, {}))
    if assumptions:
        asm.update(assumptions)

    ql.Settings.instance().evaluationDate = val_date
    # Build the schedule from issue to maturity. QuantLib will generate all
    # coupon dates per the frequency and roll for business days.
    tenor = ql.Period(int(12 / frequency), ql.Months)
    schedule = ql.Schedule(
        issue_date, maturity_date, tenor, CALENDAR,
        BUSINESS_CONVENTION, BUSINESS_CONVENTION,
        DATE_GENERATION, False
    )
    n_periods = len(schedule) - 1  # number of coupon periods

    # Dispatch to per-asset-class notional vector builder. Each builder returns
    # a list of length n_periods giving the OUTSTANDING NOTIONAL during each
    # coupon period (i.e. the balance on which that period's coupon accrues).
    # The principal repayment in period i is then notionals[i-1] - notionals[i],
    # which QuantLib handles automatically via AmortizingFixedRateBond.
    builders = {
        'CMBS':           _notionals_bullet,
        'ABCP':           _notionals_bullet,
        'ABS-Auto':       _notionals_linear,
        'CLO':            _notionals_clo_reinvest_then_sequential,
        'ABS-CreditCard': _notionals_revolving_then_controlled,
        'RMBS':           _notionals_rmbs_cpr,
    }
    notionals = builders[pool_type](
        notional=abs(notional),  # sign handled by caller
        n_periods=n_periods,
        frequency=frequency,
        asm=asm,
    )

    # Apply original sign back at the end so a short position produces a
    # negative-notional bond, which in turn produces a negative NPV — the
    # CS01 sign then flows through naturally without any manual flip.
    if notional < 0:
        notionals = [-n for n in notionals]

    # Build the bond. For pure bullets (notional vector is constant) we use
    # the simpler FixedRateBond; for everything else, AmortizingFixedRateBond.
    is_bullet = all(abs(n - notionals[0]) < 1e-9 for n in notionals)

    if is_bullet:
        bond = ql.FixedRateBond(
            0,                         # settlement days
            abs(notionals[0]),         # face value
            schedule,
            [coupon_rate],
            DAY_COUNT,
            BUSINESS_CONVENTION,
            100.0,                     # redemption %
            issue_date,
        )
        # Re-sign for shorts — FixedRateBond doesn't accept negative face,
        # so the caller's NPV handling needs to multiply by sign.
        # We attach the sign as an attribute for downstream handling.
        bond.frtb_sign = -1.0 if notional < 0 else 1.0
    else:
        bond = ql.AmortizingFixedRateBond(
            0,                         # settlement days
            notionals,                 # outstanding notional per period
            schedule,
            [coupon_rate],
            DAY_COUNT,
            BUSINESS_CONVENTION,
            issue_date,
        )
        bond.frtb_sign = 1.0  # sign already encoded in the notional vector

    return bond


# ──────────────────────────────────────────────────────────────────────────
# Per-asset-class notional vector builders
# Each returns a list of length n_periods giving outstanding notional
# during each coupon period.
# ──────────────────────────────────────────────────────────────────────────

def _notionals_bullet(notional, n_periods, frequency, asm):
    """
    Bullet profile: outstanding notional is the original notional for every
    period; full principal redeemed at maturity. Used for CMBS conduit deals
    (balloon at maturity) and ABCP (short-dated, no scheduled paydown).
    """
    return [notional] * n_periods


def _notionals_linear(notional, n_periods, frequency, asm):
    """
    Linear amortisation profile: equal principal slice repaid each period
    over the full life from issue. Used for ABS-Auto (consumer auto loans
    amortise close to linearly over their term).
    """
    slice_per_period = notional / n_periods
    return [notional - i * slice_per_period for i in range(n_periods)]


def _notionals_clo_reinvest_then_sequential(notional, n_periods, frequency, asm):
    """
    CLO profile: no principal paydown during the reinvestment period
    (the manager reinvests all proceeds), then linear sequential amortisation
    of the remaining balance from end-of-reinvestment to maturity.

    Default reinvestment period: 5 years (Western Asset/Citi; Invesco 2024;
    LSEG Yield Book; Deutsche Bank 2026 outlook). Per-deal override available
    via assumptions={'reinvestment_period_years': X}.
    """
    reinvest_yrs = asm.get('reinvestment_period_years', 5.0)
    reinvest_periods = int(round(reinvest_yrs * frequency))
    reinvest_periods = min(reinvest_periods, n_periods)  # cap at total life

    n_amort = n_periods - reinvest_periods
    if n_amort <= 0:
        # Entire life is within reinvestment period → bullet at maturity
        return [notional] * n_periods

    slice_per_period = notional / n_amort
    notionals = []
    for i in range(n_periods):
        if i < reinvest_periods:
            notionals.append(notional)
        else:
            j = i - reinvest_periods
            notionals.append(notional - j * slice_per_period)
    return notionals


def _notionals_revolving_then_controlled(notional, n_periods, frequency, asm):
    """
    ABS-CreditCard profile: revolving (coupon-only on full notional) until
    `controlled_amort_years` before maturity, then linear paydown over the
    final controlled-amortisation phase. Standard credit card master trust
    structure.

    Default controlled amortisation period: 1 year. Configurable via
    assumptions={'controlled_amort_years': X}.
    """
    controlled_yrs = asm.get('controlled_amort_years', 1.0)
    controlled_periods = int(round(controlled_yrs * frequency))
    controlled_periods = min(controlled_periods, n_periods)
    controlled_periods = max(controlled_periods, 1)  # at least 1 period of paydown

    revolving_periods = n_periods - controlled_periods
    slice_per_period = notional / controlled_periods

    notionals = []
    for i in range(n_periods):
        if i < revolving_periods:
            notionals.append(notional)
        else:
            j = i - revolving_periods
            notionals.append(notional - j * slice_per_period)
    return notionals


def _notionals_rmbs_cpr(notional, n_periods, frequency, asm):
    """
    RMBS profile: constant Conditional Prepayment Rate (CPR) on declining
    balance. The annual CPR is converted to a per-period Single Monthly
    Mortality (SMM)-equivalent rate matching the coupon frequency.

    Default CPR: 8% annual (long-run prime US RMBS proxy). Configurable via
    assumptions={'cpr_annual': X}. The methodology document discloses the
    sensitivity of total RMBS CS01 to ±2% variation in this assumption.

    Per-period survival rate: (1 - CPR_annual)^(1/frequency)
    Outstanding notional after period i: notional × survival^i

    Note: this is a pure prepayment model — no scheduled amortisation is
    overlaid. For RMBS tranches with significant scheduled paydown, the CPR
    would be added to the scheduled rate; for FRTB SA CS01 purposes, the
    CPR-only approximation is sufficient and conservative.
    """
    cpr = asm.get('cpr_annual', 0.08)
    if not 0 <= cpr < 1:
        raise ValueError(f"cpr_annual must be in [0, 1), got {cpr}")
    survival_per_period = (1.0 - cpr) ** (1.0 / frequency)
    return [notional * (survival_per_period ** i) for i in range(n_periods)]


# ──────────────────────────────────────────────────────────────────────────
# Smoke test — run `python sec_cashflows.py` to verify all builders work
# end-to-end and produce sensible cashflow profiles before wiring into
# the orchestrator.
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 72)
    print("sec_cashflows.py — smoke test")
    print("=" * 72)

    val_date = ql.Date(5, 2, 2026)
    ql.Settings.instance().evaluationDate = val_date

    test_cases = [
        # (pool_type, notional, coupon, issue, maturity, label)
        ('CMBS',           100_000_000, 0.0450, ql.Date(1, 6, 2024), ql.Date(30, 6, 2034), 'CMBS-A1 senior'),
        ('CLO',            200_000_000, 0.0520, ql.Date(15, 7, 2019), ql.Date(15, 10, 2030), 'CLO-2019-LVG-A (amortising)'),
        ('CLO',             45_000_000, 0.0610, ql.Date(15, 8, 2023), ql.Date(15, 10, 2030), 'CLO-2023-LVG-B (reinvesting)'),
        ('RMBS',            50_000_000, 0.0480, ql.Date(1, 3, 2023), ql.Date(31, 12, 2028), 'RMBS-2023-1-A senior'),
        ('ABS-Auto',        80_000_000, 0.0510, ql.Date(15, 6, 2024), ql.Date(30, 6, 2029), 'AUTO-2024-1-A senior'),
        ('ABS-CreditCard', 150_000_000, 0.0490, ql.Date(15, 3, 2024), ql.Date(15, 3, 2029), 'CC-ABS-2024-A senior'),
        ('ABCP',           300_000_000, 0.0535, ql.Date(1, 9, 2025), ql.Date(30, 9, 2026), 'ABCP MultiSeller 2025'),
    ]

    # Build a flat dummy discount curve for NPV calculation in the smoke test
    flat_curve = ql.FlatForward(val_date, 0.04, ql.Actual365Fixed())
    handle = ql.YieldTermStructureHandle(flat_curve)
    engine = ql.DiscountingBondEngine(handle)

    for pool_type, notl, cpn, iss, mat, label in test_cases:
        print(f"\n── {label} ({pool_type}) ──")
        try:
            bond = project_tranche_bond(
                pool_type=pool_type,
                notional=notl,
                coupon_rate=cpn,
                issue_date=iss,
                maturity_date=mat,
                val_date=val_date,
                frequency=4,
            )
            bond.setPricingEngine(engine)
            sign = getattr(bond, 'frtb_sign', 1.0)
            npv = bond.NPV() * sign

            # Inspect the cashflow schedule for cashflows after val date
            future_cfs = [(cf.date(), cf.amount()) for cf in bond.cashflows()
                          if cf.date() > val_date]
            print(f"  Issue:    {iss}    Maturity: {mat}")
            print(f"  Notional: {notl:>15,.0f}  Coupon: {cpn:.2%}  Frequency: 4")
            print(f"  Total scheduled cashflows (full life): {len(bond.cashflows())}")
            print(f"  Future cashflows (post-val):           {len(future_cfs)}")
            print(f"  First 3 future cashflows:")
            for d, a in future_cfs[:3]:
                print(f"    {d}  {a:>15,.2f}")
            print(f"  Last 3 future cashflows:")
            for d, a in future_cfs[-3:]:
                print(f"    {d}  {a:>15,.2f}")
            print(f"  NPV (4% flat discount): {npv:>15,.2f}  ({npv/notl*100:.2f}% of notional)")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")

    # Validation: error handling
    print("\n── Error handling tests ──")
    error_cases = [
        ('Unknown pool type',
         dict(pool_type='UNKNOWN', notional=1e6, coupon_rate=0.04,
              issue_date=ql.Date(1,1,2024), maturity_date=ql.Date(1,1,2029),
              val_date=val_date)),
        ('Maturity before issue',
         dict(pool_type='CMBS', notional=1e6, coupon_rate=0.04,
              issue_date=ql.Date(1,1,2030), maturity_date=ql.Date(1,1,2025),
              val_date=val_date)),
        ('Already matured',
         dict(pool_type='CMBS', notional=1e6, coupon_rate=0.04,
              issue_date=ql.Date(1,1,2020), maturity_date=ql.Date(1,1,2025),
              val_date=val_date)),
        ('Zero notional',
         dict(pool_type='CMBS', notional=0, coupon_rate=0.04,
              issue_date=ql.Date(1,1,2024), maturity_date=ql.Date(1,1,2029),
              val_date=val_date)),
        ('Negative coupon',
         dict(pool_type='CMBS', notional=1e6, coupon_rate=-0.01,
              issue_date=ql.Date(1,1,2024), maturity_date=ql.Date(1,1,2029),
              val_date=val_date)),
    ]
    for label, kwargs in error_cases:
        try:
            project_tranche_bond(**kwargs)
            print(f"  FAIL — {label}: should have raised but didn't")
        except ValueError as e:
            print(f"  PASS — {label}: {e}")

    print("\n" + "=" * 72)
    print("Smoke test complete")
    print("=" * 72)
