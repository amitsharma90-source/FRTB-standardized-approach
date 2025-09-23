# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 22:00:29 2025
CSR Non Securitization Curvature Risk Charge Calculator - Parallel Shift Only
Calculates Base Price, Price Up (+12% parallel), Price Down (-12% parallel)

@author: amits
"""

import pandas as pd
import QuantLib as ql
import numpy as np

today = ql.Date.todaysDate()

def setup_quantlib():
    """Initialize QuantLib settings"""
    calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today
    return calendar, today

def load_treasury_rates(file_path="All Constant Maturity TREas rates.xlsx"):
    """Load Treasury rates data"""
    rates_df = pd.read_excel(file_path)
    rates_df['observation_date'] = pd.to_datetime(rates_df['observation_date'])
    rates_df.set_index('observation_date', inplace=True)
    rates_df.columns = ['0.25Y', '0.5Y', '1Y', '2Y', '3Y', '5Y', '10Y']
    
    # Convert from percentage to decimal if needed
    if rates_df.iloc[0].max() > 1:
        rates_df = rates_df / 100
    
    return rates_df

def load_bond_holdings(file_path="Bond holdings.xlsx"):
    """Load bond holdings data"""
    holdings_df = pd.read_excel(file_path)
    
    # Convert date columns
    date_columns = ['Issue Date', 'Call date', 'Maturity']
    for col in date_columns:
        holdings_df[col] = pd.to_datetime(holdings_df[col])
    
    # Convert coupon to decimal if needed
    if holdings_df['Coupon'].max() > 1:
        holdings_df['Coupon'] = holdings_df['Coupon'] / 100
    
    return holdings_df

def create_yield_curve(rates_dict, evaluation_date):
    """Create QuantLib yield curve from rate dictionary"""
    calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    settlement_days = 1
    face_value = 100
    
    # Map tenors to dates
    tenor_map = {'0.25Y': 0.25, '0.5Y': 0.5, '1Y': 1, '2Y': 2, '3Y': 3, '5Y': 5, '10Y': 10}
    
    dates = []
    rates = []
    
    for tenor, years in tenor_map.items():
        if years < 1:
            months = int(years * 12)
            date = evaluation_date + ql.Period(months, ql.Months)
        else:
            date = evaluation_date + ql.Period(int(years), ql.Years)
        dates.append(date)
        rates.append(rates_dict[tenor])
    
    # Sort by date to ensure proper ordering
    sorted_pairs = sorted(zip(dates, rates), key=lambda x: x[0])
    dates, rates = zip(*sorted_pairs)
    
    # Build curve using bond helpers
    bond_helpers = []
    for d, r in zip(dates, rates):
        price = 100 * np.exp(-r * ql.ActualActual(ql.ActualActual.Bond).yearFraction(evaluation_date, d))
        
        helper_schedule = ql.Schedule(evaluation_date, d, ql.Period(ql.Once), calendar, 
                                    ql.Unadjusted, ql.Unadjusted, 
                                    ql.DateGeneration.Backward, False)
        
        helper = ql.FixedRateBondHelper(
            ql.QuoteHandle(ql.SimpleQuote(price)),
            settlement_days, 
            face_value, 
            helper_schedule, 
            [0.0],  # Zero coupon rate
            ql.ActualActual(ql.ActualActual.Bond)
        )
        bond_helpers.append(helper)
    
    curve = ql.PiecewiseLinearZero(evaluation_date, bond_helpers, ql.ActualActual(ql.ActualActual.Bond))
    return ql.YieldTermStructureHandle(curve)

def build_curve_from_zeros(dates, rates, calendar, settlement_days=1, face_value=100):
    """Build curve from zero rates"""
    bond_helpers = []
    for d, r in zip(dates, rates):
        price = 100 * np.exp(-r * ql.ActualActual(ql.ActualActual.Bond).yearFraction(today, d))
        
        helper_schedule = ql.Schedule(today, d, ql.Period(ql.Once), calendar, 
                                    ql.Unadjusted, ql.Unadjusted, 
                                    ql.DateGeneration.Backward, False)
        
        helper = ql.FixedRateBondHelper(
            ql.QuoteHandle(ql.SimpleQuote(price)),
            settlement_days, 
            face_value, 
            helper_schedule, 
            [0.0],
            ql.ActualActual(ql.ActualActual.Bond)
        )
        bond_helpers.append(helper)
    
    curve = ql.PiecewiseLinearZero(today, bond_helpers, ql.ActualActual(ql.ActualActual.Bond))
    return ql.YieldTermStructureHandle(curve)

def calculate_bond_parallel_shift_prices(bond_data, yield_curve_handle, calendar, zero_rates, shift_size=0.12):
    """
    Calculate Base Price, Price Up, and Price Down for parallel shifts
    shift_size: parallel shift amount (default 12% = 0.12)
    12% is maximum risk weight for CSR non securitization delta
    """
    # Extract bond parameters
    issue_date = ql.Date(bond_data['Issue Date'].day, 
                        bond_data['Issue Date'].month, 
                        bond_data['Issue Date'].year)
    maturity_date = ql.Date(bond_data['Maturity'].day, 
                           bond_data['Maturity'].month, 
                           bond_data['Maturity'].year)
    coupon_rate = bond_data['Coupon']
    
    # Check if callable
    call_date = None
    if pd.notna(bond_data['Call date']):
        call_date = ql.Date(bond_data['Call date'].day, 
                           bond_data['Call date'].month, 
                           bond_data['Call date'].year)
    
    # Create bond schedule
    schedule = ql.Schedule(issue_date, maturity_date, ql.Period(ql.Annual),
                          calendar, ql.Unadjusted, ql.Unadjusted,
                          ql.DateGeneration.Backward, False)
    
    # Create bond object and set pricing engine
    if call_date:
        # Callable bond
        callability_schedule = ql.CallabilitySchedule()
        callability_schedule.append(
            ql.Callability(
                ql.BondPrice(100.0, ql.BondPrice.Clean),
                ql.Callability.Call,
                call_date
            )
        )
        bond = ql.CallableFixedRateBond(1, 100, schedule, [coupon_rate],
                                        ql.ActualActual(ql.ActualActual.Bond),
                                        ql.Following, 100.0, issue_date, 
                                        callability_schedule)
        # Set pricing engine for base case
        hw_model = ql.HullWhite(yield_curve_handle, a=0.03, sigma=0.015)
        engine = ql.TreeCallableFixedRateBondEngine(hw_model, 500)
        bond.setPricingEngine(engine)
    else:
        # Regular bond
        bond = ql.FixedRateBond(1, 100, schedule, [coupon_rate],
                               ql.ActualActual(ql.ActualActual.Bond))
        engine = ql.DiscountingBondEngine(yield_curve_handle)
        bond.setPricingEngine(engine)
    
    # Calculate base price
    base_price = bond.cleanPrice()
    
    # Generate dates for all tenors
    zero_dates = []
    tenor_years = [0.25, 0.5, 1, 2, 3, 5, 10]
    
    for years in tenor_years:
        if years < 1:
            months = int(years * 12)
            tenor_date = ql.Settings.instance().evaluationDate + ql.Period(months, ql.Months)
        else:
            tenor_date = ql.Settings.instance().evaluationDate + ql.Period(years, ql.Years)
        zero_dates.append(tenor_date)
    
    # Calculate Price Up (parallel shift up by 12%)
    rates_up = [rate + shift_size for rate in zero_rates.values()]
    curve_up = build_curve_from_zeros(zero_dates, rates_up, calendar)
    
    if call_date:
        hw_model_up = ql.HullWhite(curve_up, a=0.03, sigma=0.015)
        bond.setPricingEngine(ql.TreeCallableFixedRateBondEngine(hw_model_up, 500))
    else:
        bond.setPricingEngine(ql.DiscountingBondEngine(curve_up))
    price_up = bond.cleanPrice()
    
    # Calculate Price Down (parallel shift down by 12%)
    rates_down = [rate - shift_size for rate in zero_rates.values()]
    curve_down = build_curve_from_zeros(zero_dates, rates_down, calendar)
    
    if call_date:
        hw_model_down = ql.HullWhite(curve_down, a=0.03, sigma=0.015)
        bond.setPricingEngine(ql.TreeCallableFixedRateBondEngine(hw_model_down, 500))
    else:
        bond.setPricingEngine(ql.DiscountingBondEngine(curve_down))
    price_down = bond.cleanPrice()
    
    return base_price, price_up, price_down

def calculate_portfolio_parallel_shifts(holdings_df, current_rates):
    """
    Calculate parallel shift prices for all bonds in portfolio
    """
    calendar, evaluation_date = setup_quantlib()
    yield_curve = create_yield_curve(current_rates, evaluation_date)
    
    results = []
    
    for idx, bond in holdings_df.iterrows():
        try:
            # Calculate parallel shift prices for this bond
            base_price, price_up, price_down = calculate_bond_parallel_shift_prices(
                bond, yield_curve, calendar, current_rates
            )
            
            # Store results
            result = {
                'Security': bond['Security'],
                'IG_HY': bond['IG/HY'],
                'Notional_Exposure': bond['Notional exposure$'],
                'Coupon': bond['Coupon'],
                'Issue_Date': bond['Issue Date'],
                'Call_Date': bond['Call date'] if pd.notna(bond['Call date']) else None,
                'Maturity': bond['Maturity'],
                'Base_Price': base_price,
                'Price_Up': price_up,
                'Price_Down': price_down,
                'Price_Change_Up': price_up - base_price,
                'Price_Change_Down': price_down - base_price
            }
            results.append(result)
            
            print(f"Processed: {bond['Security']} - Base: {base_price:.4f}, Up: {price_up:.4f}, Down: {price_down:.4f}")
            
        except Exception as e:
            print(f"Error processing bond {bond['Security']}: {e}")
    
    return pd.DataFrame(results)

def export_to_excel(results_df, output_file='csr_non_sec_12%_curvature_parallel_shifts.xlsx'):
    """Export results to Excel with formatting"""
    
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        # Write main data
        results_df.to_excel(writer, sheet_name='CSR_Non_Sec_Curvature_Data', index=False)
        
        # Get workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['CSR_Non_Sec_Curvature_Data']
        
        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        
        number_format = workbook.add_format({'num_format': '#,##0.0000'})
        price_format = workbook.add_format({'num_format': '#,##0.00'})
        currency_format = workbook.add_format({'num_format': '$#,##0'})
        percentage_format = workbook.add_format({'num_format': '0.00%'})
        date_format = workbook.add_format({'num_format': 'mm/dd/yyyy'})
        
        # Apply header formatting
        for col_num, value in enumerate(results_df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Apply column formatting
        for col_num, col_name in enumerate(results_df.columns):
            if col_name in ['Base_Price', 'Price_Up', 'Price_Down']:
                worksheet.set_column(col_num, col_num, 12, price_format)
            elif col_name in ['Price_Change_Up', 'Price_Change_Down']:
                worksheet.set_column(col_num, col_num, 15, number_format)
            elif col_name == 'Notional_Exposure':
                worksheet.set_column(col_num, col_num, 15, currency_format)
            elif col_name == 'Coupon':
                worksheet.set_column(col_num, col_num, 10, percentage_format)
            elif col_name in ['Issue_Date', 'Call_Date', 'Maturity']:
                worksheet.set_column(col_num, col_num, 12, date_format)
            else:
                worksheet.set_column(col_num, col_num, 15)
        
        # Create summary statistics
        summary_data = {
            'Metric': [
                'Total Bonds',
                'IG Bonds', 
                'HY Bonds',
                'Callable Bonds',
                'Average Base Price',
                'Average Price Up',
                'Average Price Down',
                'Average Price Change Up',
                'Average Price Change Down'
            ],
            'Value': [
                len(results_df),
                len(results_df[results_df['IG_HY'] == 'IG']),
                len(results_df[results_df['IG_HY'] == 'HY']),
                len(results_df[results_df['Call_Date'].notna()]),
                results_df['Base_Price'].mean(),
                results_df['Price_Up'].mean(),
                results_df['Price_Down'].mean(),
                results_df['Price_Change_Up'].mean(),
                results_df['Price_Change_Down'].mean()
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Format summary sheet
        summary_worksheet = writer.sheets['Summary']
        for col_num, value in enumerate(summary_df.columns.values):
            summary_worksheet.write(0, col_num, value, header_format)
            if col_num == 0:
                summary_worksheet.set_column(col_num, col_num, 25)
            else:
                summary_worksheet.set_column(col_num, col_num, 15, number_format)
    
    print(f"\nData exported successfully to {output_file}")
    print(f"Total bonds processed: {len(results_df)}")
    print(f"IG bonds: {len(results_df[results_df['IG_HY'] == 'IG'])}")
    print(f"HY bonds: {len(results_df[results_df['IG_HY'] == 'HY'])}")
    
    return results_df

def run_csr_non_sec_curvature_analysis(
    rates_file="All Constant Maturity TREas rates.xlsx",
    holdings_file="Bond holdings.xlsx",
    export_file='CSR_Non_Sec_curvature_parallel_shifts.xlsx'
):
    """
    Main function to run CSR Non Sec curvature analysis with parallel shifts only
    """
    print("CSR Non Sec  Curvature Risk Charge Calculator - Parallel Shifts Only")
    print("Shift Amount: +/- 12% (parallel across all tenors)")
    print("=" * 80)
    
    # Step 1: Load data
    print("\n1. Loading data...")
    rates_df = load_treasury_rates(rates_file)
    holdings_df = load_bond_holdings(holdings_file)
    print(f"   Loaded {len(holdings_df)} bond holdings")
    print(f"   Using rates from: {rates_df.index[-1].strftime('%Y-%m-%d')}")
    
    # Step 2: Calculate parallel shift prices
    print("\n2. Calculating parallel shift prices...")
    current_rates = rates_df.iloc[-1].to_dict()  # Use most recent rates
    
    print("Current rates:")
    for tenor, rate in current_rates.items():
        print(f"   {tenor}: {rate*100:.4f}%")
    
    results_df = calculate_portfolio_parallel_shifts(holdings_df, current_rates)
    
    # Step 3: Export to Excel
    print("\n3. Exporting to Excel...")
    final_df = export_to_excel(results_df, export_file)
    
    # Display summary
    print(f"\nSummary Statistics:")
    print(f"Average Base Price: {results_df['Base_Price'].mean():.4f}")
    print(f"Average Price Up: {results_df['Price_Up'].mean():.4f}")
    print(f"Average Price Down: {results_df['Price_Down'].mean():.4f}")
    print(f"Average Price Change Up: {results_df['Price_Change_Up'].mean():.4f}")
    print(f"Average Price Change Down: {results_df['Price_Change_Down'].mean():.4f}")
    
    return results_df

if __name__ == "__main__":
    # Run the GIRR curvature analysis
    results = run_csr_non_sec_curvature_analysis()
    print("\nAnalysis completed!")
