# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 16:21:24 2025

@author: amits
"""

import pandas as pd
import QuantLib as ql
import numpy as np
import xlsxwriter as xlsxwriter

today = ql.Date.todaysDate()

# ============================================================================
# STEP 3: KRD CALCULATION USING QUANTLIB
# ============================================================================

def setup_quantlib():
    """Initialize QuantLib settings"""
    calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today
    return calendar, today


# ============================================================================
# STEP 1: DATA LOADING AND PREPROCESSING
# ============================================================================

def load_treasury_rates(file_path="Treasury rates file"):
    """
    Load Treasury rates data
    Columns: observation_date, DGS3MO,	DGS6MO,	GS1, GS2, GS3, GS5, GS10

    """
    rates_df = pd.read_excel(file_path)
    
    # Convert observation_date to datetime
    rates_df['observation_date'] = pd.to_datetime(rates_df['observation_date'])
    
    # Set as index
    rates_df.set_index('observation_date', inplace=True)
    
    # Rename columns for clarity
    rates_df.columns = ['0.25Y', '0.5Y', '1Y', '2Y', '3Y', '5Y', '10Y']
    
    # Convert from percentage to decimal if needed
    if rates_df.iloc[0].max() > 1:
        rates_df = rates_df / 100
    
    return rates_df


def load_bond_holdings(file_path="Your holdings file"):
    """
    Load bond holdings data
    Key columns: Security, IG/HY, Notional exposure$, 
                Issue Date, Call date, Maturity, Coupon
    """
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
    """
    Create QuantLib yield curve from rate dictionary using bond helpers
    rates_dict: {'1Y': 0.0395, '2Y': 0.0378, '3Y': 0.0365, '4Y': 0.03845, '5Y': 0.0391, '7Y': 0.0405, '10Y': 0.0420}
    """
    # Set up calendar and settlement
    calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    settlement_days = 1
    face_value = 100
    
    # Map tenors to dates
    tenor_map = {'0.25Y':0.25, '0.5Y':0.5,'1Y': 1, '2Y': 2, '3Y': 3, '5Y': 5, '10Y': 10}
    
    dates = []
    rates = []
    
    for tenor, years in tenor_map.items():
        # if tenor in rates_dict:
        #     date = evaluation_date + ql.Period(years, ql.Years)
        #     dates.append(date)
        #     rates.append(rates_dict[tenor])
            
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
        
        # Create schedule with only maturity payment (no intermediate coupons)
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

def build_curve_from_zeros(dates, rates, calendar, settlement_days = 1, face_value  = 100):
    bond_helpers = []
    for d, r in zip(dates, rates):
        price = 100 * np.exp(-r * ql.ActualActual(ql.ActualActual.Bond).yearFraction(today, d))
        # Create schedule with only maturity payment (no intermediate coupons)
        helper_schedule = ql.Schedule(today, d, ql.Period(ql.Once), calendar, 
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
    
    curve = ql.PiecewiseLinearZero(today, bond_helpers, ql.ActualActual(ql.ActualActual.Bond))
    return ql.YieldTermStructureHandle(curve)

def calculate_bond_krd(bond_data, yield_curve_handle, calendar, zero_rates, key_tenors=[0.25, 0.5, 1, 2, 3, 5, 10]):
    """
    Calculate KRDs for a single bond
    """
    # Extract bond parameters
    issue_date = ql.Date(bond_data['Issue Date'].day, 
                        bond_data['Issue Date'].month, 
                        bond_data['Issue Date'].year)
    maturity_date = ql.Date(bond_data['Maturity'].day, 
                           bond_data['Maturity'].month, 
                           bond_data['Maturity'].year)
    coupon_rate = bond_data['Coupon']
    notional = bond_data['Notional exposure$']
    position_size = notional/100
    
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
    
    # Create bond object
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
        # Set pricing engine
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
    
    # Generate dates for all 7 tenors
    zero_dates = []
    tenor_years = [0.25, 0.5, 1, 2, 3, 5, 10]
    
    for years in tenor_years:
        if years < 1:
            months = int(years * 12)
            tenor_date = ql.Settings.instance().evaluationDate + ql.Period(months, ql.Months)
        else:
            tenor_date = ql.Settings.instance().evaluationDate + ql.Period(years, ql.Years)
        zero_dates.append(tenor_date)
    
    today = ql.Date.todaysDate()
    
    def calculate_krd(key_rate_index, shift_size=0.0001):
        """
        KRD using interpolated key rate shifts
        """
        # Calculate time to each key rate maturity
        key_times = [ql.ActualActual(ql.ActualActual.Bond).yearFraction(today, d) for d in zero_dates]
        
        def create_bump_function(key_times, bump_size):
            # Simple approach: bump only the exact key rate
            bump_vector = [0.0] * len(key_times)
            bump_vector[key_rate_index] = bump_size
            return bump_vector
        
        # Calculate price with positive bump
        bump_up = create_bump_function(key_times, shift_size)
        rates_up = [r + b for r, b in zip(zero_rates.values(), bump_up)]
        curve_up = build_curve_from_zeros(zero_dates, rates_up, calendar)
        
        if call_date:
            hw_model_up = ql.HullWhite(curve_up, a=0.03, sigma=0.015)
            bond.setPricingEngine(ql.TreeCallableFixedRateBondEngine(hw_model_up, 500))
        else:
            bond.setPricingEngine(ql.DiscountingBondEngine(curve_up))
        price_up = bond.cleanPrice()
        
        # Calculate price with negative bump
        bump_down = create_bump_function(key_times, -shift_size)
        rates_down = [r + b for r, b in zip(zero_rates.values(), bump_down)]
        curve_down = build_curve_from_zeros(zero_dates, rates_down, calendar)
        
        if call_date:
            hw_model_down = ql.HullWhite(curve_down, a=0.03, sigma=0.015)
            bond.setPricingEngine(ql.TreeCallableFixedRateBondEngine(hw_model_down, 500))
        else:
            bond.setPricingEngine(ql.DiscountingBondEngine(curve_down))
        price_down = bond.cleanPrice()
        
        # Calculate sensitivity
        sensitivity = ((price_up-base_price) / (shift_size))*position_size
        # Calculate Key Rate Convexity
        key_rate_convexity = (price_up + price_down - 2 * base_price) / (base_price * shift_size**2)
        return sensitivity, price_up, price_down, key_rate_convexity
        
    # Calculate KRDs for all tenors
    sensitivity_list = []
    for i, (date, rate) in enumerate(zip(zero_dates, zero_rates.values())):
        shift = 0.0001  # 1 bp
        tenor_years = ql.ActualActual(ql.ActualActual.Bond).yearFraction(today, date)
        sensitivity, price_up, price_down, key_rate_convexity = calculate_krd(i, shift)
        
        sensitivity_list.append({
            'Index': i,
            'Tenor': f"{tenor_years:.1f}Y",
            'Base_Rate_%': rate*100,
            'Price_Up': price_up,
            'Price_Down': price_down,
            'Price_Change_Up': price_up - base_price,
            'Price_Change_Down': price_down - base_price,
            'sensitivity': sensitivity,
            'Key_Rate_Convexity': key_rate_convexity
        })
        
    return sensitivity_list, base_price


# ============================================================================
# STEP 4: AGGREGATE KRDs BY IG/HY
# ============================================================================

def calculate_portfolio_krds(holdings_df, current_rates):
    """
    Calculate aggregate KRDs for IG/HY bonds
    """
    calendar, evaluation_date = setup_quantlib()
    yield_curve = create_yield_curve(current_rates, evaluation_date)
    
    IG_sensitivities = {'0.25Y':0, '0.5Y':0, '1Y': 0, '2Y': 0, '3Y': 0, '5Y': 0,  '10Y': 0}
    HY_sensitivities = {'0.25Y':0, '0.5Y':0, '1Y': 0, '2Y': 0, '3Y': 0, '5Y': 0,  '10Y': 0}
    
    results = []
    
    for idx, bond in holdings_df.iterrows():
        try:
            # Calculate KRDs for this bond
            bond_sensitivities, base_price = calculate_bond_krd(bond, yield_curve, calendar, current_rates)
            
            # Add to appropriate bucket
            if bond['IG/HY'] == 'IG':
                for sensitivities_dict in bond_sensitivities:
                    tenor = f"{int(float(sensitivities_dict['Tenor'][:-1]))}Y"  # '1.0Y' -> '1Y'
                    sensitivity_value = sensitivities_dict['sensitivity']
                    if tenor in IG_sensitivities:
                        IG_sensitivities[tenor] += sensitivity_value
            else:
                for sensitivities_dict in bond_sensitivities:
                    tenor = f"{int(float(sensitivities_dict['Tenor'][:-1]))}Y"  # '1.0Y' -> '1Y'
                    sensitivity_value = sensitivities_dict['sensitivity']
                    if tenor in HY_sensitivities:
                        HY_sensitivities[tenor] += sensitivity_value
            
            # Store results
            result = {
                'Security': bond['Security'],
                'Type': bond['IG/HY'],
                'Base_Price': base_price,
                'sensitivities': bond_sensitivities
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error processing bond {bond['Security']}: {e}")
    
    return IG_sensitivities, HY_sensitivities, pd.DataFrame(results)



def export_bond_details_to_excel(bond_details, output_file='bond_sensitivities_detailed.xlsx'):
    """
    Export bond details with expanded sensitivities to Excel format
    
    Parameters:
    bond_details: DataFrame with sensitivities column containing list of dictionaries
    output_file: Output Excel file path
    """
    
    expanded_rows = []
    
    for idx, row in bond_details.iterrows():
        # Get base information for this bond
        base_info = {
            'Security': row['Security'],
            'Type': row['Type'],
            'Base_Price': row['Base_Price']
        }
        
        # Expand each sensitivity entry (one per tenor)
        sensitivities_list = row['sensitivities']
        
        for sens_dict in sensitivities_list:
            # Combine base info with sensitivity details
            expanded_row = base_info.copy()
            
            # Add all fields from the sensitivity dictionary
            for key, value in sens_dict.items():
                expanded_row[key] = value
            
            expanded_rows.append(expanded_row)
    
    # Create DataFrame from expanded rows
    expanded_df = pd.DataFrame(expanded_rows)
    
    # Reorder columns for better readability
    column_order = [
        'Security', 'Type', 'Base_Price', 'Index', 'Tenor', 'Base_Rate_%',
        'Price_Up', 'Price_Down', 'Price_Change_Up', 'Price_Change_Down',
        'sensitivity', 'Key_Rate_Convexity'
    ]
    
    # Only include columns that exist in the DataFrame
    final_columns = [col for col in column_order if col in expanded_df.columns]
    expanded_df = expanded_df[final_columns]
    
    # Sort by Security and then by Index for better organization
    expanded_df = expanded_df.sort_values(['Security', 'Index']).reset_index(drop=True)
    
    # Export to Excel with formatting
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        # Write main data
        expanded_df.to_excel(writer, sheet_name='Bond_Sensitivities', index=False)
        
        # Get workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['Bond_Sensitivities']
        
        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        
        number_format = workbook.add_format({'num_format': '#,##0.0000'})
        percentage_format = workbook.add_format({'num_format': '0.0000%'})
        price_format = workbook.add_format({'num_format': '#,##0.00'})
        
        # Apply header formatting
        for col_num, value in enumerate(expanded_df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Apply number formatting to specific columns
        for col_num, col_name in enumerate(expanded_df.columns):
            if col_name in ['Base_Price', 'Price_Up', 'Price_Down']:
                worksheet.set_column(col_num, col_num, 12, price_format)
            elif col_name in ['Price_Change_Up', 'Price_Change_Down', 'sensitivity', 'Key_Rate_Convexity']:
                worksheet.set_column(col_num, col_num, 15, number_format)
            elif col_name == 'Base_Rate_%':
                worksheet.set_column(col_num, col_num, 12, number_format)
            else:
                worksheet.set_column(col_num, col_num, 15)
        
        # Create summary sheet
        summary_data = []
        for security in expanded_df['Security'].unique():
            security_data = expanded_df[expanded_df['Security'] == security]
            summary_row = {
                'Security': security,
                'Type': security_data['Type'].iloc[0],
                'Base_Price': security_data['Base_Price'].iloc[0],
                'Total_Sensitivity': security_data['sensitivity'].sum(),
                'Num_Tenors': len(security_data)
            }
            summary_data.append(summary_row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Format summary sheet
        summary_worksheet = writer.sheets['Summary']
        for col_num, value in enumerate(summary_df.columns.values):
            summary_worksheet.write(0, col_num, value, header_format)
            summary_worksheet.set_column(col_num, col_num, 15)
    
    print(f"Data exported successfully to {output_file}")
    print(f"Total rows in detailed view: {len(expanded_df)}")
    print(f"Unique securities: {expanded_df['Security'].nunique()}")
    print("\nColumn structure:")
    for i, col in enumerate(expanded_df.columns):
        print(f"  {i+1}. {col}")
    
    return expanded_df

# Modified main execution function to include export
def run_te_analytics_with_export(rates_file='All Constant Maturity TREas rates.xlsx',
                                 holdings_file='Bond holdings.xlsx',
                                 export_file='bond_sensitivities_detailed.xlsx'):
    """
    Main function to run complete TE analytics and export to Excel
    """
    print("Fixed Income Portfolio Tracking Error Analytics")
    print("Using EMPIRICAL DATA ONLY (Last 5 Years)")
    print("=" * 80)
    
    # Step 1: Load data
    print("\n1. Loading data...")
    rates_df = load_treasury_rates(rates_file)
    holdings_df = load_bond_holdings(holdings_file)
    print(f"   Loaded {len(holdings_df)} bond holdings")
    
    # Step 3: Calculate KRDs
    print("\n3. Calculating Key Rate Durations...")
    current_rates = rates_df.iloc[-1].to_dict()  # Use most recent rates
    IG_sensitivities, HY_sensitivities, bond_details = calculate_portfolio_krds(holdings_df, current_rates)
    
    # Create DataFrame for display
    results_data = []
    for tenor in ['0.25Y', '0.5Y', '1Y', '2Y', '3Y', '5Y',  '10Y']:
        row = {
            'Tenor': tenor,
            'IG_sensitivities': f"{IG_sensitivities[tenor]:.4f}",
            'HY_sensitivities': f"{HY_sensitivities[tenor]:.4f}"
        }
        results_data.append(row)
    
    results_df = pd.DataFrame(results_data)
    print(results_df.to_string(index=False))
    
    # Export detailed bond data to Excel
    print("\n4. Exporting detailed data to Excel...")
    expanded_df = export_bond_details_to_excel(bond_details, export_file)
    
    return results_df, bond_details, expanded_df

# Add this to your existing code's main execution block
if __name__ == "__main__":
    # Run the complete TE analytics with Excel export
    summary_df, bond_details, expanded_df = run_te_analytics_with_export()
    
    # You can also export separately if you already have bond_details
    # expanded_df = export_bond_details_to_excel(bond_details, 'my_custom_filename.xlsx')
