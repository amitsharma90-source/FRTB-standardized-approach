# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 10:15:52 2025

@author: amits
"""

# -*- coding: utf-8 -*-
"""
GIRR Curvature Risk Charge Calculator - Complete Implementation
Following the 10-step methodology provided

@author: amits
"""

import pandas as pd
# import numpy as np
import math

class GIRRCurvatureCalculator:
    def __init__(self):
        self.rw = 0.017  # Risk weight (1.7%)
        self.sensitivities_data = None
        self.parallel_shifts_data = None
        self.misc_weights_data = None
        self.results = {}
        
    def load_data(self, sensitivities_file, parallel_shifts_file):
        """
        Load all required input files
        """
        print("Loading data files...")
        
        # Load sensitivities data
        try:
            self.sensitivities_data = pd.read_excel(sensitivities_file, sheet_name='GIRR Sensitivities')
            print(f"✓ Loaded {len(self.sensitivities_data)} sensitivity records")
        except Exception as e:
            print(f"Error loading sensitivities file: {e}")
            return False
        
        # Load parallel shifts data
        try:
            self.parallel_shifts_data = pd.read_excel(parallel_shifts_file)
            print(f"✓ Loaded {len(self.parallel_shifts_data)} parallel shift records")
        except Exception as e:
            print(f"Error loading parallel shifts file: {e}")
            return False
        
        # Load misc weights data
        # try:
        #     self.misc_weights_data = pd.read_excel(misc_weights_file, sheet_name='Misc Weights')
        #     print(f"✓ Loaded misc weights data")
        # except Exception as e:
        #     print(f"Error loading misc weights file: {e}")
        #     return False
        
        return True
    
    def step1_filter_sensitivities(self):
        """
        Step 1: Collect sensitivities from "GIRR Sensitivities" sheet
        Filter only Bond assets (exclude CCBS)
        """
        print("\nStep 1: Filtering bond sensitivities...")
        
        # Filter only bonds (exclude CCBS)
        bond_sensitivities = self.sensitivities_data[
            self.sensitivities_data['Asset'] == 'Bond'
        ].copy()
        
        print(f"✓ Filtered to {len(bond_sensitivities)} bond sensitivity records")
        
        # Display sample data
        print("\nSample data:")
        print(bond_sensitivities.head())
        
        self.results['bond_sensitivities'] = bond_sensitivities
        return bond_sensitivities
    
    def step2_aggregate_currency_sensitivities(self):
        """
        Step 2: Aggregated Currency bucket delta sensitivity based on currency
        Sum all Bond Sensitivities irrespective of tenor by currency
        """
        print("\nStep 2: Aggregating sensitivities by currency...")
        
        bond_sensitivities = self.results['bond_sensitivities']
        
        # Aggregate by currency (ignore tenor)
        currency_sensitivities = bond_sensitivities.groupby('Currency')['Sensitivity'].sum()
        
        print("Currency-wise total sensitivities:")
        for currency, sensitivity in currency_sensitivities.items():
            print(f"  {currency}: {sensitivity:,.6f}")
        
        self.results['currency_sensitivities'] = currency_sensitivities
        return currency_sensitivities
    
    def step3_process_parallel_shifts(self):
        """
        Step 3: Read girr_curvature_parallel_shifts_input file for each position
        Calculate position-adjusted values
        """
        print("\nStep 3: Processing parallel shifts data...")
        
        df = self.parallel_shifts_data.copy()
        
        # Calculate position size (divide notional by 100)
        df['Position_Size'] = df['Notional_Exposure'] / 100
        
        # Calculate position-adjusted values
        df['V_base'] = df['Base_Price'] * df['Position_Size']
        df['V_RW_up'] = df['Price_Up'] * df['Position_Size']
        df['V_RW_down'] = df['Price_Down'] * df['Position_Size']
        
        print(f"✓ Processed {len(df)} positions")
        print("\nSample calculations:")
        print(df[['Security', 'Currency', 'Position_Size', 'V_base', 'V_RW_up', 'V_RW_down']].head())
        
        self.results['position_data'] = df
        return df
    
    def step4_aggregate_by_currency(self):
        """
        Step 4: Aggregate V_base, V_RW_up, V_RW_down by Currency
        """
        print("\nStep 4: Aggregating position values by currency...")
        
        df = self.results['position_data']
        
        # Aggregate by currency
        currency_aggregates = df.groupby('Currency').agg({
            'V_base': 'sum',
            'V_RW_up': 'sum',
            'V_RW_down': 'sum'
        })
        
        print("Currency-wise aggregated values:")
        print(currency_aggregates)
        
        self.results['currency_aggregates'] = currency_aggregates
        return currency_aggregates
    
    def step5_calculate_curvature_values(self):
        """
        Step 5: Calculate Curvature_Up and Curvature_Down for each currency
        """
        print("\nStep 5: Calculating curvature values...")
        
        currency_aggregates = self.results['currency_aggregates']
        currency_sensitivities = self.results['currency_sensitivities']
        
        curvature_results = {}
        
        for currency in currency_aggregates.index:
            # Get values for this currency
            v_base = currency_aggregates.loc[currency, 'V_base']
            v_rw_up = currency_aggregates.loc[currency, 'V_RW_up']
            v_rw_down = currency_aggregates.loc[currency, 'V_RW_down']
            
            # Get sensitivity for this currency
            sensitivity = currency_sensitivities.get(currency, 0)
            
            # Calculate curvature values
            curvature_up = -(v_rw_up - v_base - self.rw * sensitivity)
            curvature_down = -(v_rw_down - v_base + self.rw * sensitivity)
            
            curvature_results[currency] = {
                'V_base': v_base,
                'V_RW_up': v_rw_up,
                'V_RW_down': v_rw_down,
                'Sensitivity': sensitivity,
                'Curvature_Up': curvature_up,
                'Curvature_Down': curvature_down
            }
            
            print(f"\n{currency}:")
            print(f"  V_base: {v_base:,.2f}")
            print(f"  V_RW_up: {v_rw_up:,.2f}")
            print(f"  V_RW_down: {v_rw_down:,.2f}")
            print(f"  Sensitivity: {sensitivity:,.6f}")
            print(f"  Curvature_Up: {curvature_up:,.4f}")
            print(f"  Curvature_Down: {curvature_down:,.4f}")
        
        self.results['curvature_results'] = curvature_results
        return curvature_results
    
    def step6_calculate_bucket_charges(self):
        """
        Step 6: Bucket charge K_b = max(Curvature_Up, Curvature_Down)
        """
        print("\nStep 6: Calculating bucket charges (K_b)...")
        
        curvature_results = self.results['curvature_results']
        bucket_charges = {}
        
        for currency, values in curvature_results.items():
            k_b = max(values['Curvature_Up'], values['Curvature_Down'])
            bucket_charges[currency] = k_b
            print(f"  {currency}: K_b = {k_b:,.4f}")
        
        self.results['bucket_charges'] = bucket_charges
        return bucket_charges
    
    def step7_calculate_psi(self):
        """
        Step 7: Calculate ψ (psi) for each currency pair
        ψ(K_b, K_c) = 0 if both K_b and K_c are negative, 1 otherwise
        """
        print("\nStep 7: Calculating ψ (psi) values...")
        
        bucket_charges = self.results['bucket_charges']
        currencies = list(bucket_charges.keys())
        psi_matrix = {}
        
        print("Psi matrix:")
        print("Currency pairs where both K_b and K_c are negative get ψ = 0, otherwise ψ = 1")
        
        for i, curr1 in enumerate(currencies):
            psi_matrix[curr1] = {}
            for j, curr2 in enumerate(currencies):
                if i != j:  # Different currencies
                    k_b = bucket_charges[curr1]
                    k_c = bucket_charges[curr2]
                    
                    # ψ = 0 if both negative, 1 otherwise
                    psi = 0 if (k_b < 0 and k_c < 0) else 1
                    psi_matrix[curr1][curr2] = psi
                    
                    print(f"  ψ({curr1},{curr2}): K_b={k_b:.4f}, K_c={k_c:.4f} → ψ={psi}")
        
        self.results['psi_matrix'] = psi_matrix
        return psi_matrix
    
    def step8_get_correlations(self):
        """
        Step 8: Get inter-bucket correlations from "Misc Weights" sheet
        Calculate medium, high, and low correlations
        """
        print("\nStep 8: Calculating correlation scenarios...")
        
        # Get base correlation from misc weights (assuming single correlation value)
        # base_corr = self.misc_weights_data['Correlation%'].iloc[0] * 0.01  # Convert % to decimal
    	#γ_bc needs to be squared. So here inter bucket aggregation for curvature risk charge is different from delta risk charge
        base_corr = 0.5*0.5
        print(f"Base correlation (γ_bc): {base_corr:.4f}")
        
        # Calculate correlation scenarios
        correlations = {
            'medium': base_corr,
            'high': min(base_corr * 1.25, 1.0),  # Cap at 100%
            'low': max(2 * base_corr - 1.0, 0.75 * base_corr)  # Formula from BIS
        }
        
        print("Correlation scenarios:")
        for scenario, corr in correlations.items():
            print(f"  {scenario.capitalize()}: γ_bc = {corr:.4f}")
        
        self.results['correlations'] = correlations
        return correlations
    
    def step9_calculate_total_curvature_risk(self):
        """
        Step 9: Calculate total curvature risk for all three correlation scenarios
        Formula: √{max(0, Σ(K_b²) + ΣΣ(γ_bc × ψ(K_b,K_c) × K_b × K_c))}
        """
        print("\nStep 9: Calculating total curvature risk...")
        
        bucket_charges = self.results['bucket_charges']
        psi_matrix = self.results['psi_matrix']
        correlations = self.results['correlations']
        
        currencies = list(bucket_charges.keys())
        total_risks = {}
        
        for scenario, gamma_bc in correlations.items():
            print(f"\n{scenario.capitalize()} correlation scenario (γ_bc = {gamma_bc:.4f}):")
            
            # Calculate Σ(K_b²)
            sum_kb_squared = sum(k_b**2 for k_b in bucket_charges.values())
            print(f"  Σ(K_b²) = {sum_kb_squared:,.6f}")
            
            # Calculate cross terms: ΣΣ(γ_bc × ψ(K_b,K_c) × K_b × K_c)
            cross_terms = 0
            for i, curr1 in enumerate(currencies):
                for j, curr2 in enumerate(currencies):
                    if i != j:  # Different currencies
                        k_b = bucket_charges[curr1]
                        k_c = bucket_charges[curr2]
                        psi = psi_matrix[curr1][curr2]
                        
                        cross_term = gamma_bc * psi * k_b * k_c
                        cross_terms += cross_term
                        
                        if abs(cross_term) > 1e-10:  # Only print significant terms
                            print(f"    {curr1}-{curr2}: {gamma_bc:.4f} × {psi} × {k_b:.4f} × {k_c:.4f} = {cross_term:,.6f}")
            
            print(f"  Cross terms sum = {cross_terms:,.6f}")
            
            # Calculate total under square root
            under_sqrt = sum_kb_squared + cross_terms
            print(f"  Under √: {under_sqrt:,.6f}")
            
            # Final calculation
            total_risk = math.sqrt(max(0, under_sqrt))
            total_risks[scenario] = total_risk
            
            print(f"  Total Curvature Risk ({scenario}): {total_risk:,.4f}")
        
        self.results['total_curvature_risks'] = total_risks
        return total_risks
    
    def step10_export_results(self, output_file='girr_curvature_results.xlsx'):
        """
        Step 10: Export all results to Excel
        """
        print(f"\nStep 10: Exporting results to {output_file}...")
        
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Format definitions
            header_format = workbook.add_format({
                'bold': True, 'fg_color': '#D7E4BC', 'border': 1
            })
            number_format = workbook.add_format({'num_format': '#,##0.0000'})
            currency_format = workbook.add_format({'num_format': '$#,##0.00'})
            
            # 1. Position Level Data
            position_df = self.results['position_data'][[
                'Security', 'Currency', 'Notional_Exposure', 'Position_Size',
                'Base_Price', 'Price_Up', 'Price_Down',
                'V_base', 'V_RW_up', 'V_RW_down'
            ]]
            position_df.to_excel(writer, sheet_name='Position_Level', index=False)
            
            # 2. Currency Aggregates
            currency_agg_df = pd.DataFrame(self.results['currency_aggregates'])
            currency_agg_df.to_excel(writer, sheet_name='Currency_Aggregates')
            
            # 3. Curvature Calculations
            curvature_df = pd.DataFrame(self.results['curvature_results']).T
            curvature_df.to_excel(writer, sheet_name='Curvature_Calculations')
            
            # 4. Bucket Charges
            bucket_charges_df = pd.DataFrame(list(self.results['bucket_charges'].items()),
                                           columns=['Currency', 'K_b'])
            bucket_charges_df.to_excel(writer, sheet_name='Bucket_Charges', index=False)
            
            # 5. Correlation Matrix (expanded for all scenarios)
            corr_data = []
            for scenario, gamma in self.results['correlations'].items():
                corr_data.append({'Scenario': scenario, 'Correlation': gamma})
            corr_df = pd.DataFrame(corr_data)
            corr_df.to_excel(writer, sheet_name='Correlations', index=False)
            
            # 6. Final Results
            final_df = pd.DataFrame(list(self.results['total_curvature_risks'].items()),
                                  columns=['Scenario', 'Total_Curvature_Risk'])
            final_df.to_excel(writer, sheet_name='Final_Results', index=False)
            
            # 7. Summary Sheet
            summary_data = {
                'Metric': [
                    'Number of Positions',
                    'Number of Currencies',
                    'Risk Weight Used',
                    'Medium Correlation Risk',
                    'High Correlation Risk',
                    'Low Correlation Risk',
                    'Maximum Risk (Conservative)'
                ],
                'Value': [
                    len(self.results['position_data']),
                    len(self.results['bucket_charges']),
                    f"{self.rw*100:.1f}%",
                    f"{self.results['total_curvature_risks']['medium']:.4f}",
                    f"{self.results['total_curvature_risks']['high']:.4f}",
                    f"{self.results['total_curvature_risks']['low']:.4f}",
                    f"{max(self.results['total_curvature_risks'].values()):.4f}"
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Apply formatting to all sheets
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                worksheet.set_row(0, None, header_format)
        
        print(f"✓ Results exported to {output_file}")
        
        # Print final summary
        print(f"\n" + "="*60)
        print("GIRR CURVATURE RISK CHARGE CALCULATION - FINAL RESULTS")
        print("="*60)
        for scenario, risk in self.results['total_curvature_risks'].items():
            print(f"{scenario.upper()} Correlation Scenario: {risk:,.4f}")
        print(f"\nREQUIRED CAPITAL (Max): {max(self.results['total_curvature_risks'].values()):,.4f}")
        print("="*60)
    
    def run_complete_calculation(self, sensitivities_file, parallel_shifts_file, 
                               # misc_weights_file, 
                               output_file='girr_curvature_results.xlsx'):
        """
        Run the complete 10-step GIRR Curvature calculation
        """
        print("GIRR CURVATURE RISK CHARGE CALCULATOR")
        print("="*50)
        
        # Load data
        if not self.load_data(sensitivities_file, parallel_shifts_file):
            return None
        
        # Execute all steps
        self.step1_filter_sensitivities()
        self.step2_aggregate_currency_sensitivities()
        self.step3_process_parallel_shifts()
        self.step4_aggregate_by_currency()
        self.step5_calculate_curvature_values()
        self.step6_calculate_bucket_charges()
        self.step7_calculate_psi()
        self.step8_get_correlations()
        self.step9_calculate_total_curvature_risk()
        self.step10_export_results(output_file)
        
        return self.results


# Usage example
if __name__ == "__main__":
    calculator = GIRRCurvatureCalculator()
    
    # File paths (update these to your actual file locations)
    sensitivities_file = "Sensitivities and weights.xlsx"  # Contains "GIRR Sensitivities" sheet
    parallel_shifts_file = "girr_curvature_parallel_shifts_input.xlsx"  # Output from previous code
    # misc_weights_file = "misc_weights.xlsx"  # Contains "Misc Weights" sheet
    output_file = "girr_curvature_final_results.xlsx"
    
    # Run complete calculation
    try:
        results = calculator.run_complete_calculation(
            sensitivities_file,
            parallel_shifts_file, 
            # misc_weights_file,
            output_file
        )
        
        if results:
            print("\n✅ GIRR Curvature calculation completed successfully!")
        else:
            print("\n❌ Calculation failed. Please check input files.")
            
    except Exception as e:
        print(f"\n❌ Error during calculation: {e}")
        print("Please verify:")
        print("1. All input files exist and are accessible")
        print("2. Required sheets exist in the Excel files")
        print("3. Column names match expected format")
