# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 21:11:13 2025

@author: amits
"""

import pandas as pd
import numpy as np
# from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class GIRRDeltaCalculator:
    """
    GIRR (General Interest Rate Risk) Delta Capital Charge Calculator
    
    This class implements the complete calculation process for GIRR Delta capital charges
    following regulatory guidelines with Low/Medium/High correlation scenarios.
    """
    
    def __init__(self, excel_file_path: str):
        """
        Initialize the calculator with data from Excel file
        
        Args:
            excel_file_path: Path to the Excel file containing all required sheets
        """
        self.excel_file = excel_file_path
        self.raw_data = {}
        self.processed_data = {}
        self.results = {}
        
        # Load all required data from Excel
        self._load_data()
        
    def _load_data(self):
        """Load all required data from Excel sheets"""
        print("Loading data from Excel file...")
        
        try:
            # Load sensitivities data
            self.raw_data['sensitivities'] = pd.read_excel(
                self.excel_file, sheet_name='GIRR Sensitivities'
            )
            
            # Load risk weights
            self.raw_data['weights'] = pd.read_excel(
                self.excel_file, sheet_name='GIRR weights'
            )
            
            # Load correlations matrix
            self.raw_data['correlations'] = pd.read_excel(
                self.excel_file, sheet_name='Correlations', index_col=0
            )
            
            # Load miscellaneous weights (inter-bucket correlation)
            self.raw_data['misc_weights'] = pd.read_excel(
                self.excel_file, sheet_name='Misc Weights'
            )
            
            print("✓ All data loaded successfully")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def step1_collect_sensitivities(self):
        """
        Step 1: Collect sensitivities from GIRR Sensitivities sheet
        Expected columns: Currency (B), Tenor (C), Sensitivity (D)
        """
        print("\nStep 1: Collecting sensitivities...")
        
        sens_df = self.raw_data['sensitivities'].copy()
        
        # Clean column names and ensure proper structure
        sens_df.columns = [col.strip() for col in sens_df.columns]
        
        # Store processed sensitivities
        self.processed_data['sensitivities'] = sens_df
        
        # Get unique currencies for bucket creation
        currencies = sens_df.iloc[:, 1].unique()  # Column B (Currency)
        self.processed_data['currencies'] = currencies
        
        print(f"✓ Found {len(sens_df)} sensitivity records")
        print(f"✓ Found currencies: {currencies}")
        
        return sens_df
    
    def step2_create_currency_buckets(self):
        """
        Step 2: Create currency buckets with tenors
        Group all tenors by currency to create distinct currency buckets
        """
        print("\nStep 2: Creating currency buckets...")
        
        sens_df = self.processed_data['sensitivities']
        
        # Group by currency (Column B)
        currency_buckets = {}
        for currency in self.processed_data['currencies']:
            bucket_data = sens_df[sens_df.iloc[:, 1] == currency].copy()
            currency_buckets[currency] = bucket_data
            
        self.processed_data['currency_buckets'] = currency_buckets
        
        for currency, data in currency_buckets.items():
            print(f"✓ {currency} bucket: {len(data)} tenor records")
            
        return currency_buckets
    
    def step3_collect_risk_weights(self):
        """
        Step 3: Collect risk weights from GIRR weights sheet
        Convert from percentage to decimal by multiplying by 0.01
        """
        print("\nStep 3: Collecting risk weights...")
        
        weights_df = self.raw_data['weights'].copy()
        
        # Convert percentages to decimals (multiply by 0.01)
        weight_cols = weights_df.select_dtypes(include=[np.number]).columns
        for col in weight_cols:
            if 'tenor' not in col:  # Don't convert tenor numbers
                weights_df[col] = weights_df[col] * 0.01
        
        self.processed_data['risk_weights'] = weights_df
        
        print(f"✓ Risk weights loaded and converted to decimals")
        
        return weights_df
    
    def step4_calculate_weighted_sensitivities(self):
        """
        Step 4: Calculate weighted sensitivities
        Formula: WS_k = RW_k * S_k
        """
        print("\nStep 4: Calculating weighted sensitivities...")
        
        weighted_sensitivities = {}
        
        for currency, bucket_data in self.processed_data['currency_buckets'].items():
            bucket_ws = bucket_data.copy()
            
            # For each record in the bucket, find corresponding risk weight and calculate WS
            bucket_ws['Weighted_Sensitivity'] = 0.0
            
            for idx, row in bucket_ws.iterrows():
                tenor = row.iloc[2]  # Column C (Tenor)
                sensitivity = row.iloc[3]  # Column D (Sensitivity)
                
                # Find risk weight for this tenor
                weights_df = self.processed_data['risk_weights']
                tenor_col = weights_df.columns[0]  # Assuming first column is tenor
                weight_col = weights_df.columns[1]  # Assuming second column is weight
                
                risk_weight = weights_df[weights_df[tenor_col] == tenor][weight_col].values
                if len(risk_weight) > 0:
                    ws = sensitivity * risk_weight[0]
                    bucket_ws.loc[idx, 'Weighted_Sensitivity'] = ws
            
            weighted_sensitivities[currency] = bucket_ws
            
        self.processed_data['weighted_sensitivities'] = weighted_sensitivities
        
        for currency, data in weighted_sensitivities.items():
            total_ws = data['Weighted_Sensitivity'].sum()
            print(f"✓ {currency} total weighted sensitivity: {total_ws:.6f}")
        
        return weighted_sensitivities
    
    def step5_medium_correlations(self):
        """
        Step 5: Collect base (medium) correlations between tenors
        """
        print("\nStep 5: Setting up medium tenor correlations...")
        
        # The correlations sheet should contain the base correlation matrix
        corr_matrix = self.raw_data['correlations'].copy()
        
        self.processed_data['medium_correlations'] = corr_matrix
        
        print(f"✓ Medium correlation matrix loaded: {corr_matrix.shape}")
        
        return corr_matrix
    
    def step6_high_correlations(self):
        """
        Step 6: Calculate high correlations
        Formula: ρ_high = min(ρ_medium * 1.25, 1.0)  # Capped at 100%
        """
        print("\nStep 6: Calculating high tenor correlations...")
        
        medium_corr = self.processed_data['medium_correlations']
        high_corr = (medium_corr * 1.25).clip(upper=1.0)
        
        self.processed_data['high_correlations'] = high_corr
        
        print("✓ High correlations calculated (medium × 1.25, capped at 100%)")
        
        return high_corr
    
    def step7_low_correlations(self):
        """
        Step 7: Calculate low correlations
        Formula: ρ_low = max(2 × ρ_medium - 100%, 75% × ρ_medium)
        """
        print("\nStep 7: Calculating low tenor correlations...")
        
        medium_corr = self.processed_data['medium_correlations']
        
        # Apply the formula: max(2*ρ - 1.0, 0.75*ρ)
        option1 = 2 * medium_corr - 1.0
        option2 = 0.75 * medium_corr
        low_corr = np.maximum(option1, option2)
        
        self.processed_data['low_correlations'] = low_corr
        
        print("✓ Low correlations calculated using max(2×ρ-100%, 75%×ρ)")
        
        return low_corr
    
    def step8_calculate_bucket_capitals(self):
        """
        Step 8: Calculate bucket capitals for all three correlation scenarios
        Formula: K_b = √max(0, Σ(WS_k²) + ΣΣ(ρ_kl × WS_k × WS_l))
        """
        print("\nStep 8: Calculating bucket capitals...")
        
        correlation_scenarios = {
            'Low': self.processed_data['low_correlations'],
            'Medium': self.processed_data['medium_correlations'],
            'High': self.processed_data['high_correlations']
        }
        
        bucket_capitals = {}
        
        for scenario_name, corr_matrix in correlation_scenarios.items():
            scenario_capitals = {}
            
            for currency, bucket_data in self.processed_data['weighted_sensitivities'].items():
                # Get weighted sensitivities for this bucket
                # Aggregate weighted sensitivities by tenor
                tenor_ws_df = pd.DataFrame({'Tenor': bucket_data.iloc[:, 2].values, 'WS': bucket_data['Weighted_Sensitivity'].values}).groupby('Tenor')['WS'].sum()
                ws_values = tenor_ws_df.values
                tenors = tenor_ws_df.index.values
                
                # Calculate sum of squares
                sum_squares = np.sum(ws_values**2)
                
                # Calculate cross products with correlations
                cross_product_sum = 0.0
                
                for i, tenor_i in enumerate(tenors):
                    for j, tenor_j in enumerate(tenors):
                        if i != j:  # k ≠ l condition
                            # Find correlation between tenor_i and tenor_j
                            try:
                                correlation = corr_matrix.loc[tenor_i, tenor_j]
                                cross_product = correlation * ws_values[i] * ws_values[j]
                                cross_product_sum += cross_product
                            except:
                                # If correlation not found, assume 0
                                pass
                
                # Apply the bucket capital formula
                bucket_capital = np.sqrt(max(0, sum_squares + cross_product_sum))
                scenario_capitals[currency] = bucket_capital
                
            bucket_capitals[scenario_name] = scenario_capitals
        
        self.processed_data['bucket_capitals'] = bucket_capitals
        
        # Print results
        for scenario in ['Low', 'Medium', 'High']:
            print(f"✓ {scenario} correlation bucket capitals:")
            for currency, capital in bucket_capitals[scenario].items():
                print(f"  {currency}: {capital:.6f}")
        
        return bucket_capitals
    
    def step9_inter_bucket_correlations(self):
        """
        Step 9: Calculate inter-bucket correlations for all scenarios
        """
        print("\nStep 9: Calculating inter-bucket correlations...")
        
        # Get base inter-bucket correlation from Misc Weights sheet
        misc_df = self.raw_data['misc_weights']
        
        # Find γbc value (should be in % so convert to decimal)
        gamma_bc_base = None
        # Get base inter-bucket correlation from Misc Weights sheet (row 2, value 50)
        gamma_bc_base = misc_df.iloc[2, 1] * 0.01  # Row 2, column 2 (Correlation% column), convert from % to decimal
        
        # Calculate three scenarios
        inter_bucket_correlations = {
            'Medium': gamma_bc_base,
            'High': min(gamma_bc_base * 1.25, 1.0),  # Capped at 100%
            'Low': max(2 * gamma_bc_base - 1.0, 0.75 * gamma_bc_base)
        }
        
        self.processed_data['inter_bucket_correlations'] = inter_bucket_correlations
        
        print("✓ Inter-bucket correlations calculated:")
        for scenario, correlation in inter_bucket_correlations.items():
            print(f"  {scenario}: {correlation:.6f}")
        
        return inter_bucket_correlations
    
    def step10_across_bucket_aggregations(self):
        """
        Step 10: Perform across bucket aggregations for all three scenarios
        """
        print("\nStep 10: Performing across bucket aggregations...")
        
        # Step 10a: Calculate bucket aggregate sensitivities (same for all scenarios)
        bucket_aggregate_sensitivities = {}
        
        for currency, bucket_data in self.processed_data['weighted_sensitivities'].items():
            s_bucket = bucket_data['Weighted_Sensitivity'].sum()
            bucket_aggregate_sensitivities[currency] = s_bucket
        
        self.processed_data['bucket_aggregate_sensitivities'] = bucket_aggregate_sensitivities
        
        print("✓ Bucket aggregate sensitivities:")
        for currency, s_value in bucket_aggregate_sensitivities.items():
            print(f"  S_{currency}: {s_value:.6f}")
        
        # Step 10b & 10c: Calculate GIRR Delta capital charge for each scenario
        final_results = {}
        currencies = list(self.processed_data['currencies'])
        
        for scenario in ['Low', 'Medium', 'High']:
            print(f"\n--- {scenario} Correlation Scenario ---")
            
            # Get scenario-specific values
            bucket_capitals = self.processed_data['bucket_capitals'][scenario]
            gamma_bc = self.processed_data['inter_bucket_correlations'][scenario]
            
            # Initial calculation
            # Σ K_b² + Σ Σ γbc * S_b * S_c
            sum_kb_squared = sum(k**2 for k in bucket_capitals.values())
            
            cross_bucket_sum = 0.0
            for i, curr_b in enumerate(currencies):
                for j, curr_c in enumerate(currencies):
                    if i != j:  # b ≠ c condition
                        s_b = bucket_aggregate_sensitivities[curr_b]
                        s_c = bucket_aggregate_sensitivities[curr_c]
                        cross_bucket_sum += gamma_bc * s_b * s_c
            
            initial_capital_charge = (sum_kb_squared + cross_bucket_sum)**0.5
            
            print(f"Initial capital charge: {initial_capital_charge:.6f}")
            
            # Check if negative and apply alternative calculation if needed
            if initial_capital_charge < 0:
                print("Capital charge is negative, applying alternative calculation...")
                
                # Recalculate S_b using alternative formula
                alternative_s = {}
                for currency in currencies:
                    original_s = bucket_aggregate_sensitivities[currency]
                    k_b = bucket_capitals[currency]
                    
                    # S_b = max[min(Σ WS_k, K_b), -K_b]
                    alternative_s[currency] = max(min(original_s, k_b), -k_b)
                
                # Recalculate capital charge with alternative S values
                cross_bucket_sum_alt = 0.0
                for i, curr_b in enumerate(currencies):
                    for j, curr_c in enumerate(currencies):
                        if i != j:
                            s_b_alt = alternative_s[curr_b]
                            s_c_alt = alternative_s[curr_c]
                            cross_bucket_sum_alt += gamma_bc * s_b_alt * s_c_alt
                
                final_capital_charge = sum_kb_squared + cross_bucket_sum_alt
                
                print(f"Alternative S values: {alternative_s}")
                print(f"Final capital charge (alternative): {final_capital_charge:.6f}")
                
            else:
                final_capital_charge = initial_capital_charge
                print(f"Final capital charge: {final_capital_charge:.6f}")
            
            final_results[scenario] = {
                'initial_capital_charge': initial_capital_charge,
                'final_capital_charge': final_capital_charge,
                'bucket_capitals': bucket_capitals,
                'inter_bucket_correlation': gamma_bc,
                'bucket_aggregate_sensitivities': bucket_aggregate_sensitivities.copy()
            }
        
        self.results['final_calculations'] = final_results
        
        return final_results
    
    def calculate_all_steps(self):
        """Execute all calculation steps in sequence"""
        print("="*60)
        print("GIRR DELTA CAPITAL CHARGE CALCULATION")
        print("="*60)
        
        # Execute all steps
        self.step1_collect_sensitivities()
        self.step2_create_currency_buckets()
        self.step3_collect_risk_weights()
        self.step4_calculate_weighted_sensitivities()
        self.step5_medium_correlations()
        self.step6_high_correlations()
        self.step7_low_correlations()
        self.step8_calculate_bucket_capitals()
        self.step9_inter_bucket_correlations()
        self.step10_across_bucket_aggregations()
        
        print("\n" + "="*60)
        print("CALCULATION COMPLETED SUCCESSFULLY")
        print("="*60)
        
        return self.results
    
    def export_to_excel(self, output_file: str = 'GIRR_Delta_Results.xlsx'):
        """
        Step 11: Export all results to Excel with specified sheet structure
        """
        print(f"\nStep 11: Exporting results to {output_file}...")
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            
            # Sheet 1-3: Detailed calculations for each scenario
            for scenario in ['Low', 'Medium', 'High']:
                scenario_data = []
                
                # Add bucket capitals
                scenario_data.append(['Bucket Capitals', '', ''])
                scenario_data.append(['Currency', 'Bucket Capital', ''])
                bucket_caps = self.processed_data['bucket_capitals'][scenario]
                for currency, capital in bucket_caps.items():
                    scenario_data.append([currency, capital, ''])
                
                scenario_data.append(['', '', ''])  # Empty row
                
                # Add inter-bucket correlation
                scenario_data.append(['Inter-bucket Correlation', '', ''])
                gamma = self.processed_data['inter_bucket_correlations'][scenario]
                scenario_data.append(['γbc', gamma, ''])
                
                scenario_data.append(['', '', ''])  # Empty row
                
                # Add final capital charge
                scenario_data.append(['Final Results', '', ''])
                final_charge = self.results['final_calculations'][scenario]['final_capital_charge']
                scenario_data.append(['GIRR Delta Capital Charge', final_charge, ''])
                
                df_scenario = pd.DataFrame(scenario_data, columns=['Metric', 'Value', 'Notes'])
                df_scenario.to_excel(writer, sheet_name=f'{scenario}_Correlation', index=False)
            
            # Sheet 4: Inter-bucket correlations
            inter_bucket_data = []
            for scenario, correlation in self.processed_data['inter_bucket_correlations'].items():
                inter_bucket_data.append([scenario, correlation])
            
            df_inter_bucket = pd.DataFrame(inter_bucket_data, columns=['Correlation Scenario', 'γbc Value'])
            df_inter_bucket.to_excel(writer, sheet_name='Inter_Bucket_Correlations', index=False)
            
            # Sheet 5: Bucket capital charges
            bucket_capital_data = []
            for scenario in ['Low', 'Medium', 'High']:
                for currency, capital in self.processed_data['bucket_capitals'][scenario].items():
                    bucket_capital_data.append([scenario, currency, capital])
            
            df_bucket_capitals = pd.DataFrame(
                bucket_capital_data, 
                columns=['Correlation Scenario', 'Currency', 'Bucket Capital']
            )
            df_bucket_capitals.to_excel(writer, sheet_name='Bucket_Capital_Charge', index=False)
            
            # Sheet 6: Delta GIRR Capital charges
            delta_capital_data = []
            for scenario in ['Low', 'Medium', 'High']:
                final_charge = self.results['final_calculations'][scenario]['final_capital_charge']
                delta_capital_data.append([scenario, 'USD', final_charge])
            
            df_delta_capital = pd.DataFrame(
                delta_capital_data,
                columns=['Correlation Scenario', 'Currency', 'Delta GIRR Capital Charge']
            )
            df_delta_capital.to_excel(writer, sheet_name='Delta_GIRR_Capital_Charge', index=False)
        
        print(f"✓ Results exported successfully to {output_file}")
        
        # Print summary
        print("\n" + "="*50)
        print("FINAL RESULTS SUMMARY")
        print("="*50)
        for scenario in ['Low', 'Medium', 'High']:
            final_charge = self.results['final_calculations'][scenario]['final_capital_charge']
            print(f"{scenario} Correlation GIRR Delta Capital Charge: {final_charge:.6f}")
        print("="*50)


def main():
    """
    Main function to run the GIRR Delta calculation
    Usage: Ensure 'Sensitivities and weights.xlsx' is in the same directory
    """
    
    # Initialize calculator with Excel file
    excel_file = "Sensitivities and weights.xlsx"
    
    try:
        calculator = GIRRDeltaCalculator(excel_file)
        
        # Perform all calculations
        results = calculator.calculate_all_steps()
        
        # Export results to Excel
        calculator.export_to_excel('GIRR_Delta_Capital_Results.xlsx')
        
        return calculator
        
    except FileNotFoundError:
        print(f"Error: Could not find Excel file '{excel_file}'")
        print("Please ensure the file exists in the current directory.")
        return None
    except Exception as e:
        print(f"Error during calculation: {e}")
        return None


if __name__ == "__main__":
    # Run the calculation
    calculator = main()
    
    # If successful, the calculator object contains all intermediate results
    # for debugging and validation
    if calculator:
        print("\nCalculation completed successfully!")
        print("Check 'GIRR_Delta_Capital_Results.xlsx' for detailed results.")
