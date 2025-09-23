# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 09:11:06 2025

@author: amits
"""

# -*- coding: utf-8 -*-
"""
FRTB CSR (Credit Spread Risk) Vega Capital Charge Calculator

This calculator implements the complete FRTB CSR Vega capital charge calculation
following regulatory guidelines with Low/Medium/High correlation scenarios.

Created for CSR Vega capital charge calculation
"""

import pandas as pd
import numpy as np
import math
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class CSRVegaCalculator:
    """
    CSR (Credit Spread Risk) Vega Capital Charge Calculator
    
    This class implements the complete calculation process for CSR Vega capital charges
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
        
        # Constants for CSR Vega calculations
        self.VEGA_TENORS = ['0.5Y', '1Y', '3Y', '5Y', '10Y']
        self.VEGA_RISK_WEIGHT = 1.0  # 100% risk weight for all vega tenors
        self.ALPHA_OPTION_EXPIRY = 0.01  # 1% alpha parameter for option expiry correlation
        
        # Correlation scenario multipliers
        self.HIGH_CORR_MULTIPLIER = 1.25
        self.LOW_CORR_FACTOR_75 = 0.75
        
        # Load all required data from Excel
        self._load_data()
        
    def _load_data(self):
        """Load all required data from Excel sheets"""
        print("Loading data from Excel file...")
        
        try:
            # Load individual bond vegas
            self.raw_data['bond_vegas'] = pd.read_excel(
                self.excel_file, sheet_name='Individual_Bond_Vegas'
            )
            
            # Load delta correlations
            self.raw_data['delta_correlations'] = pd.read_excel(
                self.excel_file, sheet_name='Delta correlations', index_col=0
            )
            
            # Load inter-bucket correlations
            self.raw_data['interbucket_correlations'] = pd.read_excel(
                self.excel_file, sheet_name='CSR non securitiz sectoral corr', index_col=0
            )
            
            print("✓ All data loaded successfully")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def _parse_bucket_notation(self, bucket_str):
        """Parse bucket notation like "1/9" to return list of buckets [1, 9]"""
        if '/' in str(bucket_str):
            return [int(x.strip()) for x in str(bucket_str).split('/')]
        else:
            return [int(bucket_str)]
    
    def _find_interbucket_correlation(self, bucket_b, bucket_c, corr_matrix):
        """Find correlation between two buckets in the correlation matrix"""
        for row_idx in corr_matrix.index:
            for col_idx in corr_matrix.columns:
                row_buckets = self._parse_bucket_notation(row_idx)
                col_buckets = self._parse_bucket_notation(col_idx)
                
                if bucket_b in row_buckets and bucket_c in col_buckets:
                    corr_value = corr_matrix.loc[row_idx, col_idx]
                    if pd.notna(corr_value):
                        return corr_value 
                
                if bucket_b in col_buckets and bucket_c in row_buckets:
                    corr_value = corr_matrix.loc[row_idx, col_idx]
                    if pd.notna(corr_value):
                        return corr_value
        
        return 0.10  # Default 10% if not found
    
    def step1_collect_bond_vegas(self):
        """Step 1: Collect bond vega sensitivities from Individual_Bond_Vegas sheet"""
        print("\nStep 1: Collecting bond vega sensitivities...")
        
        bond_vegas_df = self.raw_data['bond_vegas'].copy()
        
        # Clean column names
        bond_vegas_df.columns = [col.strip() for col in bond_vegas_df.columns]
        
        # Store processed bond vegas
        self.processed_data['bond_vegas'] = bond_vegas_df
        
        # Get unique buckets and securities
        buckets = bond_vegas_df['CSR Bucket'].unique()
        securities = bond_vegas_df['Security'].unique()
        
        self.processed_data['buckets'] = buckets
        self.processed_data['securities'] = securities
        
        print(f"✓ Found {len(bond_vegas_df)} bond vega records")
        print(f"✓ Found buckets: {sorted(buckets)}")
        print(f"✓ Found {len(securities)} unique securities")
        
        return bond_vegas_df
    
    def step2_aggregate_bucket_vegas(self):
        """Step 4: Aggregate vega sensitivities by bucket and tenor"""
        print("\nStep 2: Aggregating vega sensitivities by bucket and tenor...")
        
        bond_vegas_df = self.processed_data['bond_vegas']
        
        # Aggregate vegas by bucket and tenor
        bucket_aggregated_vegas = {}
        
        for bucket in self.processed_data['buckets']:
            bucket_data = bond_vegas_df[bond_vegas_df['CSR Bucket'] == bucket]
            
            # Get sector for this bucket (assuming all securities in bucket have same sector)
            bucket_sector = bucket_data['Sector'].iloc[0]
            
            # Aggregate vega sensitivities across all securities in the bucket
            aggregated_vegas = {}
            for tenor in self.VEGA_TENORS:
                tenor_column = f'Expiry_{tenor}'
                if tenor_column in bucket_data.columns:
                    aggregated_vegas[tenor] = bucket_data[tenor_column].sum()
                else:
                    aggregated_vegas[tenor] = 0.0
            
            bucket_aggregated_vegas[bucket] = {
                'sector': bucket_sector,
                'vegas': aggregated_vegas
            }
            
            print(f"✓ Bucket {bucket} aggregated vegas: {aggregated_vegas}")
        
        self.processed_data['bucket_aggregated_vegas'] = bucket_aggregated_vegas
        
        return bucket_aggregated_vegas
    
    def step3_calculate_weighted_sensitivities(self):
        """Step 6: Calculate weighted sensitivities (multiply by risk weight)"""
        print("\nStep 3: Calculating weighted vega sensitivities...")
        
        weighted_vegas = {}
        
        for bucket, bucket_info in self.processed_data['bucket_aggregated_vegas'].items():
            vegas = bucket_info['vegas']
            sector = bucket_info['sector']
            
            # Apply risk weight (100% = 1.0)
            weighted_vegas_bucket = {}
            for tenor, vega_value in vegas.items():
                weighted_vegas_bucket[tenor] = vega_value * self.VEGA_RISK_WEIGHT
            
            weighted_vegas[bucket] = {
                'sector': sector,
                'weighted_vegas': weighted_vegas_bucket
            }
            
            total_weighted = sum(weighted_vegas_bucket.values())
            print(f"✓ Bucket {bucket} total weighted vega: {total_weighted:.6f}")
        
        self.processed_data['weighted_vegas'] = weighted_vegas
        
        return weighted_vegas
    
    def step4_calculate_option_expiry_correlations(self):
        """Step 7: Calculate option expiry correlations using exponential decay formula"""
        print("\nStep 4: Calculating option expiry correlations...")
        
        # Convert tenor strings to numeric values
        tenor_values = {
            '0.5Y': 0.5,
            '1Y': 1.0,
            '3Y': 3.0,
            '5Y': 5.0,
            '10Y': 10.0
        }
        
        option_expiry_correlations = {}
        
        for tenor_k in self.VEGA_TENORS:
            for tenor_l in self.VEGA_TENORS:
                if tenor_k == tenor_l:
                    correlation = 1.0
                else:
                    Tk = tenor_values[tenor_k]
                    Tl = tenor_values[tenor_l]
                    
                    # Formula: ρ = e^(-α * |Tk - Tl| / min(Tk, Tl))
                    correlation = math.exp(-self.ALPHA_OPTION_EXPIRY * abs(Tk - Tl) / min(Tk, Tl))
                
                option_expiry_correlations[(tenor_k, tenor_l)] = correlation
        
        self.processed_data['option_expiry_correlations'] = option_expiry_correlations
        
        # Print correlation matrix for verification
        print("✓ Option expiry correlation matrix:")
        print("     ", "   ".join(f"{t:>6}" for t in self.VEGA_TENORS))
        for tenor_k in self.VEGA_TENORS:
            row = [f"{option_expiry_correlations[(tenor_k, tenor_l)]:.4f}" for tenor_l in self.VEGA_TENORS]
            print(f"{tenor_k:>4} ", "  ".join(f"{val:>6}" for val in row))
        
        return option_expiry_correlations
    
    def step5_calculate_medium_tenor_correlations(self):
        """Step 8: Calculate medium tenor correlations (delta × option expiry)"""
        print("\nStep 5: Calculating medium tenor correlations...")
        # Add tenor mapping
        tenor_to_numeric = {
            '0.5Y': 0.5,
            '1Y': 1.0,
            '3Y': 3.0,
            '5Y': 5.0,
            '10Y': 10.0
        }
        
        delta_corr_matrix = self.raw_data['delta_correlations']
        option_expiry_corr = self.processed_data['option_expiry_correlations']
        
        medium_tenor_correlations = {}
        
        for tenor_k in self.VEGA_TENORS:
            for tenor_l in self.VEGA_TENORS:
                # Convert string tenors to numeric for delta matrix lookup
                numeric_k = tenor_to_numeric[tenor_k]
                numeric_l = tenor_to_numeric[tenor_l]
                # Get delta correlation
                try:
                    delta_correlation = delta_corr_matrix.loc[numeric_k , numeric_l ] 
                except:
                    delta_correlation = 1.0 if numeric_k  == numeric_l  else 0.65  # Default values
                
                # Get option expiry correlation
                option_expiry_correlation = option_expiry_corr[(tenor_k, tenor_l)]
                
                # Final correlation = delta × option expiry
                final_correlation = delta_correlation * option_expiry_correlation
                
                medium_tenor_correlations[(tenor_k, tenor_l)] = final_correlation
        
        self.processed_data['medium_tenor_correlations'] = medium_tenor_correlations
        
        # Print correlation matrix
        print("✓ Medium tenor correlation matrix:")
        print("     ", "   ".join(f"{t:>6}" for t in self.VEGA_TENORS))
        for tenor_k in self.VEGA_TENORS:
            row = [f"{medium_tenor_correlations[(tenor_k, tenor_l)]:.4f}" for tenor_l in self.VEGA_TENORS]
            print(f"{tenor_k:>4} ", "  ".join(f"{val:>6}" for val in row))
        
        return medium_tenor_correlations
    
    def step6_calculate_high_low_correlations(self):
        """Step 8: Calculate high and low correlation scenarios"""
        print("\nStep 6: Calculating high and low tenor correlations...")
        
        medium_corr = self.processed_data['medium_tenor_correlations']
        
        high_correlations = {}
        low_correlations = {}
        
        for tenor_pair, medium_val in medium_corr.items():
            # High: medium × 1.25, capped at 100%
            high_val = min(medium_val * self.HIGH_CORR_MULTIPLIER, 1.0)
            high_correlations[tenor_pair] = high_val
            
            # Low: max(2×medium - 100%, 75%×medium)
            option1 = 2 * medium_val - 1.0
            option2 = self.LOW_CORR_FACTOR_75 * medium_val
            low_val = max(option1, option2)
            low_correlations[tenor_pair] = low_val
        
        self.processed_data['high_tenor_correlations'] = high_correlations
        self.processed_data['low_tenor_correlations'] = low_correlations
            
        print("✓ High and low tenor correlations calculated")
        
        return high_correlations, low_correlations
    
    def step7_calculate_bucket_capitals(self):
        """Step 9: Calculate bucket capital charges for all correlation scenarios"""
        print("\nStep 7: Calculating bucket capital charges...")
        
        correlation_scenarios = {
            'Low': self.processed_data['low_tenor_correlations'],
            'Medium': self.processed_data['medium_tenor_correlations'],
            'High': self.processed_data['high_tenor_correlations']
        }
        
        bucket_capitals = {}
        
        for scenario_name, tenor_correlations in correlation_scenarios.items():
            scenario_capitals = {}
            
            for bucket, bucket_info in self.processed_data['weighted_vegas'].items():
                weighted_vegas = bucket_info['weighted_vegas']
                
                # Create array of weighted sensitivities
                ws_values = [weighted_vegas[tenor] for tenor in self.VEGA_TENORS]
                
                # Calculate sum of squares
                sum_squares = sum(ws**2 for ws in ws_values)
                
                # Calculate cross products with correlations
                cross_product_sum = 0.0
                for i, tenor_k in enumerate(self.VEGA_TENORS):
                    for j, tenor_l in enumerate(self.VEGA_TENORS):
                        if i != j:  # k ≠ l condition
                            correlation = tenor_correlations[(tenor_k, tenor_l)]
                            cross_product = correlation * ws_values[i] * ws_values[j]
                            cross_product_sum += cross_product
                
                # Apply bucket capital formula
                bucket_capital = math.sqrt(max(0, sum_squares + cross_product_sum))
                scenario_capitals[bucket] = bucket_capital
            
            bucket_capitals[scenario_name] = scenario_capitals
        
        self.processed_data['bucket_capitals'] = bucket_capitals
        
        # Print results
        for scenario in ['Low', 'Medium', 'High']:
            print(f"✓ {scenario} correlation bucket capitals:")
            for bucket, capital in bucket_capitals[scenario].items():
                print(f"  Bucket {bucket}: {capital:.6f}")
        
        return bucket_capitals
    
    def step8_calculate_interbucket_correlations(self):
        """Step 11: Calculate inter-bucket correlations for all scenarios"""
        print("\nStep 8: Calculating inter-bucket correlations...")
        
        buckets = list(self.processed_data['buckets'])
        
        # Get sector for each bucket
        bucket_sectors = {}
        for bucket in buckets:
            bucket_sectors[bucket] = self.processed_data['weighted_vegas'][bucket]['sector']
        
        # Calculate inter-bucket correlations
        interbucket_medium_correlations = {}
        interbucket_corr_matrix = self.raw_data['interbucket_correlations']
        
        correlation_lookup_details = []
        
        for bucket_b in buckets:
            for bucket_c in buckets:
                if bucket_b != bucket_c:
                    # γ_rating = 1 (all are IG)
                    gamma_rating = 1.0
                    
                    # γ_sector: check if same sector or get from correlation matrix
                    sector_b = bucket_sectors[bucket_b]
                    sector_c = bucket_sectors[bucket_c]
                    
                    if sector_b == sector_c:
                        gamma_sector = 1.0
                        lookup_source = "Same Sector"
                    else:
                        gamma_sector = self._find_interbucket_correlation(bucket_b, bucket_c, interbucket_corr_matrix)
                        lookup_source = "Correlation Matrix"
                    
                    # Calculate medium inter-bucket correlation
                    gamma_bc = gamma_rating * gamma_sector
                    interbucket_medium_correlations[(bucket_b, bucket_c)] = gamma_bc
                    
                    correlation_lookup_details.append({
                        'Bucket_B': bucket_b,
                        'Bucket_C': bucket_c,
                        'Sector_B': sector_b,
                        'Sector_C': sector_c,
                        'Gamma_Rating': gamma_rating,
                        'Gamma_Sector': gamma_sector,
                        'Gamma_BC': gamma_bc,
                        'Lookup_Source': lookup_source
                    })
        
        self.processed_data['correlation_lookup_details'] = pd.DataFrame(correlation_lookup_details)
        
        # Calculate high and low scenarios
        interbucket_correlations = {
            'Medium': interbucket_medium_correlations,
            'High': {},
            'Low': {}
        }
        
        for bucket_pair, medium_corr in interbucket_medium_correlations.items():
            # High: γ × 1.25, capped at 100%
            high_corr = min(medium_corr * self.HIGH_CORR_MULTIPLIER, 1.0)
            interbucket_correlations['High'][bucket_pair] = high_corr
            
            # Low: max(2×γ-100%, 75%×γ)
            option1 = 2 * medium_corr - 1.0
            option2 = self.LOW_CORR_FACTOR_75 * medium_corr
            low_corr = max(option1, option2)
            interbucket_correlations['Low'][bucket_pair] = low_corr
        
        self.processed_data['interbucket_correlations'] = interbucket_correlations
        
        print("✓ Inter-bucket correlations calculated for all scenarios")
        
        return interbucket_correlations
    
    def step9_across_bucket_aggregations(self):
        """Step 12: Perform across bucket aggregations for all three scenarios"""
        print("\nStep 9: Performing across bucket aggregations...")
        
        # Calculate bucket aggregate sensitivities
        bucket_aggregate_sensitivities = {}
        
        for bucket, bucket_info in self.processed_data['weighted_vegas'].items():
            weighted_vegas = bucket_info['weighted_vegas']
            s_bucket = sum(weighted_vegas.values())
            bucket_aggregate_sensitivities[bucket] = s_bucket
        
        self.processed_data['bucket_aggregate_sensitivities'] = bucket_aggregate_sensitivities
        
        print("✓ Bucket aggregate sensitivities:")
        for bucket, s_value in bucket_aggregate_sensitivities.items():
            print(f"  S_bucket_{bucket}: {s_value:.6f}")
        
        # Calculate CSR Vega capital charge for each scenario
        final_results = {}
        buckets = list(self.processed_data['buckets'])
        
        for scenario in ['Low', 'Medium', 'High']:
            print(f"\n--- {scenario} Correlation Scenario ---")
            
            # Get scenario-specific values
            bucket_capitals = self.processed_data['bucket_capitals'][scenario]
            interbucket_corrs = self.processed_data['interbucket_correlations'][scenario]
            
            # Initial calculation: Σ K_b² + Σ Σ γbc * S_b * S_c
            sum_kb_squared = sum(k**2 for k in bucket_capitals.values())
            
            cross_bucket_sum = 0.0
            for bucket_b in buckets:
                for bucket_c in buckets:
                    if bucket_b != bucket_c:
                        s_b = bucket_aggregate_sensitivities[bucket_b]
                        s_c = bucket_aggregate_sensitivities[bucket_c]
                        gamma_bc = interbucket_corrs.get((bucket_b, bucket_c), 0.0)
                        cross_bucket_sum += gamma_bc * s_b * s_c
            
            initial_sum = sum_kb_squared + cross_bucket_sum
            
            print(f"Sum of K_b²: {sum_kb_squared:.6f}")
            print(f"Cross bucket sum: {cross_bucket_sum:.6f}")
            print(f"Initial sum: {initial_sum:.6f}")
            
            # Check if negative and apply alternative calculation if needed
            if initial_sum < 0:
                print("Sum is negative, applying alternative calculation...")
                
                # Recalculate S_b using alternative formula
                alternative_s = {}
                for bucket in buckets:
                    original_s = bucket_aggregate_sensitivities[bucket]
                    k_b = bucket_capitals[bucket]
                    
                    # S_b = max[min(Σ WS_k, K_b), -K_b]
                    alternative_s[bucket] = max(min(original_s, k_b), -k_b)
                
                # Recalculate capital charge with alternative S values
                cross_bucket_sum_alt = 0.0
                for bucket_b in buckets:
                    for bucket_c in buckets:
                        if bucket_b != bucket_c:
                            s_b_alt = alternative_s[bucket_b]
                            s_c_alt = alternative_s[bucket_c]
                            gamma_bc = interbucket_corrs.get((bucket_b, bucket_c), 0.0)
                            cross_bucket_sum_alt += gamma_bc * s_b_alt * s_c_alt
                
                final_sum = sum_kb_squared + cross_bucket_sum_alt
                final_capital_charge = math.sqrt(max(0, final_sum))
                
                print(f"Alternative S values: {alternative_s}")
                print(f"Final sum (alternative): {final_sum:.6f}")
                
            else:
                final_capital_charge = math.sqrt(initial_sum)
                print(f"Final capital charge: {final_capital_charge:.6f}")
            
            final_results[scenario] = {
                'initial_sum': initial_sum,
                'final_capital_charge': final_capital_charge,
                'bucket_capitals': bucket_capitals,
                'interbucket_correlations': interbucket_corrs,
                'bucket_aggregate_sensitivities': bucket_aggregate_sensitivities.copy()
            }
        
        self.results['final_calculations'] = final_results
        
        return final_results
    
    def calculate_all_steps(self):
        """Execute all calculation steps in sequence"""
        print("="*70)
        print("CSR VEGA CAPITAL CHARGE CALCULATION")
        print("="*70)
        
        # Execute all steps
        self.step1_collect_bond_vegas()
        self.step2_aggregate_bucket_vegas()
        self.step3_calculate_weighted_sensitivities()
        self.step4_calculate_option_expiry_correlations()
        self.step5_calculate_medium_tenor_correlations()
        self.step6_calculate_high_low_correlations()
        self.step7_calculate_bucket_capitals()
        self.step8_calculate_interbucket_correlations()
        self.step9_across_bucket_aggregations()
        
        print("\n" + "="*70)
        print("CALCULATION COMPLETED SUCCESSFULLY")
        print("="*70)
        
        return self.results
    
    def export_to_excel(self, output_file: str = 'CSR_Vega_Results.xlsx'):
        """Export all results to Excel with comprehensive audit details"""
        print(f"\nExporting results to {output_file}...")
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            
            # Sheet 1: Final Summary
            final_summary = []
            final_summary.append(['CSR Vega Capital Charge Results', '', ''])
            final_summary.append(['', '', ''])
            final_summary.append(['Correlation Scenario', 'Capital Charge', 'Notes'])
            
            for scenario in ['Low', 'Medium', 'High']:
                final_charge = self.results['final_calculations'][scenario]['final_capital_charge']
                final_summary.append([scenario, final_charge, f'{scenario} correlation scenario'])
            
            df_summary = pd.DataFrame(final_summary, columns=['Metric', 'Value', 'Notes'])
            df_summary.to_excel(writer, sheet_name='Final_Summary', index=False)
            
            # Sheet 2: Bucket Aggregated Vegas
            bucket_vegas_data = []
            for bucket, bucket_info in self.processed_data['bucket_aggregated_vegas'].items():
                sector = bucket_info['sector']
                vegas = bucket_info['vegas']
                for tenor, vega_value in vegas.items():
                    bucket_vegas_data.append([bucket, sector, tenor, vega_value])
            
            df_bucket_vegas = pd.DataFrame(bucket_vegas_data, 
                columns=['Bucket', 'Sector', 'Tenor', 'Aggregated_Vega'])
            df_bucket_vegas.to_excel(writer, sheet_name='Bucket_Aggregated_Vegas', index=False)
            
            # Sheet 3: Option Expiry Correlations
            option_expiry_data = []
            for (tenor_k, tenor_l), correlation in self.processed_data['option_expiry_correlations'].items():
                option_expiry_data.append([tenor_k, tenor_l, correlation])
            
            df_option_expiry = pd.DataFrame(option_expiry_data,
                columns=['Tenor_K', 'Tenor_L', 'Option_Expiry_Correlation'])
            df_option_expiry.to_excel(writer, sheet_name='Option_Expiry_Correlations', index=False)
            
            # Sheet 4: Final Tenor Correlations
            tenor_corr_data = []
            for scenario in ['Low', 'Medium', 'High']:
                correlations = self.processed_data[f'{scenario.lower()}_tenor_correlations']
                for (tenor_k, tenor_l), correlation in correlations.items():
                    tenor_corr_data.append([scenario, tenor_k, tenor_l, correlation])
            
            df_tenor_corr = pd.DataFrame(tenor_corr_data,
                columns=['Scenario', 'Tenor_K', 'Tenor_L', 'Final_Correlation'])
            df_tenor_corr.to_excel(writer, sheet_name='Tenor_Correlations', index=False)
            
            # Sheet 5: Bucket Capital Calculations
            bucket_calc_data = []
            for scenario in ['Low', 'Medium', 'High']:
                for bucket, capital in self.processed_data['bucket_capitals'][scenario].items():
                    bucket_calc_data.append([scenario, bucket, capital])
            
            df_bucket_calc = pd.DataFrame(bucket_calc_data,
                columns=['Scenario', 'Bucket', 'Bucket_Capital'])
            df_bucket_calc.to_excel(writer, sheet_name='Bucket_Capital_Calculations', index=False)
            
            # Sheet 6: Inter-Bucket Correlation Lookup
            if hasattr(self.processed_data, 'correlation_lookup_details'):
                self.processed_data['correlation_lookup_details'].to_excel(
                    writer, sheet_name='InterBucket_Corr_Lookup', index=False)
        
        print(f"✓ Results exported successfully to {output_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("FINAL RESULTS SUMMARY")
        print("="*60)
        for scenario in ['Low', 'Medium', 'High']:
            final_charge = self.results['final_calculations'][scenario]['final_capital_charge']
            print(f"{scenario} Correlation CSR Vega Capital Charge: {final_charge:.6f}")
        print("="*60)


def main():
    """Main function to run the CSR Vega calculation"""
    
    # Initialize calculator with Excel file
    excel_file = "frtb_vega_sensitivities and delta correlations.xlsx"  # Update path as needed
    
    try:
        calculator = CSRVegaCalculator(excel_file)
        
        # Perform all calculations
        results = calculator.calculate_all_steps()
        
        # Export results to Excel
        calculator.export_to_excel('CSR_Vega_Capital_Results.xlsx')
        
        return calculator
        
    except FileNotFoundError:
        print(f"Error: Could not find Excel file '{excel_file}'")
        print("Please ensure the file exists and update the file path.")
        return None
    except Exception as e:
        print(f"Error during calculation: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the calculation
    calculator = main()
    
    if calculator:
        print("\nCalculation completed successfully!")
        print("Check 'CSR_Vega_Capital_Results.xlsx' for detailed results.")
        
        # Print validation summary
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        # Show bucket composition
        for bucket in calculator.processed_data['buckets']:
            bucket_info = calculator.processed_data['bucket_aggregated_vegas'][bucket]
            vegas = bucket_info['vegas']
            total_vega = sum(vegas.values())
            print(f"Bucket {bucket}: Total aggregated vega = {total_vega:.6f}")
        
        print("="*60)
