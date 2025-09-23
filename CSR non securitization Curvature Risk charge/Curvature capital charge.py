# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 19:49:58 2025

@author: amits
"""

# -*- coding: utf-8 -*-
"""
FRTB CSR (Credit Spread Risk) Curvature Capital Charge Calculator

This calculator implements the complete FRTB CSR Curvature capital charge calculation
following regulatory guidelines with Low/Medium/High correlation scenarios.

Created for CSR Curvature capital charge calculation
"""

import pandas as pd
import numpy as np
import math
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class CSRCurvatureCalculator:
    """
    CSR (Credit Spread Risk) Curvature Capital Charge Calculator
    
    This class implements the complete calculation process for CSR Curvature capital charges
    following regulatory guidelines with Low/Medium/High correlation scenarios.
    """
    
    def __init__(self, delta_file_path: str, curvature_file_path: str):
        """
        Initialize the calculator with data from Excel files
        
        Args:
            delta_file_path: Path to the bond delta sensitivities Excel file
            curvature_file_path: Path to the curvature parallel shifts Excel file
        """
        self.delta_file = delta_file_path
        self.curvature_file = curvature_file_path
        self.raw_data = {}
        self.processed_data = {}
        self.results = {}
        
        # Constants for CSR Curvature calculations
        self.CURVATURE_RISK_WEIGHT = 0.12  # 12% maximum risk weight from bucket 11
        
        # Correlation scenario multipliers
        self.HIGH_CORR_MULTIPLIER = 1.25
        self.LOW_CORR_FACTOR_75 = 0.75
        
        # Load all required data from Excel files
        self._load_data()
        
    def _load_data(self):
        """Load all required data from Excel sheets"""
        print("Loading data from Excel files...")
        
        try:
            # Load delta sensitivities data
            self.raw_data['delta_sensitivities'] = pd.read_excel(
                self.delta_file, sheet_name='Bond delta Sensitivities'
            )
            
            # Load curvature data
            self.raw_data['curvature_data'] = pd.read_excel(
                self.curvature_file, sheet_name='CSR_Non_Sec_Curvature_Data'
            )
            
            # Load inter-bucket correlations from curvature file
            self.raw_data['interbucket_correlations'] = pd.read_excel(
                self.curvature_file, sheet_name='CSR non securitiz sectoral corr', index_col=0
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
    
    def step1_collect_delta_sensitivities(self):
        """Step 1: Collect delta sensitivities from Bond delta Sensitivities sheet"""
        print("\nStep 1: Collecting delta sensitivities...")
        
        delta_df = self.raw_data['delta_sensitivities'].copy()
        
        # Clean column names
        delta_df.columns = [col.strip() for col in delta_df.columns]
        
        # Store processed delta sensitivities
        self.processed_data['delta_sensitivities'] = delta_df
        
        # Get unique buckets and securities
        buckets = delta_df['CSR Bucket'].unique()
        securities = delta_df['Security'].unique()
        
        self.processed_data['buckets'] = buckets
        self.processed_data['securities'] = securities
        
        print(f"✓ Found {len(delta_df)} delta sensitivity records")
        print(f"✓ Found buckets: {sorted(buckets)}")
        print(f"✓ Found {len(securities)} unique securities")
        
        return delta_df
    
    def step2_aggregate_bucket_delta_sensitivities(self):
        """Step 2: Aggregate delta sensitivities by CSR bucket (sum across all tenors)"""
        print("\nStep 2: Aggregating bucket delta sensitivities...")
        
        delta_df = self.processed_data['delta_sensitivities']
        
        # Aggregate delta sensitivities by bucket (sum all tenors and securities within bucket)
        bucket_delta_sensitivities = {}
        
        for bucket in self.processed_data['buckets']:
            bucket_data = delta_df[delta_df['CSR Bucket'] == bucket]
            
            # Sum all sensitivities for this bucket (across all securities and tenors)
            total_sensitivity = bucket_data['Sensitivity'].sum()
            bucket_delta_sensitivities[bucket] = total_sensitivity
            
            print(f"✓ Bucket {bucket} total delta sensitivity: {total_sensitivity:.6f}")
        
        self.processed_data['bucket_delta_sensitivities'] = bucket_delta_sensitivities
        
        return bucket_delta_sensitivities
    
    def step3_collect_curvature_data(self):
        """Step 3: Collect curvature data for each position"""
        print("\nStep 3: Collecting curvature data...")
        
        curvature_df = self.raw_data['curvature_data'].copy()
        
        # Clean column names
        curvature_df.columns = [col.strip() for col in curvature_df.columns]
        
        # Calculate position-level values
        curvature_positions = []
        
        for _, row in curvature_df.iterrows():
            security = row['Security']
            csr_bucket = row['CSR Bucket']
            
            # Step 3c: Position size = Notional_Exposure / 100
            position_size = row['Notional_Exposure'] / 100.0
            
            # Step 3d: V_base = Base_Price * position_size
            v_base = row['Base_Price'] * position_size
            
            # Step 3e: V_RW_up = Price_Up * position_size
            v_rw_up = row['Price_Up'] * position_size
            
            # Step 3f: V_RW_down = Price_Down * position_size
            v_rw_down = row['Price_Down'] * position_size
            
            curvature_positions.append({
                'Security': security,
                'CSR_Bucket': csr_bucket,
                'Position_Size': position_size,
                'V_base': v_base,
                'V_RW_up': v_rw_up,
                'V_RW_down': v_rw_down
            })
        
        curvature_positions_df = pd.DataFrame(curvature_positions)
        self.processed_data['curvature_positions'] = curvature_positions_df
        
        print(f"✓ Processed {len(curvature_positions)} curvature positions")
        
        return curvature_positions_df
    
    def step4_aggregate_bucket_curvature_values(self):
        """Step 5: Aggregate V_base, V_RW_up, V_RW_down by bucket"""
        print("\nStep 4: Aggregating curvature values by bucket...")
        
        curvature_df = self.processed_data['curvature_positions']
        
        # Aggregate by bucket
        bucket_curvature_values = {}
        
        for bucket in self.processed_data['buckets']:
            bucket_data = curvature_df[curvature_df['CSR_Bucket'] == bucket]
            
            if len(bucket_data) > 0:
                v_base_agg = bucket_data['V_base'].sum()
                v_rw_up_agg = bucket_data['V_RW_up'].sum()
                v_rw_down_agg = bucket_data['V_RW_down'].sum()
                
                bucket_curvature_values[bucket] = {
                    'V_base': v_base_agg,
                    'V_RW_up': v_rw_up_agg,
                    'V_RW_down': v_rw_down_agg
                }
                
                print(f"✓ Bucket {bucket}: V_base={v_base_agg:.2f}, V_RW_up={v_rw_up_agg:.2f}, V_RW_down={v_rw_down_agg:.2f}")
            else:
                print(f"⚠ Warning: No curvature data found for bucket {bucket}")
        
        self.processed_data['bucket_curvature_values'] = bucket_curvature_values
        
        return bucket_curvature_values
    
    def step5_calculate_bucket_curvatures(self):
        """Step 6: Calculate Curvature_Up and Curvature_Down for each bucket"""
        print("\nStep 5: Calculating bucket curvatures...")
        
        bucket_curvature_values = self.processed_data['bucket_curvature_values']
        bucket_delta_sensitivities = self.processed_data['bucket_delta_sensitivities']
        
        bucket_curvatures = {}
        
        for bucket in self.processed_data['buckets']:
            if bucket in bucket_curvature_values and bucket in bucket_delta_sensitivities:
                values = bucket_curvature_values[bucket]
                delta_sensitivity = bucket_delta_sensitivities[bucket]
                
                # Step 6a: Curvature_Up = -(V_RW_up - V_base - 0.12 * Sum_of_sensitivity)
                curvature_up = -(values['V_RW_up'] - values['V_base'] - 
                               self.CURVATURE_RISK_WEIGHT * delta_sensitivity)
                
                # Step 6b: Curvature_Down = -(V_RW_down - V_base + 0.12 * Sum_of_sensitivity)
                curvature_down = -(values['V_RW_down'] - values['V_base'] + 
                                 self.CURVATURE_RISK_WEIGHT * delta_sensitivity)
                
                bucket_curvatures[bucket] = {
                    'Curvature_Up': curvature_up,
                    'Curvature_Down': curvature_down
                }
                
                print(f"✓ Bucket {bucket}: Curvature_Up={curvature_up:.6f}, Curvature_Down={curvature_down:.6f}")
            else:
                print(f"⚠ Warning: Missing data for bucket {bucket}")
        
        self.processed_data['bucket_curvatures'] = bucket_curvatures
        
        return bucket_curvatures
    
    def step6_calculate_bucket_capitals(self):
        """Step 8: Calculate bucket capital charges K_b = max(Curvature_Up, Curvature_Down)"""
        print("\nStep 6: Calculating bucket capital charges...")
        
        bucket_curvatures = self.processed_data['bucket_curvatures']
        bucket_capitals = {}
        
        for bucket, curvatures in bucket_curvatures.items():
            # K_b = max(Curvature_Up, Curvature_Down)
            k_b = max(curvatures['Curvature_Up'], curvatures['Curvature_Down'])
            bucket_capitals[bucket] = k_b
            
            print(f"✓ Bucket {bucket}: K_b = {k_b:.6f}")
        
        self.processed_data['bucket_capitals'] = bucket_capitals
        
        return bucket_capitals
    
    def step7_calculate_psi_function(self):
        """Step 9: Calculate ψ(K_b, K_c) for each bucket pair"""
        print("\nStep 7: Calculating ψ function for bucket pairs...")
        
        bucket_capitals = self.processed_data['bucket_capitals']
        buckets = list(self.processed_data['buckets'])
        
        psi_values = {}
        
        for bucket_b in buckets:
            for bucket_c in buckets:
                if bucket_b != bucket_c:
                    k_b = bucket_capitals.get(bucket_b, 0)
                    k_c = bucket_capitals.get(bucket_c, 0)
                    
                    # ψ(K_b, K_c) = 0 if both K_b and K_c are negative, 1 otherwise
                    if k_b < 0 and k_c < 0:
                        psi_value = 0
                    else:
                        psi_value = 1
                    
                    psi_values[(bucket_b, bucket_c)] = psi_value
        
        self.processed_data['psi_values'] = psi_values
        
        print(f"✓ Calculated ψ function for {len(psi_values)} bucket pairs")
        
        return psi_values
    
    def step8_calculate_interbucket_correlations(self):
        """Step 10: Calculate inter-bucket correlations with squaring"""
        print("\nStep 8: Calculating inter-bucket correlations...")
        
        buckets = list(self.processed_data['buckets'])
        
        # Get sector for each bucket from delta sensitivities file
        delta_df = self.processed_data['delta_sensitivities']
        bucket_sectors = {}
        for bucket in buckets:
            bucket_data = delta_df[delta_df['CSR Bucket'] == bucket]
            if len(bucket_data) > 0:
                bucket_sectors[bucket] = bucket_data['Sector'].iloc[0]
        
        # Calculate inter-bucket correlations
        interbucket_medium_correlations = {}
        interbucket_corr_matrix = self.raw_data['interbucket_correlations']
        
        correlation_lookup_details = []
        
        for bucket_b in buckets:
            for bucket_c in buckets:
                if bucket_b != bucket_c:
                    # Step 10a: γ_rating = 1 (all are IG)
                    gamma_rating = 1.0
                    
                    # Step 10b: γ_sector: check if same sector or get from correlation matrix
                    sector_b = bucket_sectors.get(bucket_b, "Unknown")
                    sector_c = bucket_sectors.get(bucket_c, "Unknown")
                    
                    if sector_b == sector_c:
                        gamma_sector = 1.0
                        lookup_source = "Same Sector"
                    else:
                        gamma_sector = self._find_interbucket_correlation(bucket_b, bucket_c, interbucket_corr_matrix)
                        lookup_source = "Correlation Matrix"
                    
                    # Step 10c: Calculate medium inter-bucket correlation
                    gamma_bc = gamma_rating * gamma_sector
                    
                    # Step 10d: Square γ_bc
                    gamma_bc_squared = gamma_bc * gamma_bc
                    
                    interbucket_medium_correlations[(bucket_b, bucket_c)] = gamma_bc_squared
                    
                    correlation_lookup_details.append({
                        'Bucket_B': bucket_b,
                        'Bucket_C': bucket_c,
                        'Sector_B': sector_b,
                        'Sector_C': sector_c,
                        'Gamma_Rating': gamma_rating,
                        'Gamma_Sector': gamma_sector,
                        'Gamma_BC': gamma_bc,
                        'Gamma_BC_Squared': gamma_bc_squared,
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
            # Step 10e: High: γ × 1.25, capped at 100%
            high_corr = min(medium_corr * self.HIGH_CORR_MULTIPLIER, 1.0)
            interbucket_correlations['High'][bucket_pair] = high_corr
            
            # Step 10f: Low: max(2×γ-100%, 75%×γ)
            option1 = 2 * medium_corr - 1.0
            option2 = self.LOW_CORR_FACTOR_75 * medium_corr
            low_corr = max(option1, option2)
            interbucket_correlations['Low'][bucket_pair] = low_corr
        
        self.processed_data['interbucket_correlations'] = interbucket_correlations
        
        print("✓ Inter-bucket correlations calculated with squaring applied")
        
        return interbucket_correlations
    
    def step9_calculate_final_curvature_charges(self):
        """Step 11: Calculate final curvature capital charges for all scenarios"""
        print("\nStep 9: Calculating final curvature capital charges...")
        
        buckets = list(self.processed_data['buckets'])
        bucket_capitals = self.processed_data['bucket_capitals']
        psi_values = self.processed_data['psi_values']
        interbucket_correlations = self.processed_data['interbucket_correlations']
        
        final_results = {}
        
        for scenario in ['Low', 'Medium', 'High']:
            print(f"\n--- {scenario} Correlation Scenario ---")
            
            # Get scenario-specific inter-bucket correlations
            gamma_bc_values = interbucket_correlations[scenario]
            
            # Calculate sum of squared bucket capitals
            sum_kb_squared = sum(k**2 for k in bucket_capitals.values())
            
            # Calculate cross-bucket terms with ψ function
            cross_bucket_sum = 0.0
            
            for bucket_b in buckets:
                for bucket_c in buckets:
                    if bucket_b != bucket_c:  # c ≠ b condition
                        k_b = bucket_capitals.get(bucket_b, 0)
                        k_c = bucket_capitals.get(bucket_c, 0)
                        gamma_bc = gamma_bc_values.get((bucket_b, bucket_c), 0.0)
                        psi_bc = psi_values.get((bucket_b, bucket_c), 1)
                        
                        cross_term = gamma_bc * psi_bc * k_b * k_c
                        cross_bucket_sum += cross_term
            
            # Apply final formula: √max(0, Σ K_b² + Σ Σ γ_bc ψ(K_b,K_c) × K_b × K_c)
            total_sum = sum_kb_squared + cross_bucket_sum
            final_curvature_charge = math.sqrt(max(0, total_sum))
            
            print(f"Sum of K_b²: {sum_kb_squared:.6f}")
            print(f"Cross bucket sum: {cross_bucket_sum:.6f}")
            print(f"Total sum: {total_sum:.6f}")
            print(f"Final curvature charge: {final_curvature_charge:.6f}")
            
            final_results[scenario] = {
                'sum_kb_squared': sum_kb_squared,
                'cross_bucket_sum': cross_bucket_sum,
                'total_sum': total_sum,
                'final_curvature_charge': final_curvature_charge,
                'bucket_capitals': bucket_capitals.copy(),
                'interbucket_correlations': gamma_bc_values.copy()
            }
        
        self.results['final_calculations'] = final_results
        
        return final_results
    
    def calculate_all_steps(self):
        """Execute all calculation steps in sequence"""
        print("="*70)
        print("CSR CURVATURE CAPITAL CHARGE CALCULATION")
        print("="*70)
        
        # Execute all steps
        self.step1_collect_delta_sensitivities()
        self.step2_aggregate_bucket_delta_sensitivities()
        self.step3_collect_curvature_data()
        self.step4_aggregate_bucket_curvature_values()
        self.step5_calculate_bucket_curvatures()
        self.step6_calculate_bucket_capitals()
        self.step7_calculate_psi_function()
        self.step8_calculate_interbucket_correlations()
        self.step9_calculate_final_curvature_charges()
        
        print("\n" + "="*70)
        print("CALCULATION COMPLETED SUCCESSFULLY")
        print("="*70)
        
        return self.results
    
    def export_to_excel(self, output_file: str = 'CSR_Curvature_Results.xlsx'):
        """Step 12: Export all results to Excel with comprehensive details"""
        print(f"\nStep 12: Exporting results to {output_file}...")
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            
            # Sheet 1: Final Summary
            final_summary = []
            final_summary.append(['CSR Curvature Capital Charge Results', '', ''])
            final_summary.append(['', '', ''])
            final_summary.append(['Correlation Scenario', 'Capital Charge', 'Notes'])
            
            for scenario in ['Low', 'Medium', 'High']:
                final_charge = self.results['final_calculations'][scenario]['final_curvature_charge']
                final_summary.append([scenario, final_charge, f'{scenario} correlation scenario'])
            
            df_summary = pd.DataFrame(final_summary, columns=['Metric', 'Value', 'Notes'])
            df_summary.to_excel(writer, sheet_name='Final_Summary', index=False)
            
            # Sheet 2: Bucket Delta Sensitivities
            bucket_delta_data = []
            for bucket, sensitivity in self.processed_data['bucket_delta_sensitivities'].items():
                bucket_delta_data.append([bucket, sensitivity])
            
            df_bucket_delta = pd.DataFrame(bucket_delta_data, columns=['Bucket', 'Total_Delta_Sensitivity'])
            df_bucket_delta.to_excel(writer, sheet_name='Bucket_Delta_Sensitivities', index=False)
            
            # Sheet 3: Position-Level Curvature Data
            self.processed_data['curvature_positions'].to_excel(writer, sheet_name='Position_Curvature_Data', index=False)
            
            # Sheet 4: Bucket Curvature Values
            bucket_curv_data = []
            for bucket, values in self.processed_data['bucket_curvature_values'].items():
                bucket_curv_data.append([
                    bucket, values['V_base'], values['V_RW_up'], values['V_RW_down']
                ])
            
            df_bucket_curv = pd.DataFrame(bucket_curv_data, 
                columns=['Bucket', 'V_base', 'V_RW_up', 'V_RW_down'])
            df_bucket_curv.to_excel(writer, sheet_name='Bucket_Curvature_Values', index=False)
            
            # Sheet 5: Bucket Curvature Calculations
            bucket_calc_data = []
            for bucket, curvatures in self.processed_data['bucket_curvatures'].items():
                bucket_calc_data.append([
                    bucket, curvatures['Curvature_Up'], curvatures['Curvature_Down'],
                    self.processed_data['bucket_capitals'][bucket]
                ])
            
            df_bucket_calc = pd.DataFrame(bucket_calc_data, 
                columns=['Bucket', 'Curvature_Up', 'Curvature_Down', 'Bucket_Capital_K_b'])
            df_bucket_calc.to_excel(writer, sheet_name='Bucket_Curvature_Calculations', index=False)
            
            # Sheet 6: Psi Function Values
            psi_data = []
            for (bucket_b, bucket_c), psi_value in self.processed_data['psi_values'].items():
                psi_data.append([bucket_b, bucket_c, psi_value])
            
            df_psi = pd.DataFrame(psi_data, columns=['Bucket_B', 'Bucket_C', 'Psi_Value'])
            df_psi.to_excel(writer, sheet_name='Psi_Function_Values', index=False)
            
            # Sheet 7: Inter-Bucket Correlation Details
            if hasattr(self.processed_data, 'correlation_lookup_details'):
                self.processed_data['correlation_lookup_details'].to_excel(
                    writer, sheet_name='InterBucket_Corr_Details', index=False)
            
            # Sheet 8: Final Calculation Trail
            final_calc_data = []
            for scenario in ['Low', 'Medium', 'High']:
                result = self.results['final_calculations'][scenario]
                
                final_calc_data.append([scenario, 'Sum of K_b²', result['sum_kb_squared'], ''])
                final_calc_data.append([scenario, 'Cross Bucket Sum', result['cross_bucket_sum'], 'Σ Σ γ_bc ψ × K_b × K_c'])
                final_calc_data.append([scenario, 'Total Sum', result['total_sum'], 'Sum K_b² + Cross Bucket'])
                final_calc_data.append([scenario, 'Final Curvature Charge', result['final_curvature_charge'], '√max(0, Total Sum)'])
                final_calc_data.append(['', '', '', ''])  # Separator
            
            df_final_calc = pd.DataFrame(final_calc_data, 
                columns=['Scenario', 'Component', 'Value', 'Formula'])
            df_final_calc.to_excel(writer, sheet_name='Final_Calculation_Trail', index=False)
        
        print(f"✓ Results exported successfully to {output_file}")
        print("✓ Comprehensive calculation details included in multiple sheets")
        
        # Print summary
        print("\n" + "="*60)
        print("FINAL RESULTS SUMMARY")
        print("="*60)
        for scenario in ['Low', 'Medium', 'High']:
            final_charge = self.results['final_calculations'][scenario]['final_curvature_charge']
            print(f"{scenario} Correlation CSR Curvature Capital Charge: {final_charge:.6f}")
        print("="*60)


def main():
    """Main function to run the CSR Curvature calculation"""
    
    # Initialize calculator with Excel files
    delta_file = r"bond_sensitivities_detailed_for CSR Delta.xlsx"
    curvature_file = r"CSR_Non_Sec_curvature_parallel_shifts.xlsx"
    
    try:
        calculator = CSRCurvatureCalculator(delta_file, curvature_file)
        
        # Perform all calculations
        results = calculator.calculate_all_steps()
        
        # Export results to Excel
        calculator.export_to_excel('CSR_Curvature_Capital_Results.xlsx')
        
        return calculator
        
    except FileNotFoundError as e:
        print(f"Error: Could not find Excel file - {e}")
        print("Please ensure both files exist and update the file paths.")
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
        print("Check 'CSR_Curvature_Capital_Results.xlsx' for detailed results.")
        
        # Print validation summary
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        # Show bucket composition
        if hasattr(calculator.processed_data, 'bucket_capitals'):
            for bucket, capital in calculator.processed_data['bucket_capitals'].items():
                print(f"Bucket {bucket}: K_b = {capital:.6f}")
        
        print("="*60)
