# -*- coding: utf-8 -*-
"""
Created on Sun Sep  7 13:27:40 2025

@author: amits
"""

# -*- coding: utf-8 -*-
"""
FRTB CSR (Credit Spread Risk) Non-Securitization Delta Capital Charge Calculator

This calculator implements the complete FRTB CSR Delta capital charge calculation
following regulatory guidelines with Low/Medium/High correlation scenarios.

Created for CSR Delta capital charge calculation
"""

import pandas as pd
import numpy as np
import itertools
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class CSRDeltaCalculator:
    """
    CSR (Credit Spread Risk) Non-Securitization Delta Capital Charge Calculator
    
    This class implements the complete calculation process for CSR Delta capital charges
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
        
        # Constants for CSR calculations
        self.CSR_TENORS = ['0.5Y', '1.0Y', '3.0Y', '5.0Y', '10.0Y']
        self.GIRR_TO_CSR_MAPPING = {
            '0.2Y': {'0.5Y': 1.0},  # Complete allocation to 0.5Y
            '0.5Y': {'0.5Y': 1.0},  # Direct mapping
            '1.0Y': {'1.0Y': 1.0},  # Direct mapping
            '2.0Y': {'1.0Y': 0.5, '3.0Y': 0.5},  # 50% each to 1.0Y and 3.0Y
            '3.0Y': {'3.0Y': 1.0},  # Direct mapping
            '5.0Y': {'5.0Y': 1.0},  # Direct mapping
            '10.0Y': {'10.0Y': 1.0}  # Direct mapping
        }
        
        # Correlation constants
        self.RHO_NAME_SAME = 1.0
        self.RHO_NAME_DIFFERENT = 0.35  # 35% as per instructions
        self.RHO_TENOR_SAME = 1.0
        self.RHO_TENOR_DIFFERENT = 0.65  # 65% as per instructions
        self.RHO_BASIS = 1.0  # Always 1 as per instructions
        
        # Correlation scenario multipliers
        self.HIGH_CORR_MULTIPLIER = 1.25
        self.LOW_CORR_FACTOR_75 = 0.75
        
        # Load all required data from Excel
        self._load_data()
        
    def _load_data(self):
        """Load all required data from Excel sheets"""
        print("Loading data from Excel file...")
        
        try:
            # Load bond sensitivities data
            self.raw_data['sensitivities'] = pd.read_excel(
                self.excel_file, sheet_name='Bond_Sensitivities'
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
        """
        Parse bucket notation like "1/9" to return list of buckets [1, 9]
        """
        if '/' in str(bucket_str):
            return [int(x.strip()) for x in str(bucket_str).split('/')]
        else:
            return [int(bucket_str)]
    
    def _find_interbucket_correlation(self, bucket_b, bucket_c, corr_matrix):
        """
        Find correlation between two buckets in the correlation matrix
        Handles "/" notation properly
        """
        # Get all possible bucket identifiers from matrix
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
    
    def step1_collect_sensitivities(self):
        """
        Step 1: Collect sensitivities from Bond_Sensitivities sheet
        Expected columns: Security, Type, Sector, CSR Bucket, CSR Risk weight, Tenor, Sensitivity
        """
        print("\nStep 1: Collecting bond sensitivities...")
        
        sens_df = self.raw_data['sensitivities'].copy()
        
        # Clean column names
        sens_df.columns = [col.strip() for col in sens_df.columns]
        
        # Debug: Print column names and sample data
        print("Available columns:", sens_df.columns.tolist())
        print("Sample CSR Bucket values:", sens_df['CSR Bucket'].unique())
        
        # Store processed sensitivities
        self.processed_data['sensitivities'] = sens_df
        
        # Get unique buckets and securities
        buckets = sens_df['CSR Bucket'].unique()
        securities = sens_df['Security'].unique()
        
        self.processed_data['buckets'] = buckets
        self.processed_data['securities'] = securities
        
        print(f"✓ Found {len(sens_df)} sensitivity records")
        print(f"✓ Found buckets: {sorted(buckets)}")
        print(f"✓ Found {len(securities)} unique securities")
        
        return sens_df
    
    def step2_create_csr_buckets(self):
        """
        Step 2: Create CSR buckets grouping securities by bucket number
        """
        print("\nStep 2: Creating CSR buckets...")
        
        sens_df = self.processed_data['sensitivities']
        
        # Group by CSR bucket
        csr_buckets = {}
        for bucket in self.processed_data['buckets']:
            bucket_data = sens_df[sens_df['CSR Bucket'] == bucket].copy()
            csr_buckets[bucket] = bucket_data
            
        self.processed_data['csr_buckets'] = csr_buckets
        
        for bucket, data in csr_buckets.items():
            print(f"✓ Bucket {bucket}: {len(data)} sensitivity records")
            
        return csr_buckets
    
    def step3_interpolate_girr_to_csr_tenors(self):
        """
        Step 3: Interpolate GIRR tenor sensitivity to CSR tenor sensitivity
        Using the predefined mapping rules
        """
        print("\nStep 3: Interpolating GIRR tenors to CSR tenors...")
        
        csr_sensitivities = {}
        
        for bucket, bucket_data in self.processed_data['csr_buckets'].items():
            # Create new dataframe for CSR sensitivities
            csr_bucket_data = []
            
            # Group by security to handle interpolation
            for security in bucket_data['Security'].unique():
                security_data = bucket_data[bucket_data['Security'] == security]
                
                # Get security metadata
                security_info = security_data.iloc[0]
                sector = security_info['Sector']
                csr_bucket = security_info['CSR Bucket']  # Fixed: Use correct case
                csr_risk_weight = security_info['CSR Risk weight']
                
                # Create CSR tenor sensitivities for this security
                csr_tenor_sensitivities = {tenor: 0.0 for tenor in self.CSR_TENORS}
                
                # Apply interpolation mapping
                for _, row in security_data.iterrows():
                    girr_tenor = row['Tenor']
                    girr_sensitivity = row['Sensitivity']
                    
                    if girr_tenor in self.GIRR_TO_CSR_MAPPING:
                        allocation_map = self.GIRR_TO_CSR_MAPPING[girr_tenor]
                        for csr_tenor, allocation_ratio in allocation_map.items():
                            csr_tenor_sensitivities[csr_tenor] += girr_sensitivity * allocation_ratio
                
                # Create records for each CSR tenor
                for csr_tenor, csr_sensitivity in csr_tenor_sensitivities.items():
                    if csr_sensitivity != 0:  # Only include non-zero sensitivities
                        csr_bucket_data.append({
                            'Security': security,
                            'Sector': sector,
                            'CSR Bucket': csr_bucket,  # Fixed: Use correct case
                            'CSR Risk weight': csr_risk_weight,
                            'Tenor': csr_tenor,
                            'Sensitivity': csr_sensitivity
                        })
            
            csr_sensitivities[bucket] = pd.DataFrame(csr_bucket_data)
        
        self.processed_data['csr_sensitivities'] = csr_sensitivities
        
        for bucket, data in csr_sensitivities.items():
            print(f"✓ Bucket {bucket}: {len(data)} CSR sensitivity records after interpolation")
        
        return csr_sensitivities
    
    def step4_calculate_weighted_sensitivities(self):
        """
        Step 4: Calculate weighted sensitivities
        Formula: WS_k = RW_k * S_k
        """
        print("\nStep 4: Calculating weighted sensitivities...")
        
        weighted_sensitivities = {}
        
        for bucket, bucket_data in self.processed_data['csr_sensitivities'].items():
            bucket_ws = bucket_data.copy()
            
            # Calculate weighted sensitivity
            bucket_ws['Weighted_Sensitivity'] = (
                bucket_ws['Sensitivity'] * bucket_ws['CSR Risk weight']
            )
            
            weighted_sensitivities[bucket] = bucket_ws
        
        self.processed_data['weighted_sensitivities'] = weighted_sensitivities
        
        for bucket, data in weighted_sensitivities.items():
            total_ws = data['Weighted_Sensitivity'].sum()
            print(f"✓ Bucket {bucket} total weighted sensitivity: {total_ws:.6f}")
        
        return weighted_sensitivities
    
    def step5_calculate_medium_correlations(self):
        """
        Step 5: Calculate medium level correlations between weighted sensitivities within buckets
        """
        print("\nStep 5: Calculating medium correlations within buckets...")
        
        medium_correlations = {}
        
        for bucket, bucket_data in self.processed_data['weighted_sensitivities'].items():
            # Create correlation pairs with details in single structure
            correlation_details = []
            
            for i, row_i in bucket_data.iterrows():
                for j, row_j in bucket_data.iterrows():
                    if i < j:  # Avoid duplicate pairs and self-correlation
                        # Calculate correlation components
                        # ρ_name: 1 if same security, 0.35 if different
                        rho_name = (self.RHO_NAME_SAME if row_i['Security'] == row_j['Security'] 
                                   else self.RHO_NAME_DIFFERENT)
                        
                        # ρ_tenor: 1 if same tenor, 0.65 if different
                        rho_tenor = (self.RHO_TENOR_SAME if row_i['Tenor'] == row_j['Tenor'] 
                                    else self.RHO_TENOR_DIFFERENT)
                        
                        # ρ_basis: always 1
                        rho_basis = self.RHO_BASIS
                        
                        # Final correlation
                        rho_kl = rho_name * rho_tenor * rho_basis
                        
                        # Store all details in single structure
                        correlation_details.append({
                            'pair': (i, j),
                            'security_i': row_i['Security'],
                            'security_j': row_j['Security'],
                            'tenor_i': row_i['Tenor'],
                            'tenor_j': row_j['Tenor'],
                            'ws_i': row_i['Weighted_Sensitivity'],
                            'ws_j': row_j['Weighted_Sensitivity'],
                            'rho_name': rho_name,
                            'rho_tenor': rho_tenor,
                            'rho_basis': rho_basis,
                            'correlation': rho_kl,
                            'cross_product': rho_kl * row_i['Weighted_Sensitivity'] * row_j['Weighted_Sensitivity']
                        })
            
            medium_correlations[bucket] = {
                'correlation_details': correlation_details,
                'data': bucket_data
            }
            
            print(f"✓ Bucket {bucket}: {len(correlation_details)} correlation pairs calculated")
        
        self.processed_data['medium_correlations'] = medium_correlations
        
        return medium_correlations
    
    def step6_calculate_high_correlations(self):
        """
        Step 6: Calculate high correlations
        Formula: ρ_high = min(ρ_medium * 1.25, 1.0)
        """
        print("\nStep 6: Calculating high correlations...")
        
        high_correlations = {}
        
        for bucket, bucket_corr_data in self.processed_data['medium_correlations'].items():
            correlation_details = []
            
            for detail in bucket_corr_data['correlation_details']:
                high_corr = min(detail['correlation'] * self.HIGH_CORR_MULTIPLIER, 1.0)
                high_detail = detail.copy()
                high_detail['correlation'] = high_corr
                high_detail['cross_product'] = high_corr * detail['ws_i'] * detail['ws_j']
                correlation_details.append(high_detail)
            
            high_correlations[bucket] = {
                'correlation_details': correlation_details,
                'data': bucket_corr_data['data']
            }
        
        self.processed_data['high_correlations'] = high_correlations
        
        print("✓ High correlations calculated (medium × 1.25, capped at 100%)")
        
        return high_correlations
    
    def step7_calculate_low_correlations(self):
        """
        Step 7: Calculate low correlations
        Formula: ρ_low = max(2 × ρ_medium - 100%, 75% × ρ_medium)
        """
        print("\nStep 7: Calculating low correlations...")
        
        low_correlations = {}
        
        for bucket, bucket_corr_data in self.processed_data['medium_correlations'].items():
            correlation_details = []
            
            for detail in bucket_corr_data['correlation_details']:
                medium_corr = detail['correlation']
                option1 = 2 * medium_corr - 1.0  # 2 × ρ - 100%
                option2 = self.LOW_CORR_FACTOR_75 * medium_corr  # 75% × ρ
                low_corr = max(option1, option2)
                
                low_detail = detail.copy()
                low_detail['correlation'] = low_corr
                low_detail['cross_product'] = low_corr * detail['ws_i'] * detail['ws_j']
                correlation_details.append(low_detail)
            
            low_correlations[bucket] = {
                'correlation_details': correlation_details,
                'data': bucket_corr_data['data']
            }
        
        self.processed_data['low_correlations'] = low_correlations
        
        print("✓ Low correlations calculated using max(2×ρ-100%, 75%×ρ)")
        
        return low_correlations
    
    def step8_calculate_bucket_capitals(self):
        """
        Step 8: Calculate bucket capital charges for all three correlation scenarios
        Formula: K_b = √max(0, Σ(WS_k²) + ΣΣ(ρ_kl × WS_k × WS_l))
        """
        print("\nStep 8: Calculating bucket capital charges...")
        
        correlation_scenarios = {
            'Low': self.processed_data['low_correlations'],
            'Medium': self.processed_data['medium_correlations'],
            'High': self.processed_data['high_correlations']
        }
        
        bucket_capitals = {}
        
        for scenario_name, scenario_data in correlation_scenarios.items():
            scenario_capitals = {}
            
            for bucket, bucket_corr_data in scenario_data.items():
                bucket_data = bucket_corr_data['data']
                correlation_details = bucket_corr_data['correlation_details']
                
                # Get weighted sensitivities
                ws_values = bucket_data['Weighted_Sensitivity'].values
                
                # Calculate sum of squares
                sum_squares = np.sum(ws_values**2)
                
                # Calculate cross products with correlations
                cross_product_sum = sum(detail['cross_product'] for detail in correlation_details)
                
                # Apply bucket capital formula
                bucket_capital = np.sqrt(max(0, sum_squares + cross_product_sum))
                scenario_capitals[bucket] = bucket_capital
            
            bucket_capitals[scenario_name] = scenario_capitals
        
        self.processed_data['bucket_capitals'] = bucket_capitals
        
        # Print results
        for scenario in ['Low', 'Medium', 'High']:
            print(f"✓ {scenario} correlation bucket capitals:")
            for bucket, capital in bucket_capitals[scenario].items():
                print(f"  Bucket {bucket}: {capital:.6f}")
        
        return bucket_capitals
    
    def step9_calculate_interbucket_correlations(self):
        """
        Step 9: Calculate inter-bucket correlations for all scenarios
        """
        print("\nStep 9: Calculating inter-bucket correlations...")
        
        # Get unique buckets
        buckets = list(self.processed_data['buckets'])
        
        # Get sector for each bucket
        bucket_sectors = {}
        for bucket in buckets:
            bucket_data = self.processed_data['weighted_sensitivities'][bucket]
            bucket_sectors[bucket] = bucket_data['Sector'].iloc[0]
        
        # Calculate inter-bucket correlations
        interbucket_medium_correlations = {}
        interbucket_corr_matrix = self.raw_data['interbucket_correlations']
        
        # Store correlation lookup details for audit
        correlation_lookup_details = []
        
        for bucket_b in buckets:
            for bucket_c in buckets:
                if bucket_b != bucket_c:
                    # γ_rating = 1 (all are IG as per instructions)
                    gamma_rating = 1.0
                    
                    # γ_sector: check if same sector or get from correlation matrix
                    sector_b = bucket_sectors[bucket_b]
                    sector_c = bucket_sectors[bucket_c]
                    
                    if sector_b == sector_c:
                        gamma_sector = 1.0
                        lookup_source = "Same Sector"
                    else:
                        # Get from correlation matrix using proper parsing
                        gamma_sector = self._find_interbucket_correlation(bucket_b, bucket_c, interbucket_corr_matrix)
                        lookup_source = "Correlation Matrix"
                    
                    # Calculate medium inter-bucket correlation
                    gamma_bc = gamma_rating * gamma_sector
                    interbucket_medium_correlations[(bucket_b, bucket_c)] = gamma_bc
                    
                    # Store details for audit
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
        
        # Store correlation lookup details for audit
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
    
    def step10_across_bucket_aggregations(self):
        """
        Step 10: Perform across bucket aggregations for all three scenarios
        """
        print("\nStep 10: Performing across bucket aggregations...")
        
        # Calculate bucket aggregate sensitivities (same for all scenarios)
        bucket_aggregate_sensitivities = {}
        
        for bucket, bucket_data in self.processed_data['weighted_sensitivities'].items():
            s_bucket = bucket_data['Weighted_Sensitivity'].sum()
            bucket_aggregate_sensitivities[bucket] = s_bucket
        
        self.processed_data['bucket_aggregate_sensitivities'] = bucket_aggregate_sensitivities
        
        print("✓ Bucket aggregate sensitivities:")
        for bucket, s_value in bucket_aggregate_sensitivities.items():
            print(f"  S_bucket_{bucket}: {s_value:.6f}")
        
        # Calculate CSR Delta capital charge for each scenario
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
                    if bucket_b != bucket_c:  # b ≠ c condition
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
                final_capital_charge = np.sqrt(max(0, final_sum))
                
                print(f"Alternative S values: {alternative_s}")
                print(f"Final sum (alternative): {final_sum:.6f}")
                
            else:
                final_capital_charge = np.sqrt(initial_sum)
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
        print("CSR NON-SECURITIZATION DELTA CAPITAL CHARGE CALCULATION")
        print("="*70)
        
        # Execute all steps
        self.step1_collect_sensitivities()
        self.step2_create_csr_buckets()
        self.step3_interpolate_girr_to_csr_tenors()
        self.step4_calculate_weighted_sensitivities()
        self.step5_calculate_medium_correlations()
        self.step6_calculate_high_correlations()
        self.step7_calculate_low_correlations()
        self.step8_calculate_bucket_capitals()
        self.step9_calculate_interbucket_correlations()
        self.step10_across_bucket_aggregations()
        
        print("\n" + "="*70)
        print("CALCULATION COMPLETED SUCCESSFULLY")
        print("="*70)
        
        return self.results
    
    def export_to_excel(self, output_file: str = 'CSR_Delta_Results.xlsx'):
        """
        Step 11: Export all results to Excel with comprehensive audit details
        """
        print(f"\nStep 11: Exporting results to {output_file}...")
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            
            # Sheet 1: Final Summary Results
            final_summary = []
            final_summary.append(['CSR Non-Securitization Delta Capital Charge Results', '', ''])
            final_summary.append(['', '', ''])
            final_summary.append(['Correlation Scenario', 'Capital Charge', 'Notes'])
            
            for scenario in ['Low', 'Medium', 'High']:
                final_charge = self.results['final_calculations'][scenario]['final_capital_charge']
                final_summary.append([scenario, final_charge, f'{scenario} correlation scenario'])
            
            df_summary = pd.DataFrame(final_summary, columns=['Metric', 'Value', 'Notes'])
            df_summary.to_excel(writer, sheet_name='Final_Summary', index=False)
            
            # Sheet 2: Pre-Interpolation Sensitivity Details
            pre_interp_data = []
            for bucket, bucket_data in self.processed_data['csr_buckets'].items():
                for _, row in bucket_data.iterrows():
                    pre_interp_data.append([
                        row['Security'], 
                        bucket, 
                        row['Sector'],
                        row['Tenor'], 
                        row['Sensitivity'],
                        row['CSR Risk weight'],
                        'GIRR Tenor'
                    ])
            
            df_pre_interp = pd.DataFrame(pre_interp_data, 
                columns=['Security', 'Bucket', 'Sector', 'Tenor', 'Sensitivity', 'Risk_Weight', 'Source'])
            df_pre_interp.to_excel(writer, sheet_name='Pre_Interpolation_Detail', index=False)
            
            # Sheet 3: Post-Interpolation CSR Sensitivity Details
            post_interp_data = []
            for bucket, bucket_data in self.processed_data['csr_sensitivities'].items():
                for _, row in bucket_data.iterrows():
                    post_interp_data.append([
                        row['Security'], 
                        bucket, 
                        row['Sector'],
                        row['Tenor'], 
                        row['Sensitivity'],
                        row['CSR Risk weight'],
                        'CSR Tenor (Post-Interpolation)'
                    ])
            
            df_post_interp = pd.DataFrame(post_interp_data, 
                columns=['Security', 'Bucket', 'Sector', 'Tenor', 'Sensitivity', 'Risk_Weight', 'Source'])
            df_post_interp.to_excel(writer, sheet_name='Post_Interpolation_Detail', index=False)
            
            # Sheet 4: Correlation Pair Details (Medium Scenario)
            correlation_pairs_data = []
            for bucket, bucket_corr_data in self.processed_data['medium_correlations'].items():
                for detail in bucket_corr_data['correlation_details']:
                    correlation_pairs_data.append([
                        bucket,
                        f"{detail['security_i']}_{detail['tenor_i']}",
                        f"{detail['security_j']}_{detail['tenor_j']}",
                        detail['security_i'] == detail['security_j'],
                        detail['tenor_i'] == detail['tenor_j'],
                        detail['rho_name'],
                        detail['rho_tenor'],
                        detail['rho_basis'],
                        detail['correlation'],
                        detail['ws_i'],
                        detail['ws_j'],
                        detail['cross_product']
                    ])
            
            df_corr_pairs = pd.DataFrame(correlation_pairs_data, 
                columns=['Bucket', 'Security_Tenor_1', 'Security_Tenor_2', 'Same_Security', 'Same_Tenor', 
                        'Rho_Name', 'Rho_Tenor', 'Rho_Basis', 'Final_Correlation', 'WS_1', 'WS_2', 'Cross_Product'])
            df_corr_pairs.to_excel(writer, sheet_name='Correlation_Pair_Details', index=False)
            
            # Sheet 5: Bucket Capital Calculations
            bucket_calc_data = []
            for scenario in ['Low', 'Medium', 'High']:
                for bucket, capital in self.processed_data['bucket_capitals'][scenario].items():
                    # Get sum of squares for this bucket
                    bucket_data = self.processed_data['weighted_sensitivities'][bucket]
                    ws_values = bucket_data['Weighted_Sensitivity'].values
                    sum_squares = np.sum(ws_values**2)
                    
                    # Get cross products sum
                    bucket_corr_data = self.processed_data[f'{scenario.lower()}_correlations'][bucket]
                    cross_product_sum = sum(detail['cross_product'] for detail in bucket_corr_data['correlation_details'])
                    
                    bucket_calc_data.append([
                        scenario, bucket, sum_squares, cross_product_sum, 
                        sum_squares + cross_product_sum, capital
                    ])
            
            df_bucket_calc = pd.DataFrame(bucket_calc_data, 
                columns=['Scenario', 'Bucket', 'Sum_Squares', 'Cross_Product_Sum', 
                        'Total_Sum', 'Bucket_Capital'])
            df_bucket_calc.to_excel(writer, sheet_name='Bucket_Capital_Calculations', index=False)
            
            # Sheet 6: Inter-Bucket Correlation Lookup Details
            if hasattr(self.processed_data, 'correlation_lookup_details'):
                self.processed_data['correlation_lookup_details'].to_excel(
                    writer, sheet_name='InterBucket_Corr_Lookup', index=False)
            
            # Sheet 7: Final Capital Calculation Trail
            final_calc_data = []
            for scenario in ['Low', 'Medium', 'High']:
                result = self.results['final_calculations'][scenario]
                
                # Add bucket capitals sum of squares
                sum_kb_squared = sum(k**2 for k in result['bucket_capitals'].values())
                final_calc_data.append([scenario, 'Sum of K_b²', sum_kb_squared, ''])
                
                # Add cross bucket terms
                buckets = list(self.processed_data['buckets'])
                cross_bucket_sum = 0.0
                for bucket_b in buckets:
                    for bucket_c in buckets:
                        if bucket_b != bucket_c:
                            s_b = result['bucket_aggregate_sensitivities'][bucket_b]
                            s_c = result['bucket_aggregate_sensitivities'][bucket_c]
                            gamma_bc = result['interbucket_correlations'].get((bucket_b, bucket_c), 0.0)
                            cross_term = gamma_bc * s_b * s_c
                            cross_bucket_sum += cross_term
                            
                            final_calc_data.append([
                                scenario, f'γ({bucket_b},{bucket_c}) × S_{bucket_b} × S_{bucket_c}', 
                                cross_term, f'{gamma_bc:.6f} × {s_b:.6f} × {s_c:.6f}'
                            ])
                
                final_calc_data.append([scenario, 'Cross Bucket Sum', cross_bucket_sum, ''])
                final_calc_data.append([scenario, 'Total Sum', result['initial_sum'], ''])
                final_calc_data.append([scenario, 'Final Capital Charge', result['final_capital_charge'], '√(Total Sum)'])
                final_calc_data.append(['', '', '', ''])  # Separator
            
            df_final_calc = pd.DataFrame(final_calc_data, 
                columns=['Scenario', 'Component', 'Value', 'Formula'])
            df_final_calc.to_excel(writer, sheet_name='Final_Calculation_Trail', index=False)
        
        print(f"✓ Results exported successfully to {output_file}")
        print("✓ Comprehensive audit details included in multiple sheets")
        
        # Print summary
        print("\n" + "="*60)
        print("FINAL RESULTS SUMMARY")
        print("="*60)
        for scenario in ['Low', 'Medium', 'High']:
            final_charge = self.results['final_calculations'][scenario]['final_capital_charge']
            print(f"{scenario} Correlation CSR Delta Capital Charge: {final_charge:.6f}")
        print("="*60)


def main():
    """
    Main function to run the CSR Delta calculation
    """
    
    # Initialize calculator with Excel file
    excel_file = r"bond_sensitivities_detailed_for CSR Delta.xlsx"
    
    try:
        calculator = CSRDeltaCalculator(excel_file)
        
        # Perform all calculations
        results = calculator.calculate_all_steps()
        
        # Export results to Excel
        calculator.export_to_excel('CSR_Delta_Capital_Results.xlsx')
        
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
        print("Check 'CSR_Delta_Capital_Results.xlsx' for detailed results.")
        
        # Print validation summary
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        # Show bucket composition
        for bucket in calculator.processed_data['buckets']:
            bucket_data = calculator.processed_data['weighted_sensitivities'][bucket]
            securities = bucket_data['Security'].unique()
            print(f"Bucket {bucket}: {len(securities)} securities")
            
        # Show tenor distribution after interpolation
        print("\nTenor distribution after GIRR->CSR interpolation:")
        for bucket in calculator.processed_data['buckets']:
            bucket_data = calculator.processed_data['csr_sensitivities'][bucket]
            tenor_counts = bucket_data['Tenor'].value_counts()
            print(f"Bucket {bucket}: {dict(tenor_counts)}")
        
        print("="*60)
