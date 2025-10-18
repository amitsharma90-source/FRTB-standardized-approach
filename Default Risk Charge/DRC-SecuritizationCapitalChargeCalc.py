# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 20:07:27 2025

@author: amits
"""

# -*- coding: utf-8 -*-
"""
DRC Securitization Calculator
Based on Basel Framework MAR22/MAR23 - SEC-ERBA
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple

class DRCSecuritizationCalculator:
    """
    Default Risk Charge (DRC) Calculator for Securitizations
    Based on Basel Framework MAR22/MAR23
    """
    
    def __init__(self, holdings_file: str):
        """
        Initialize calculator with holdings file
        
        Args:
            holdings_file: Path to Excel file with holdings data
        """
        # Load securitization holdings
        self.df = pd.read_excel(holdings_file, sheet_name='Securitization Holdings')
        
        # Load risk weight tables
        self.rw_lt = pd.read_excel(holdings_file, sheet_name='SecuritizaRW-LT')
        self.rw_st = pd.read_excel(holdings_file, sheet_name='SecuritizaRW-ST')
        
        self.today = datetime.now()
        
        print(f"Loaded {len(self.df)} securitization positions")
        print(f"Loaded {len(self.rw_lt)} long-term risk weight mappings")
        print(f"Loaded {len(self.rw_st)} short-term risk weight mappings")
        
    def calculate_gross_jtd(self) -> pd.DataFrame:
        """
        Step 2: Calculate Gross JTD for each security
        
        For securitizations: JTD = Market Value (no LGD multiplier per MAR22.27)
        Sign is already in Market Value (positive for long, negative for short)
        
        Returns:
            DataFrame with Gross_JTD column
        """
        # Gross JTD is simply the market value (sign already applied)
        self.df['Gross_JTD_unscaled'] = self.df['Market Value']
        
        return self.df
    
    def calculate_year_fraction(self) -> pd.DataFrame:
        """
        Step 2.iii: Calculate year fraction of expected maturity
        
        If tenure >= 1 year: year_fraction = 1.0
        If tenure < 1 year: year_fraction = tenure in years
        
        Returns:
            DataFrame with Year_Fraction column
        """
        # Calculate tenure (Expected Maturity - Today)
        self.df['Expected Maturity'] = pd.to_datetime(self.df['Expected Maturity'])
        self.df['Tenure_Days'] = (self.df['Expected Maturity'] - self.today).dt.days
        self.df['Tenure_Years'] = self.df['Tenure_Days'] / 365.25
        
        # Year fraction capped at 1.0
        self.df['Year_Fraction'] = np.minimum(self.df['Tenure_Years'], 1.0)
        
        # Handle negative tenures (already matured) as 0
        self.df['Year_Fraction'] = np.maximum(self.df['Year_Fraction'], 0)
        
        return self.df
    
    def scale_gross_jtd(self) -> pd.DataFrame:
        """
        Step 2.iv: Scale each Gross JTD by year fraction
        This is the final Gross JTD
        
        Returns:
            DataFrame with scaled Gross_JTD
        """
        self.df['Gross_JTD'] = self.df['Gross_JTD_unscaled'] * self.df['Year_Fraction']
        return self.df
    
    def get_netting_key(self, row) -> tuple:
        """
        Create unique key for netting positions
        
        Per MAR22.29 and your specification:
        - Same Pool ID
        - Same Attachment %
        - Same Detachment %
        - Same Bucket
        - Same Tranche Type
        (Rating can differ - removed from netting criteria)
        
        Args:
            row: DataFrame row
            
        Returns:
            Tuple representing netting key
        """
        return (
            row['Pool ID'],
            row['Attachment Point (%)'],
            row['Detachment Point (%)'],
            row['Bucket'],
            row['Tranche Type']
        )
    
    def calculate_all_net_jtds(self) -> pd.DataFrame:
        """
        Step 3: Calculate Net JTDs from Gross JTDs
        
        Group positions by netting key and aggregate:
        - Same sign (all long or all short) → add them
        - Mixed signs (long + short) → net them
        
        Returns:
            DataFrame of Net JTDs
        """
        all_net_jtds = []
        
        # Add netting key to dataframe
        self.df['Netting_Key'] = self.df.apply(self.get_netting_key, axis=1)
        
        # Group by netting key
        for netting_key, group in self.df.groupby('Netting_Key'):
            # Get common attributes for this group
            pool_id = group['Pool ID'].iloc[0]
            bucket = group['Bucket'].iloc[0]
            tranche_type = group['Tranche Type'].iloc[0]
            attachment = group['Attachment Point (%)'].iloc[0]
            detachment = group['Detachment Point (%)'].iloc[0]
            
            # For maturity: use largest (latest) maturity
            expected_maturity = group['Expected Maturity'].max()
            
            # For rating: use most common rating (or first if tie)
            # Note: Ratings can differ in netted positions, we take the most frequent one
            rating = group['Rating'].mode().iloc[0] if len(group['Rating'].mode()) > 0 else group['Rating'].iloc[0]
            
            # Sum all Gross JTDs (positive for long, negative for short)
            net_jtd_value = group['Gross_JTD'].sum()
            
            # Determine direction based on sign
            if abs(net_jtd_value) > 1e-10:  # Not essentially zero
                direction = 'Long' if net_jtd_value > 0 else 'Short'
                
                all_net_jtds.append({
                    'Pool_ID': pool_id,
                    'Bucket': bucket,
                    'Tranche_Type': tranche_type,
                    'Attachment_%': attachment,
                    'Detachment_%': detachment,
                    'Rating': rating,
                    'Expected_Maturity': expected_maturity,
                    'Net_JTD': net_jtd_value,
                    'Direction': direction
                })
        
        self.net_jtd_df = pd.DataFrame(all_net_jtds)
        return self.net_jtd_df
    
    def calculate_maturity_weights(self, net_maturity_years: float) -> Tuple[float, float]:
        """
        Step 5b: Calculate maturity weights for linear interpolation
        
        Linear interpolation between 1-year and 5-year risk weights
        
        Args:
            net_maturity_years: Maturity in years
            
        Returns:
            Tuple of (1_year_weight, 5_year_weight) in percentages
        """
        if net_maturity_years < 1:
            # Net maturity < 1 year: 100% weight on 1-year, 0% on 5-year
            return (100.0, 0.0)
        elif net_maturity_years >= 5:
            # Net maturity >= 5 years: 0% weight on 1-year, 100% on 5-year
            return (0.0, 100.0)
        else:
            # Linear interpolation between 1 and 5 years
            # Weight decreases linearly from 100% to 0% for 1-year
            # Weight increases linearly from 0% to 100% for 5-year
            years_from_1 = net_maturity_years - 1
            total_span = 5 - 1  # 4 years
            
            five_year_weight = (years_from_1 / total_span) * 100
            one_year_weight = 100 - five_year_weight
            
            return (one_year_weight, five_year_weight)
    
    def lookup_risk_weights_lt(self, rating: str, tranche_type: str) -> Tuple[float, float]:
        """
        Step 5c.i and 5c.ii: Lookup risk weights from long-term table
        
        Args:
            rating: Credit rating
            tranche_type: 'Senior' or 'Mezzanine'/'Junior'
            
        Returns:
            Tuple of (1_year_RW, 5_year_RW) in percentages, or (None, None) if not found
        """
        # Find rating row
        rating_row = self.rw_lt[self.rw_lt['Rating'] == rating]
        
        if len(rating_row) == 0:
            return (None, None)
        
        rating_row = rating_row.iloc[0]
        
        # Determine which columns to use based on tranche type
        if tranche_type == 'Senior':
            # Senior tranche: columns B and C
            rw_1y = rating_row['Senior tranche maturity 1 year']
            rw_5y = rating_row['Senior tranche maturity 5 year']
        else:  # Mezzanine or Junior
            # Non-senior tranche: columns D and E
            rw_1y = rating_row['Non Senior tranche maturity 1 year']
            rw_5y = rating_row['Non Senior tranche maturity 5 year']
        
        # Check if values are valid (not NaN)
        if pd.isna(rw_1y) or pd.isna(rw_5y):
            return (None, None)
        
        return (float(rw_1y), float(rw_5y))
    
    def lookup_risk_weight_st(self, rating: str) -> float:
        """
        Step 5c.iii: Lookup risk weight from short-term table
        
        Args:
            rating: Short-term credit rating (e.g., A-1, P-1)
            
        Returns:
            Risk weight in percentage, or None if not found
        """
        # Search through all rows in the short-term table
        # Each row may contain multiple ratings
        for _, row in self.rw_st.iterrows():
            rating_str = str(row['Rating'])
            # Check if this row's rating string contains the target rating
            # Handle cases like "A-1/P-1" where multiple ratings map to same RW
            if rating in rating_str or rating.replace('-', '') in rating_str.replace('-', ''):
                rw = row['Risk weight']
                if not pd.isna(rw):
                    return float(rw)
        
        return None
    
    def calculate_risk_weight_for_net_jtd(self, row) -> float:
        """
        Step 5: Calculate final risk weight for a Net JTD position
        
        Process:
        a) Calculate net maturity
        b) Get maturity weights (interpolation)
        c) Lookup 1-year and 5-year risk weights
        d) Calculate tranche thickness
        e) Calculate combined risk weight
        f) Apply tranche thickness adjustment
        
        Args:
            row: Net JTD row
            
        Returns:
            Final risk weight as decimal (not percentage)
        """
        # Step 5a: Calculate net maturity
        net_maturity_days = (row['Expected_Maturity'] - self.today).days
        net_maturity_years = net_maturity_days / 365.25
        net_maturity_years = max(0, net_maturity_years)  # Handle negative
        
        # Step 5b: Get maturity weights
        weight_1y, weight_5y = self.calculate_maturity_weights(net_maturity_years)
        
        # Step 5c: Determine risk weights
        rating = row['Rating']
        tranche_type = row['Tranche_Type']
        
        # Try long-term table first
        rw_1y, rw_5y = self.lookup_risk_weights_lt(rating, tranche_type)
        
        if rw_1y is not None and rw_5y is not None:
            # Found in long-term table
            # Step 5e: Calculate combined risk weight
            combined_rw = (rw_1y * weight_1y + rw_5y * weight_5y) / 100 #weight_1y and weight_5y are in %
        else:
            # Step 5c.iii: Try short-term table
            rw_st = self.lookup_risk_weight_st(rating)
            
            if rw_st is not None:
                # Found in short-term table
                combined_rw = rw_st
            else:
                # Not found in either table - raise error or use default
                print(f"WARNING: No risk weight found for Rating={rating}, Tranche={tranche_type}")
                combined_rw = 1.0  # Default to 100% risk weight
        
        # Step 5d: Calculate tranche thickness
        T = row['Detachment_%'] - row['Attachment_%']
        
        # Step 5f: Apply tranche thickness adjustment
        # Final RW = Combined RW × (1 - min(T, 50%) / 100)
        thickness_adjustment = 1 - (min(T, 50) / 100)
        final_rw = combined_rw * thickness_adjustment
        
        return final_rw
    
    def calculate_all_risk_weights(self) -> pd.DataFrame:
        """
        Calculate risk weights for all Net JTD positions
        
        Returns:
            DataFrame with Risk_Weight column added
        """
        self.net_jtd_df['Risk_Weight'] = self.net_jtd_df.apply(
            self.calculate_risk_weight_for_net_jtd, 
            axis=1
        )
        
        return self.net_jtd_df
    
    def calculate_hbr_for_bucket(self, bucket_df: pd.DataFrame) -> float:
        """
        Step 6: Calculate Hedge Benefit Ratio (HBR) for a bucket
        
        HBR = Σ net JTD_long / (Σ net JTD_long + Σ |net JTD_short|)
        
        HBR = 0 means 100% hedging
        HBR = 1 (100%) means 0% hedging
        
        Args:
            bucket_df: DataFrame of Net JTDs for one bucket
            
        Returns:
            HBR value between 0 and 1
        """
        long_jtds = bucket_df[bucket_df['Direction'] == 'Long']['Net_JTD'].sum()
        short_jtds = bucket_df[bucket_df['Direction'] == 'Short']['Net_JTD'].abs().sum()
        
        denominator = long_jtds + short_jtds
        
        if denominator == 0:
            return 1.0  # No positions = no hedging benefit
        
        hbr = long_jtds / denominator
        return hbr
    
    def calculate_bucket_drc(self, bucket_df: pd.DataFrame, hbr: float) -> float:
        """
        Step 7: Calculate DRC for a bucket
        
        Bucket DRC = max[(Σ RW_i × net JTD_i)_long - HBR × (Σ RW_i × |net JTD_i|)_short, 0]
        
        Args:
            bucket_df: DataFrame of Net JTDs for one bucket
            hbr: Hedge Benefit Ratio for this bucket
            
        Returns:
            DRC capital charge for the bucket
        """
        # Calculate risk-weighted long exposure
        long_positions = bucket_df[bucket_df['Direction'] == 'Long'].copy()
        rw_long = (long_positions['Risk_Weight'] * long_positions['Net_JTD']).sum()
        
        # Calculate risk-weighted short exposure (absolute value)
        short_positions = bucket_df[bucket_df['Direction'] == 'Short'].copy()
        rw_short = (short_positions['Risk_Weight'] * short_positions['Net_JTD'].abs()).sum()
        
        # Apply DRC formula
        bucket_drc = max(rw_long - hbr * rw_short, 0)
        
        return bucket_drc
    
    def calculate_total_drc(self) -> Tuple[float, Dict]:
        """
        Steps 6-8: Calculate DRC for all buckets and aggregate
        
        Returns:
            Tuple of (total_drc, bucket_details)
        """
        bucket_details = {}
        total_drc = 0
        
        # Process each bucket
        for bucket_num in sorted(self.net_jtd_df['Bucket'].unique()):
            bucket_df = self.net_jtd_df[self.net_jtd_df['Bucket'] == bucket_num]
            
            # Calculate HBR for this bucket
            hbr = self.calculate_hbr_for_bucket(bucket_df)
            
            # Calculate DRC for this bucket
            bucket_drc = self.calculate_bucket_drc(bucket_df, hbr)
            
            # Store details
            bucket_details[f'Bucket_{bucket_num}'] = {
                'HBR': hbr,
                'Bucket_DRC': bucket_drc,
                'Num_Positions': len(bucket_df),
                'Total_Long': bucket_df[bucket_df['Direction'] == 'Long']['Net_JTD'].sum(),
                'Total_Short': bucket_df[bucket_df['Direction'] == 'Short']['Net_JTD'].sum()
            }
            
            # Accumulate to total DRC
            total_drc += bucket_drc
        
        return total_drc, bucket_details
    
    def run_full_calculation(self) -> Tuple[float, Dict, pd.DataFrame, pd.DataFrame]:
        """
        Execute complete DRC securitization calculation workflow
        
        Returns:
            Tuple of (total_drc, bucket_details, gross_jtd_df, net_jtd_df)
        """
        print("\n" + "="*60)
        print("DRC SECURITIZATION CALCULATION")
        print("="*60)
        
        print("\nStep 1: Loading holdings data...")
        print(f"Loaded {len(self.df)} securitization positions\n")
        
        print("Step 2: Calculating Gross JTD...")
        self.calculate_gross_jtd()
        self.calculate_year_fraction()
        self.scale_gross_jtd()
        print(f"Calculated Gross JTD for {len(self.df)} positions\n")
        
        print("Step 3: Calculating Net JTD...")
        self.calculate_all_net_jtds()
        print(f"Calculated {len(self.net_jtd_df)} Net JTD positions\n")
        
        print("Step 5: Calculating Risk Weights...")
        self.calculate_all_risk_weights()
        print("Risk weights calculated for all Net JTD positions\n")
        
        print("Steps 6-8: Calculating DRC by bucket...")
        total_drc, bucket_details = self.calculate_total_drc()
        
        print("\n" + "="*60)
        print("BUCKET-LEVEL RESULTS")
        print("="*60)
        for bucket_name, details in bucket_details.items():
            print(f"\n{bucket_name}:")
            print(f"  Number of Net JTD positions: {details['Num_Positions']}")
            print(f"  Total Long Exposure: {details['Total_Long']:,.2f}")
            print(f"  Total Short Exposure: {details['Total_Short']:,.2f}")
            print(f"  Hedge Benefit Ratio (HBR): {details['HBR']:.2%}")
            print(f"  Bucket DRC: {details['Bucket_DRC']:,.2f}")
        
        print("\n" + "="*60)
        print(f"TOTAL DRC SECURITIZATION CAPITAL CHARGE: {total_drc:,.2f}")
        print("="*60)
        
        return total_drc, bucket_details, self.df, self.net_jtd_df


def main():
    """
    Main execution function
    """
    # Initialize calculator with holdings file
    calculator = DRCSecuritizationCalculator(
        "Default Risk charge holdings.xlsx"
    )
    
    # Run full calculation
    total_drc, bucket_details, gross_jtd_df, net_jtd_df = calculator.run_full_calculation()
    
    # Optional: Save detailed results to Excel
    with pd.ExcelWriter('DRC_Securitization_Results.xlsx') as writer:
        gross_jtd_df.to_excel(writer, sheet_name='Gross_JTD', index=False)
        net_jtd_df.to_excel(writer, sheet_name='Net_JTD', index=False)
        
        # Bucket summary
        bucket_summary = pd.DataFrame(bucket_details).T
        bucket_summary.to_excel(writer, sheet_name='Bucket_Summary')
    
    print("\nDetailed results saved to 'DRC_Securitization_Results.xlsx'")


if __name__ == "__main__":
    main()
