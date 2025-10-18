# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 07:35:19 2025

@author: amits
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple

class DRCCalculator:
    """
    Default Risk Charge (DRC) Calculator for Non-Securitizations
    Based on Basel Framework MAR22
    """
    
    def __init__(self, holdings_file: str):
        """
        Initialize calculator with holdings file
        
        Args:
            holdings_file: Path to Excel file with holdings data
        """
        self.df = pd.read_excel(holdings_file)
        self.today = datetime.now()
        
        # Seniority hierarchy (higher number = more senior)
        # Note: LGD is based on Seniority
        self.seniority_rank = {
            'Covered': 3,
            'Senior': 2,
            'Non Senior': 1
        }
        
        # Risk weights based on rating (from MAR22.24 Table 2)
        # Note: Risk Weight is based on Rating. A rating is associated with an issuer.
        self.risk_weights = {
            'AAA': 0.5,
            'AA': 2.0,
            'A': 3.0,
            'BBB': 6.0,
            'BB': 15.0,
            'B': 30.0,
            'CCC': 50.0,
            'Unrated': 15.0,
            'Defaulted': 100.0
        }
        
    def calculate_pnl(self) -> pd.DataFrame:
        """
        Step 2a: Calculate P&L for each security
        P&L = Market Value - Notional
        
        Returns:
            DataFrame with P&L column added
        """
        self.df['PnL'] = self.df['Market Value$'] - self.df['Notional exposure$']
        return self.df
    
    def calculate_gross_jtd(self) -> pd.DataFrame:
        """
        Step 2b: Calculate Gross JTD for each security
        
        For Long positions: Gross JTD(Long) = max(LGD × Notional + PnL, 0)
        For Short positions: Gross JTD(Short) = min(LGD × Notional + PnL, 0)
        
        Returns:
            DataFrame with Gross_JTD column
        """
        # Convert LGD from percentage to decimal
        lgd_decimal = self.df['LGD%'] / 100
        
        # Calculate base JTD
        base_jtd = lgd_decimal * self.df['Notional exposure$'] + self.df['PnL']
        
        # Apply max/min based on Long/Short
        self.df['Gross_JTD_unscaled'] = np.where(
            self.df['Long/Short'] == 'Long',
            np.maximum(base_jtd, 0),  # Long: max(..., 0)
            np.minimum(base_jtd, 0)   # Short: min(..., 0)
        )
        
        return self.df
    
    def calculate_year_fraction(self) -> pd.DataFrame:
        """
        Step 2b.iii: Calculate year fraction of expected maturity
        
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
        Step 2b.iv: Scale each Gross JTD by year fraction
        This is the final Gross JTD
        
        Returns:
            DataFrame with scaled Gross_JTD
        """
        self.df['Gross_JTD'] = self.df['Gross_JTD_unscaled'] * self.df['Year_Fraction']
        return self.df
    
    def can_net(self, long_seniority: str, short_seniority: str) -> bool:
        """
        Check if short position can offset long position based on seniority.
        
        Rule: Short exposure must have same or LOWER seniority than long exposure.
        Seniority hierarchy: Covered > Senior > Non Senior
        
        Args:
            long_seniority: Seniority of long position
            short_seniority: Seniority of short position
            
        Returns:
            True if netting is allowed, False otherwise
        """
        # Short can net if its rank <= long's rank (i.e., short is equal or less senior)
        return self.seniority_rank[short_seniority] <= self.seniority_rank[long_seniority]
    
    def calculate_net_jtd_for_issuer(self, issuer_df: pd.DataFrame) -> List[Dict]:
        """
        Step 3: Calculate Net JTD for a single issuer
        
        Rules:
        - If all positions are same sign (all long or all short): simply add them
        - If mixed signs: check seniority hierarchy
          - If short has lower/equal seniority than long: allow netting (add them)
          - If short has higher seniority than long: keep as separate Net JTDs
        
        Args:
            issuer_df: DataFrame containing all positions for one issuer
            
        Returns:
            List of Net JTD dictionaries
        """
        net_jtds = []
        
        # Separate long and short positions
        long_positions = issuer_df[issuer_df['Gross_JTD'] > 0].copy()
        short_positions = issuer_df[issuer_df['Gross_JTD'] < 0].copy()
        
        issuer_name = issuer_df['Issuer'].iloc[0]
        bucket = issuer_df['Bucket'].iloc[0]
        risk_weight = issuer_df['Risk weight'].iloc[0]
        
        # Case 1: Only long positions - simply add them
        if len(short_positions) == 0:
            if len(long_positions) > 0:
                net_jtds.append({
                    'Issuer': issuer_name,
                    'Bucket': bucket,
                    'Risk_Weight': risk_weight,
                    'Net_JTD': long_positions['Gross_JTD'].sum(),
                    'Direction': 'Long'
                })
            return net_jtds
        
        # Case 2: Only short positions - simply add them (preserving negative sign)
        if len(long_positions) == 0:
            net_jtds.append({
                'Issuer': issuer_name,
                'Bucket': bucket,
                'Risk_Weight': risk_weight,
                'Net_JTD': short_positions['Gross_JTD'].sum(),
                'Direction': 'Short'
            })
            return net_jtds
        
        # Case 3: Mixed long and short - check seniority for netting
        # Check if ANY short can net with ANY long
        netting_allowed = False
        for _, long_pos in long_positions.iterrows():
            for _, short_pos in short_positions.iterrows():
                if self.can_net(long_pos['Seniority'], short_pos['Seniority']):
                    netting_allowed = True
                    break
            if netting_allowed:
                break
        
        if netting_allowed:
            # Net all longs and shorts together
            total_long = long_positions['Gross_JTD'].sum()
            total_short = short_positions['Gross_JTD'].sum()
            net_value = total_long + total_short  # Short is negative, so this subtracts
            
            if abs(net_value) > 1e-10:  # Avoid floating point zero issues
                direction = 'Long' if net_value > 0 else 'Short'
                net_jtds.append({
                    'Issuer': issuer_name,
                    'Bucket': bucket,
                    'Risk_Weight': risk_weight,
                    'Net_JTD': net_value,
                    'Direction': direction
                })
        else:
            # Netting not allowed - keep longs and shorts as separate Net JTDs
            net_jtds.append({
                'Issuer': issuer_name,
                'Bucket': bucket,
                'Risk_Weight': risk_weight,
                'Net_JTD': long_positions['Gross_JTD'].sum(),
                'Direction': 'Long'
            })
            net_jtds.append({
                'Issuer': issuer_name,
                'Bucket': bucket,
                'Risk_Weight': risk_weight,
                'Net_JTD': short_positions['Gross_JTD'].sum(),
                'Direction': 'Short'
            })
        
        return net_jtds
    
    def calculate_all_net_jtds(self) -> pd.DataFrame:
        """
        Step 3: Calculate Net JTDs for all issuers
        
        Returns:
            DataFrame of Net JTDs
        """
        all_net_jtds = []
        
        # Group by issuer and calculate Net JTD for each
        for issuer, group in self.df.groupby('Issuer'):
            net_jtds = self.calculate_net_jtd_for_issuer(group)
            all_net_jtds.extend(net_jtds)
        
        self.net_jtd_df = pd.DataFrame(all_net_jtds)
        return self.net_jtd_df
    
    def calculate_hbr_for_bucket(self, bucket_df: pd.DataFrame) -> float:
        """
        Step 5: Calculate Hedge Benefit Ratio (HBR) for a bucket
        
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
        Step 6: Calculate DRC for a bucket
        
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
        Steps 4-7: Calculate DRC for all buckets and aggregate
        
        Returns:
            Tuple of (total_drc, bucket_details)
        """
        bucket_details = {}
        total_drc = 0
        
        # Process each bucket
        for bucket_name in self.net_jtd_df['Bucket'].unique():
            bucket_df = self.net_jtd_df[self.net_jtd_df['Bucket'] == bucket_name]
            
            # Calculate HBR for this bucket
            hbr = self.calculate_hbr_for_bucket(bucket_df)
            
            # Calculate DRC for this bucket
            bucket_drc = self.calculate_bucket_drc(bucket_df, hbr)
            
            # Store details
            bucket_details[bucket_name] = {
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
        Execute complete DRC calculation workflow
        
        Returns:
            Tuple of (total_drc, bucket_details, gross_jtd_df, net_jtd_df)
        """
        print("Step 1: Loading holdings data...")
        print(f"Loaded {len(self.df)} positions\n")
        
        print("Step 2: Calculating Gross JTD...")
        self.calculate_pnl()
        self.calculate_gross_jtd()
        self.calculate_year_fraction()
        self.scale_gross_jtd()
        print(f"Calculated Gross JTD for {len(self.df)} positions\n")
        
        print("Step 3: Calculating Net JTD...")
        self.calculate_all_net_jtds()
        print(f"Calculated {len(self.net_jtd_df)} Net JTD positions\n")
        
        print("Steps 4-7: Calculating DRC by bucket...")
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
        print(f"TOTAL DRC NON-SECURITIZATION CAPITAL CHARGE: {total_drc:,.2f}")
        print("="*60)
        
        return total_drc, bucket_details, self.df, self.net_jtd_df


def main():
    """
    Main execution function
    """
    # Initialize calculator with holdings file
    calculator = DRCCalculator("Default Risk charge holdings.xlsx")
    
    # Run full calculation
    total_drc, bucket_details, gross_jtd_df, net_jtd_df = calculator.run_full_calculation()
    
    # Optional: Save detailed results to Excel
    with pd.ExcelWriter('DRC_Results.xlsx') as writer:
        gross_jtd_df.to_excel(writer, sheet_name='Gross_JTD', index=False)
        net_jtd_df.to_excel(writer, sheet_name='Net_JTD', index=False)
        
        # Bucket summary
        bucket_summary = pd.DataFrame(bucket_details).T
        bucket_summary.to_excel(writer, sheet_name='Bucket_Summary')
    
    print("\nDetailed results saved to 'DRC_Results.xlsx'")


if __name__ == "__main__":
    main()
