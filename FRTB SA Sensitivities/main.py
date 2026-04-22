"""
FRTB SA — Main Entry Point
Run this file to generate sensitivities for all portfolio instruments.

Usage:
  python main.py

Inputs (place in data/ folder):
  - FRTB_Combined_Portfolio_v4.xlsx    (portfolio holdings)
  - market_data_snapshot_5Feb2026.xlsx (market data + BCOMTR weights)
  - MAR21_Config_RW_Corr.xlsx         (regulatory config)

Output:
  - output/FRTB_Sensitivities.xlsx     (one row per instrument, all risk factors)
"""
import os
import sys

# Add project root to path so modules can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator import compute_all_sensitivities
from output_writer import write_sensitivity_output


def main():
    # ── File paths ──────────────────────────────────────────
    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    portfolio_path = os.path.join(DATA_DIR, "FRTB_Combined_Portfolio_v4.xlsx")
    market_data_path = os.path.join(DATA_DIR, "market_data_snapshot_5Feb2026.xlsx")
    config_path = os.path.join(DATA_DIR, "MAR21_Config_RW_Corr.xlsx")
    output_path = os.path.join(OUTPUT_DIR, "FRTB_Sensitivities.xlsx")
    
    # ── Validate inputs exist ───────────────────────────────
    for fp, desc in [(portfolio_path, "Portfolio"), 
                     (market_data_path, "Market Data"),
                     (config_path, "MAR21 Config")]:
        if not os.path.exists(fp):
            print(f"ERROR: {desc} file not found: {fp}")
            print(f"       Place your files in the '{DATA_DIR}' folder.")
            sys.exit(1)
    
    # ── Run ─────────────────────────────────────────────────
    print("=" * 60)
    print("FRTB SA Sensitivity Engine")
    print("=" * 60)
    
    print("\n[1/2] Computing sensitivities...")
    df = compute_all_sensitivities(portfolio_path, market_data_path, config_path)
    
    print(f"\n[2/2] Writing output...")
    write_sensitivity_output(df, output_path)
    
    print("\n" + "=" * 60)
    print(f"Done. Output: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
