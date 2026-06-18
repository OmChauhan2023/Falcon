import pandas as pd
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
LEDGER_FILE = os.path.join(OUTPUT_DIR, "reports", "trade_ledger.csv")
REPORT_FILE = os.path.join(OUTPUT_DIR, "ledger_mining", "mining_report.txt")

def analyze_ledger():
    if not os.path.exists(LEDGER_FILE):
        print("Trade Ledger not found!")
        return

    # Redirect print to both console and file
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w", encoding='utf-8')
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            self.terminal.flush()
            self.log.flush()
            
    sys.stdout = Logger(REPORT_FILE)

    df = pd.read_csv(LEDGER_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M')

    print("\n" + "="*70)
    print("  TRADE LEDGER MINING REPORT")
    print("="*70)

    # 1. Segment-level (Monthly) Performance
    print("\n1. MONTHLY (SEGMENT) PERFORMANCE")
    monthly = df.groupby('Month').agg(
        Trades=('R', 'count'),
        Avg_R=('R', 'mean'),
        Win_Rate=('Net_PnL', lambda x: (x > 0).mean() * 100),
        Total_R=('R', 'sum'),
        Total_PnL=('Net_PnL', 'sum')
    )
    for idx, row in monthly.iterrows():
        print(f"  {idx} | Trades: {row['Trades']:<3} | Avg R: {row['Avg_R']:>5.2f} | Win Rate: {row['Win_Rate']:>5.1f}% | Total PnL: Rs{row['Total_PnL']:,.0f}")

    # 2. Session-level Performance
    print("\n2. SESSION PERFORMANCE")
    session = df.groupby('Session').agg(
        Trades=('R', 'count'),
        Avg_R=('R', 'mean'),
        Win_Rate=('Net_PnL', lambda x: (x > 0).mean() * 100),
        Total_R=('R', 'sum'),
        Total_PnL=('Net_PnL', 'sum')
    )
    for idx, row in session.iterrows():
        print(f"  {idx:<8} | Trades: {row['Trades']:<3} | Avg R: {row['Avg_R']:>5.2f} | Win Rate: {row['Win_Rate']:>5.1f}% | Total PnL: Rs{row['Total_PnL']:,.0f}")

    # 3. Regime-level Performance
    print("\n3. VOLATILITY REGIME PERFORMANCE")
    regime = df.groupby('Regime').agg(
        Trades=('R', 'count'),
        Avg_R=('R', 'mean'),
        Win_Rate=('Net_PnL', lambda x: (x > 0).mean() * 100),
        Total_R=('R', 'sum'),
        Total_PnL=('Net_PnL', 'sum')
    )
    for idx, row in regime.iterrows():
        print(f"  {idx:<8} | Trades: {row['Trades']:<3} | Avg R: {row['Avg_R']:>5.2f} | Win Rate: {row['Win_Rate']:>5.1f}% | Total PnL: Rs{row['Total_PnL']:,.0f}")

    # 4. Probability Decile Performance
    print("\n4. PROBABILITY DECILE PERFORMANCE")
    # We create strictly separated buckets
    bins = [0.65, 0.67, 0.70, 0.73, 0.75, 1.0]
    labels = ['0.65-0.67', '0.67-0.70', '0.70-0.73', '0.73-0.75', '0.75+']
    
    # We take max of prob and (1-prob) because we use short probabilities (<0.35) which map to (>0.65) confidence
    df['Confidence'] = np.where(df['Probability'] < 0.5, 1 - df['Probability'], df['Probability'])
    
    df['Prob_Bucket'] = pd.cut(df['Confidence'], bins=bins, labels=labels, include_lowest=True)
    
    prob_perf = df.groupby('Prob_Bucket').agg(
        Trades=('R', 'count'),
        Avg_R=('R', 'mean'),
        Win_Rate=('Net_PnL', lambda x: (x > 0).mean() * 100),
        Total_R=('R', 'sum')
    )
    for idx, row in prob_perf.iterrows():
        if row['Trades'] > 0:
            print(f"  {idx:<9} | Trades: {row['Trades']:<3} | Avg R: {row['Avg_R']:>5.2f} | Win Rate: {row['Win_Rate']:>5.1f}% | Total R: {row['Total_R']:>5.2f}R")

    # 5. The Golden Cross-Section (Finding the 80/20)
    print("\n5. THE 80/20 TRADES (TOP 3 REGIME+SESSION CLUSTERS)")
    cross = df.groupby(['Session', 'Regime']).agg(
        Trades=('R', 'count'),
        Avg_R=('R', 'mean'),
        Total_R=('R', 'sum')
    ).sort_values('Total_R', ascending=False)
    
    for idx, row in cross.head(3).iterrows():
        print(f"  {idx[0]:<7} + {idx[1]:<8} | Trades: {row['Trades']:<3} | Avg R: {row['Avg_R']:>5.2f} | Total R: {row['Total_R']:>5.2f}R")
        
    print("\n" + "="*70)

if __name__ == "__main__":
    analyze_ledger()
