# cli.py
import argparse
from datetime import datetime
from storage import append_transaction_csv, read_transactions_csv
from models import Transaction
from analysis import txs_to_df, daily_summary, detect_anomalies_iqr
from viz import plot_time_series, highlight_anomalies
import matplotlib.pyplot as plt
import sys

CSV_PATH = "my_budget.csv"

def cmd_add(args):
    # parse timestamp or use now
    if args.timestamp:
        ts = datetime.fromisoformat(args.timestamp)
    else:
        ts = datetime.now()
    tx = Transaction(timestamp=ts, amount=args.amount, category=args.category, note=args.note)
    append_transaction_csv(CSV_PATH, tx)
    print(f"Saved: {tx}")

def cmd_show(args):
    txs = read_transactions_csv(CSV_PATH)
    df = txs_to_df(txs)
    if df.empty:
        print("No transactions yet.")
        return
    # show daily summary
    daily = daily_summary(df)
    anomalies = detect_anomalies_iqr(daily)
    ax1 = plot_time_series(df)
    ax2 = highlight_anomalies(anomalies)
    plt.show()

def build_parser():
    p = argparse.ArgumentParser("budgeter")
    sub = p.add_subparsers(dest="cmd")
    a = sub.add_parser("add")
    a.add_argument("--timestamp", help="ISO timestamp, e.g. 2025-08-29T20:00", default=None)
    a.add_argument("amount", type=float, help="Amount spent (positive number).")
    a.add_argument("category", help="Category string, e.g. groceries")
    a.add_argument("--note", default="", help="Optional note")
    s = sub.add_parser("show")
    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "add":
        cmd_add(args)
    elif args.cmd == "show":
        cmd_show(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
