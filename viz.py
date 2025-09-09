# viz.py
import matplotlib.pyplot as plt
import pandas as pd

def plot_time_series(df: pd.DataFrame, ax=None, title="Spending over time"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,4))
    df = df.set_index("timestamp").sort_index()
    # plot cumulative or daily totals; here we'll show daily total and 7-day rolling mean
    daily = df["amount"].resample("D").sum()
    rolling = daily.rolling(7, min_periods=1).mean()
    ax.plot(daily.index, daily.values, label="Daily total")
    ax.plot(rolling.index, rolling.values, label="7-day rolling mean", linewidth=2)
    ax.set_title(title)
    ax.set_ylabel("Amount")
    ax.set_xlabel("Date")
    ax.legend()
    plt.tight_layout()
    return ax

def plot_category_pie(df: pd.DataFrame, ax=None, title="Spending by category"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    cat = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    ax.pie(cat.values, labels=cat.index, autopct="%1.1f%%")
    ax.set_title(title)
    return ax

def highlight_anomalies(df_daily_with_flag: pd.DataFrame, ax=None, title="Daily spend with anomalies"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,4))
    dates = pd.to_datetime(df_daily_with_flag["timestamp"])
    ax.plot(dates, df_daily_with_flag["total_spent"], label="Daily total")
    anomalies = df_daily_with_flag[df_daily_with_flag["is_anomaly"]]
    ax.scatter(anomalies["timestamp"], anomalies["total_spent"], color="red", label="Anomaly")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Amount")
    ax.legend()
    plt.tight_layout()
    return ax
