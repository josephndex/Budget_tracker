# analysis.py
import pandas as pd
import numpy as np
from typing import List
from models import Transaction

def txs_to_df(txs: List[Transaction]) -> pd.DataFrame:
    df = pd.DataFrame([t.to_dict() for t in txs])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["amount"] = df["amount"].astype(float)
    return df

def daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index("timestamp")
    daily = df["amount"].resample("D").sum().rename("total_spent")
    daily = daily.reset_index()
    return daily

def weekly_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index("timestamp")
    weekly = df["amount"].resample("W-MON").sum().rename("total_spent")
    weekly = weekly.reset_index()
    return weekly

def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index("timestamp")
    monthly = df["amount"].resample("M").sum().rename("total_spent")
    monthly = monthly.reset_index()
    return monthly

def rolling_stats(df: pd.DataFrame, window_days:int=7) -> pd.Series:
    d = daily_summary(df).set_index("timestamp")
    return d["total_spent"].rolling(window=window_days, min_periods=1).mean()

def detect_anomalies_iqr(df_daily: pd.DataFrame, factor=1.5):
    """
    IQR-based anomaly detection on daily totals.
    Returns a DataFrame with 'is_anomaly' boolean column.
    """
    s = df_daily["total_spent"]
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    df = df_daily.copy()
    df["is_anomaly"] = (df["total_spent"] < lower) | (df["total_spent"] > upper)
    return df
