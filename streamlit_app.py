# streamlit_app.py
"""
Interactive Budget Tracker (Streamlit) — with Savings tracking.

Assumes storage.py, models.py and analysis.py from prior steps in the same folder.
Saves/reads my_budget.csv via append_transaction_csv/read_transactions_csv.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
import io
import math
import os

# Import your existing modules (from previous step)
from storage import append_transaction_csv, read_transactions_csv, ensure_csv, read_categories, add_category, update_category, delete_category, save_categories
from models import Transaction, Category
from analysis import txs_to_df, daily_summary, detect_anomalies_iqr, rolling_stats


CSV_PATH = "my_budget.csv"
CATEGORY_CSV = "categories.csv"
ensure_csv(CSV_PATH)

st.set_page_config(page_title="Budget Tracker (with Savings)", layout="wide", initial_sidebar_state="expanded")

# -----------------------
# Helper functions (savings-related)
# -----------------------
def df_from_storage():
    txs = read_transactions_csv(CSV_PATH)
    df = txs_to_df(txs)
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def categories_from_storage():
    return read_categories(CATEGORY_CSV)

def is_savings_row(row):
    """
    Determine whether a transaction should be considered a savings event.
    Rules (best-effort):
     - category contains the token 'savings' (case-insensitive), OR
     - note contains 'savings' (case-insensitive)
    """
    try:
        cat = str(row.get('category','') or '').lower()
        note = str(row.get('note','') or '').lower()
        return 'savings' in cat or 'savings' in note
    except Exception:
        return False

def add_transaction_ui():
    st.header("Add transaction")
    col1, col2, col3 = st.columns(3)
    with col1:
        amount = st.number_input("Amount", min_value=0.0, format="%.2f")
    with col2:
        category = st.text_input("Category", value="uncategorized")
    with col3:
        ts = st.datetime_input("Timestamp", value=datetime.now())
    note = st.text_input("Note (optional)")
    if st.button("Save transaction"):
        tx = Transaction(timestamp=ts, amount=float(amount), category=category, note=note)
        append_transaction_csv(CSV_PATH, tx)
        st.success("Saved transaction")
        st.experimental_rerun()

def apply_filters(df, start_date, end_date, categories):
    if df.empty:
        return df
    df = df[(df['timestamp'] >= pd.to_datetime(start_date)) & (df['timestamp'] <= pd.to_datetime(end_date))]
    if categories:
        df = df[df['category'].isin(categories)]
    return df

def isolation_forest_anomalies(df, contamination=0.02):
    if df.empty or len(df) < 20:
        return pd.DataFrame()
    X = pd.DataFrame()
    X['amount'] = df['amount'].values
    X['hour'] = df['timestamp'].dt.hour
    X['dow'] = df['timestamp'].dt.dayofweek
    clf = IsolationForest(contamination=contamination, random_state=42)
    labels = clf.fit_predict(X)
    df2 = df.copy()
    df2['if_anomaly'] = labels == -1
    return df2

def savings_summary(df):
    """
    Returns:
      - total_savings (sum)
      - monthly_savings_df (timestamp month-end, total saved that month)
      - cumulative_df (date, cumulative sum)
    """
    if df.empty:
        return 0.0, pd.DataFrame(), pd.DataFrame()
    df_sav = df[df['is_savings']]
    if df_sav.empty:
        return 0.0, pd.DataFrame(), pd.DataFrame()
    total_sav = df_sav['amount'].sum()

    # monthly savings
    df_sav = df_sav.set_index('timestamp')
    monthly = df_sav['amount'].resample('M').sum().rename('monthly_saved').reset_index()

    # daily cumulative (for plotting)
    daily = df_sav['amount'].resample('D').sum().cumsum().rename('cumulative_saved').reset_index()

    return float(total_sav), monthly, daily

def months_between(d1, d2):
    """Return number of months from d1 to d2 (approx integer)."""
    return max(0, (d2.year - d1.year) * 12 + (d2.month - d1.month))

# -----------------------
# Sidebar controls (including savings goal)
# -----------------------
# Sidebar controls (including savings goal)
# -----------------------

# --- Sidebar: Controls ---
st.sidebar.title("Controls")
st.sidebar.markdown("Filter, tune anomaly detection, and set savings goals.")

df_all = df_from_storage()
categories = categories_from_storage()
cat_names = [c.name for c in categories]
cat_colors = {c.name: c.color for c in categories}

# Category management UI
with st.sidebar.expander("Manage categories", expanded=False):
    st.markdown("**Add new category**")
    with st.form("add_cat_form", clear_on_submit=True):
        new_cat = st.text_input("Category name", key="new_cat")
        new_color = st.color_picker("Color", value="#1976d2", key="new_color")
        new_budget = st.number_input("Budget (optional)", min_value=0.0, value=0.0, key="new_budget")
        add_cat_btn = st.form_submit_button("Add category")
        if add_cat_btn and new_cat:
            if add_category(Category(name=new_cat, color=new_color, budget=new_budget)):
                st.success(f"Added category {new_cat}")
                st.experimental_rerun()
            else:
                st.warning("Category already exists.")
    st.markdown("**Edit/delete categories**")
    for c in categories:
        col1, col2, col3 = st.columns([2,1,1])
        with col1:
            new_name = st.text_input(f"Name_{c.name}", value=c.name, key=f"edit_name_{c.name}")
        with col2:
            new_color = st.color_picker(f"Color_{c.name}", value=c.color, key=f"edit_color_{c.name}")
        with col3:
            new_budget = st.number_input(f"Budget_{c.name}", min_value=0.0, value=float(c.budget), key=f"edit_budget_{c.name}")
        col4, col5 = st.columns([1,1])
        with col4:
            if st.button(f"Update_{c.name}", key=f"update_{c.name}"):
                update_category(Category(name=new_name, color=new_color, budget=new_budget))
                st.success(f"Updated {new_name}")
                st.experimental_rerun()
        with col5:
            if st.button(f"Delete_{c.name}", key=f"delete_{c.name}"):
                delete_category(c.name)
                st.warning(f"Deleted {c.name}")
                st.experimental_rerun()

# Date range defaults
if df_all.empty:
    default_end = datetime.now()
    default_start = default_end - timedelta(days=30)
else:
    default_end = df_all['timestamp'].max()
    default_start = df_all['timestamp'].min()

start_date = st.sidebar.date_input("Start date", default_start.date())
end_date = st.sidebar.date_input("End date", default_end.date())

# categories multiselect
all_categories = sorted(set(cat_names + (df_all['category'].dropna().unique().tolist() if not df_all.empty else [])))
selected_categories = st.sidebar.multiselect("Categories (empty = all)", options=all_categories, default=all_categories)

# rolling window
rolling_window = st.sidebar.slider("Rolling window (days)", 1, 30, 7)

# anomaly detection options
st.sidebar.subheader("Anomaly detection")
anomaly_method = st.sidebar.selectbox("Method", ["IQR (daily totals)", "IsolationForest (transactions)"])
if anomaly_method == "IsolationForest (transactions)":
    contamination = st.sidebar.slider("Contamination (fraction)", 0.001, 0.2, 0.02)
else:
    contamination = None

# budgets & savings
st.sidebar.subheader("Budgets & Savings")
budget_category = st.sidebar.selectbox("Budget category", options=["__all__"] + all_categories)
budget_monthly = st.sidebar.number_input("Monthly budget amount (selected category)", min_value=0.0, value=0.0)

st.sidebar.markdown("## Savings goal")
savings_goal_amount = st.sidebar.number_input("Goal amount (total)", min_value=0.0, value=0.0)
savings_target_date = st.sidebar.date_input("Target date (optional)", value=(datetime.now().date() + timedelta(days=30)))

# data import/export
st.sidebar.subheader("Import / Export")
uploaded = st.sidebar.file_uploader("Upload bank CSV to import", type=["csv"])
st.sidebar.markdown("Download filtered CSV below.")

# Quick-add form in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("Quick add a transaction")
with st.sidebar.form("quick_add", clear_on_submit=True):
    q_amount = st.number_input("Amount", min_value=0.0, format="%.2f", key="q_amount")
    q_cat = st.selectbox("Category", options=all_categories, key="q_cat")
    now = datetime.now()
    q_date = st.date_input("Date", value=now.date(), key="q_date")
    q_time = st.time_input("Time", value=now.time(), key="q_time")
    q_note = st.text_input("Note", key="q_note")
    submitted = st.form_submit_button("Add")
    if submitted:
        q_ts = datetime.combine(q_date, q_time)
        tx = Transaction(timestamp=q_ts, amount=float(q_amount), category=q_cat, note=q_note)
        append_transaction_csv(CSV_PATH, tx)
        st.success("Added!")
        st.experimental_rerun()

# -----------------------
# Top-level UI
# -----------------------
st.title("Interactive Budget Tracker — Savings-enabled")
st.markdown("Transactions categorized with **'savings'** (category or note) are treated as savings and tracked separately.")

# Import flow (same as before)
if uploaded is not None:
    try:
        uploaded_df = pd.read_csv(uploaded)
        st.info("Preview of uploaded file (first rows)")
        st.dataframe(uploaded_df.head())
        with st.expander("Column mapping and import"):
            st.write("Map columns in your uploaded CSV to expected columns: `timestamp`, `amount`, `category`, `note`")
            col_ts = st.selectbox("Timestamp column", options=uploaded_df.columns.tolist())
            col_amt = st.selectbox("Amount column", options=uploaded_df.columns.tolist())
            col_cat = st.selectbox("Category column", options=uploaded_df.columns.tolist())
            col_note = st.selectbox("Note column", options=uploaded_df.columns.tolist())
            if st.button("Import mapped rows"):
                imported = 0
                for _, row in uploaded_df.iterrows():
                    try:
                        ts = pd.to_datetime(row[col_ts])
                        amt = float(row[col_amt])
                        cat = str(row[col_cat]) if col_cat in uploaded_df.columns else "imported"
                        note = str(row[col_note]) if col_note in uploaded_df.columns else ""
                        tx = Transaction(timestamp=ts.to_pydatetime(), amount=amt, category=cat, note=note)
                        append_transaction_csv(CSV_PATH, tx)
                        imported += 1
                    except Exception:
                        continue
                st.success(f"Imported {imported} rows.")
                st.experimental_rerun()
    except Exception as e:
        st.error("Failed to read uploaded CSV: " + str(e))

# Apply filters to the main dataframe
df = apply_filters(df_all, pd.to_datetime(start_date), pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1), selected_categories)

# Add an 'is_savings' flag
if not df.empty:
    df['is_savings'] = df.apply(is_savings_row, axis=1)
else:
    df['is_savings'] = pd.Series(dtype=bool)


# KPIs: show total spent (excluding savings) and total saved
col1, col2, col3, col4 = st.columns(4)
total_savings, monthly_savings_df, cumulative_savings_df = savings_summary(df)
total_spent_excl_savings = float(df[~df['is_savings']]['amount'].sum()) if not df.empty else 0.0
avg_daily = daily_summary(df[~df['is_savings']])['total_spent'].mean() if not df.empty else 0.0
num_tx = len(df) if not df.empty else 0

col1.metric("Total spent (excl. savings)", f"{total_spent_excl_savings:,.2f}")
col2.metric("Total saved", f"{total_savings:,.2f}")
col3.metric("Transactions (filtered)", f"{num_tx}")
col4.metric("Period", f"{start_date} → {end_date}")

# --- Transaction table with edit/delete ---
st.subheader("Filtered transactions (table, editable)")
df_sorted = df.sort_values('timestamp', ascending=False).reset_index(drop=True)
edit_idx = st.number_input("Row # to edit/delete (0 = top)", min_value=0, max_value=max(0, len(df_sorted)-1), value=0, step=1)
if len(df_sorted) > 0:
    row = df_sorted.iloc[edit_idx]
    with st.expander(f"Edit/delete transaction #{edit_idx}", expanded=False):
        with st.form(f"edit_tx_{row['id']}"):
            new_amt = st.number_input("Amount", value=float(row['amount']), key=f"edit_amt_{row['id']}")
            new_cat = st.selectbox("Category", options=all_categories, index=all_categories.index(row['category']) if row['category'] in all_categories else 0, key=f"edit_cat_{row['id']}")
            old_dt = pd.to_datetime(row['timestamp'])
            new_date = st.date_input("Date", value=old_dt.date(), key=f"edit_date_{row['id']}")
            new_time = st.time_input("Time", value=old_dt.time(), key=f"edit_time_{row['id']}")
            new_ts = datetime.combine(new_date, new_time)
            new_note = st.text_input("Note", value=row['note'], key=f"edit_note_{row['id']}")
            update_btn = st.form_submit_button("Update")
            delete_btn = st.form_submit_button("Delete")
            if update_btn:
                # Update: remove old, add new
                txs = read_transactions_csv(CSV_PATH)
                txs = [t for t in txs if t.id != row['id']]
                txs.append(Transaction(id=row['id'], timestamp=new_ts, amount=new_amt, category=new_cat, note=new_note))
                # Save all
                pd.DataFrame([t.to_dict() for t in txs]).to_csv(CSV_PATH, index=False)
                st.success("Transaction updated.")
                st.experimental_rerun()
            if delete_btn:
                txs = read_transactions_csv(CSV_PATH)
                txs = [t for t in txs if t.id != row['id']]
                pd.DataFrame([t.to_dict() for t in txs]).to_csv(CSV_PATH, index=False)
                st.warning("Transaction deleted.")
                st.experimental_rerun()
st.dataframe(df_sorted)

# -----------------------
# Charts area
# -----------------------
st.subheader("Spending & Savings overview")
if df.empty:
    st.info("No transactions in the selected range.")
else:
    # Spending time series (exclude savings)
    df_spend = df[~df['is_savings']]
    if not df_spend.empty:
        daily_spend = daily_summary(df_spend)
        daily_spend['timestamp'] = pd.to_datetime(daily_spend['timestamp'])
        daily_spend['rolling'] = daily_spend['total_spent'].rolling(window=rolling_window, min_periods=1).mean()
        fig_spend = px.line(daily_spend, x='timestamp', y='total_spent', title="Daily spend (excl. savings)", labels={'timestamp':'Date','total_spent':'Amount'})
        fig_spend.add_scatter(x=daily_spend['timestamp'], y=daily_spend['rolling'], mode='lines', name=f'{rolling_window}-day rolling')
        st.plotly_chart(fig_spend, use_container_width=True)
    else:
        st.info("No spending transactions (excluding savings) in range.")

    # Savings time series & cumulative
    st.subheader("Savings timeline")
    if total_savings > 0:
        # daily cumulative plot
        cumulative_savings_df['timestamp'] = pd.to_datetime(cumulative_savings_df['timestamp'])
        fig_cum = px.line(cumulative_savings_df, x='timestamp', y='cumulative_saved', title="Cumulative savings over time")
        st.plotly_chart(fig_cum, use_container_width=True)

        # monthly savings bar chart
        if not monthly_savings_df.empty:
            monthly_savings_df['timestamp'] = pd.to_datetime(monthly_savings_df['timestamp'])
            fig_month_sav = px.bar(monthly_savings_df, x='timestamp', y='monthly_saved', title="Monthly savings")
            st.plotly_chart(fig_month_sav, use_container_width=True)
    else:
        st.info("No savings transactions in range. Tag transactions with category or note containing 'savings' to track them here.")

    # Category breakdown (spending + savings visible)
    st.subheader("Category breakdown (all)")
    cat_sum = df.groupby('category')['amount'].sum().reset_index().sort_values('amount', ascending=False)
    fig_cat = px.bar(cat_sum, x='category', y='amount', title="Spend by category (includes savings)")
    st.plotly_chart(fig_cat, use_container_width=True)

    # Heatmap (day vs hour)
    st.subheader("Heatmap: transactions by day of week & hour")
    df_heat = df.copy()
    df_heat['hour'] = df_heat['timestamp'].dt.hour
    df_heat['dow'] = df_heat['timestamp'].dt.day_name()
    heat = df_heat.groupby(['dow','hour'])['amount'].sum().reset_index()
    pivot = heat.pivot(index='dow', columns='hour', values='amount').fillna(0)
    days_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    pivot = pivot.reindex(days_order).fillna(0)
    fig_heat = px.imshow(pivot, labels=dict(x="Hour", y="Day", color="Amount"), title="Heatmap of spending (day vs hour)")
    st.plotly_chart(fig_heat, use_container_width=True)

# -----------------------
# Anomaly detection area
# -----------------------
st.subheader("Anomaly detection")
if df.empty:
    st.info("No data to detect anomalies.")
else:
    if anomaly_method == "IQR (daily totals)":
        # IQR on daily totals (exclude savings or include? we'll run on overall daily totals by default)
        daily_all = daily_summary(df)
        daily_with_flag = detect_anomalies_iqr(daily_all, factor=1.5)
        fig_anom = px.line(daily_with_flag, x='timestamp', y='total_spent', title="Daily totals with IQR anomalies")
        anom_days = daily_with_flag[daily_with_flag['is_anomaly']]
        if not anom_days.empty:
            fig_anom.add_scatter(x=anom_days['timestamp'], y=anom_days['total_spent'], mode='markers', marker=dict(color='red', size=10), name='Anomaly')
        st.plotly_chart(fig_anom, use_container_width=True)
        with st.expander("List anomalous days"):
            st.dataframe(anom_days.sort_values('timestamp', ascending=False).reset_index(drop=True))
    else:
        df_if = isolation_forest_anomalies(df, contamination=contamination)
        if df_if.empty:
            st.info("Not enough data for IsolationForest.")
        else:
            n_anom = int(df_if['if_anomaly'].sum())
            st.write(f"IsolationForest flagged {n_anom} transactions (contamination={contamination})")
            fig_if = px.scatter(df_if, x='timestamp', y='amount', color='if_anomaly', title="Transaction anomalies (IsolationForest)")
            st.plotly_chart(fig_if, use_container_width=True)
            with st.expander("Anomalous transactions"):
                st.dataframe(df_if[df_if['if_anomaly']].sort_values('timestamp', ascending=False))

# -----------------------
# Budgets & savings goal progress
# -----------------------
st.subheader("Budgets & Savings goals")

# Budget progress (same as before)
if budget_monthly > 0:
    today = pd.to_datetime(datetime.now())
    current_month_start = pd.Timestamp(year=today.year, month=today.month, day=1)
    current_month_end = (current_month_start + pd.offsets.MonthEnd(0))
    df_month = df[(df['timestamp'] >= current_month_start) & (df['timestamp'] <= current_month_end)]
    if budget_category == "__all__":
        spent_month = df_month[~df_month['is_savings']]['amount'].sum()  # exclude savings from spending budgets
    else:
        spent_month = df_month[(df_month['category'] == budget_category) & (~df_month['is_savings'])]['amount'].sum()
    pct = min(1.0, spent_month / (budget_monthly + 1e-9))
    st.write(f"Budget for **{budget_category}**: {budget_monthly:.2f} — spent this month (excl. savings): {spent_month:.2f}")
    st.progress(pct)
    if spent_month > budget_monthly:
        st.error("Budget exceeded! 🔥")
    elif spent_month > 0.9 * budget_monthly:
        st.warning("Approaching budget limit.")

# Savings goal progress
if savings_goal_amount > 0:
    saved_to_date = total_savings
    remaining = max(0.0, savings_goal_amount - saved_to_date)
    # average monthly saved (use monthly_savings_df)
    avg_monthly_saved = monthly_savings_df['monthly_saved'].mean() if (not monthly_savings_df.empty) else 0.0
    pct_goal = min(1.0, saved_to_date / (savings_goal_amount + 1e-9))
    st.write(f"Savings goal: **{savings_goal_amount:.2f}** — saved so far: **{saved_to_date:.2f}** — remaining: **{remaining:.2f}**")
    st.progress(pct_goal)
    if saved_to_date >= savings_goal_amount:
        st.success("Goal achieved! 🎉")
    else:
        if avg_monthly_saved > 0:
            months_needed = remaining / avg_monthly_saved
            months_needed = math.ceil(months_needed)
            est_date = (datetime.now() + pd.DateOffset(months=months_needed)).date()
            st.write(f"Avg monthly saved: {avg_monthly_saved:.2f}. Estimated months to reach goal: {months_needed}. ETA ≈ {est_date}")
        else:
            st.info("No monthly savings history yet to estimate ETA. Save some entries first (tag them 'savings').")

# -----------------------
# Data table and download
# -----------------------
st.subheader("Filtered transactions (table)")
st.dataframe(df.sort_values('timestamp', ascending=False).reset_index(drop=True))

# make CSV for download
def to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

csv_bytes = to_csv_bytes(df)
st.download_button("Download filtered CSV", csv_bytes, file_name="filtered_transactions.csv", mime="text/csv")

# Optional: Sync to Google Sheets if service_account.json is present
st.markdown("---")
st.subheader("Optional: Sync to Google Sheet")
if os.path.exists("service_account.json"):
    st.success("Found service_account.json — Sheets sync available.")
    sheet_id = st.text_input("Google Sheet ID (to append)")
    if st.button("Append filtered rows to Google Sheet"):
        try:
            import gspread
            from google.oauth2.service_account import Credentials
            SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
            creds = Credentials.from_service_account_file("service_account.json", scopes=SCOPES)
            client = gspread.authorize(creds)
            sh = client.open_by_key(sheet_id).sheet1
            rows = df[['timestamp','amount','category','note']].astype(str).values.tolist()
            for r in rows:
                sh.append_row(r)
            st.success("Appended to sheet.")
        except Exception as e:
            st.error("Failed to append: " + str(e))
else:
    st.info("Place `service_account.json` in app folder to enable Google Sheets sync.")

st.markdown("App version: interactive Streamlit demo — local CSV backend with savings tracking.")
