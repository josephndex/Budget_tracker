# Budget Tracker — The Ultimate Personal Finance App

## Features

- **Modern Web UI**: Streamlit-powered, responsive, and intuitive.
- **Transaction Management**: Add, edit, delete, and import/export transactions.
- **Category Management**: Add/edit/delete categories, assign colors and budgets.
- **Savings Tracking**: Tag transactions as savings, set savings goals, and visualize progress.
- **Budgets**: Set monthly budgets per category or overall, with progress bars and alerts.
- **Advanced Analytics**: Daily/weekly/monthly summaries, rolling stats, anomaly detection (IQR, IsolationForest).
- **Visualizations**: Time series, bar charts, pie charts, heatmaps, anomaly highlights.
- **Import/Export**: CSV import/export, Google Sheets sync (optional).
- **Error Handling**: Robust input validation and user guidance.

## Setup

1. **Install Python 3.8+**
2. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```
3. **(Optional) For Google Sheets sync:**
   - Place your `service_account.json` in the app folder.
   - Share your Google Sheet with the service account email.

## Usage

1. **Start the app:**
   ```sh
   streamlit run streamlit_app.py
   ```
2. **Open the app** in your browser (usually at http://localhost:8501)

## Main Features Guide

### Transactions
- **Add**: Use the sidebar or main form to add new transactions.
- **Edit/Delete**: Use the table at the bottom, select a row, and use the edit/delete form.
- **Import**: Use the sidebar to upload a CSV and map columns.
- **Export**: Download filtered transactions as CSV.

### Categories
- **Manage**: Use the sidebar expander to add, edit, or delete categories. Assign colors and budgets.
- **Assign**: When adding/editing transactions, select a category.

### Savings
- **Tag**: Any transaction with 'savings' in the category or note is treated as savings.
- **Goal**: Set a savings goal and target date in the sidebar. Progress and ETA are shown.

### Budgets
- **Set**: Assign monthly budgets per category or overall in the sidebar.
- **Track**: Progress bars and alerts show your spending vs. budget.

### Analytics & Visuals
- **KPIs**: See total spent, saved, transaction count, and period.
- **Charts**: Time series, rolling averages, category breakdown, heatmaps.
- **Anomalies**: Detect outliers in spending with IQR or IsolationForest.

### Google Sheets Sync (Optional)
- Place `service_account.json` in the folder.
- Enter your Google Sheet ID and click to sync filtered transactions.

## Tips
- Use categories and notes to organize and analyze your spending.
- Tag savings consistently for accurate tracking.
- Adjust budgets and goals as your finances evolve.

## Troubleshooting
- If you see missing package errors, run `pip install -r requirements.txt` again.
- For Google Sheets, ensure your service account has access to the sheet.

---
Enjoy your new, powerful budget tracker!
