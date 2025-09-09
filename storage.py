# storage.py
import os
import pandas as pd
from filelock import FileLock
from datetime import datetime
from typing import List
from models import Transaction, Category

CSV_COLUMNS = ["id", "timestamp", "amount", "category", "note"]
CATEGORY_CSV = "categories.csv"

def ensure_csv(path: str):
    if not os.path.exists(path):
        df = pd.DataFrame(columns=CSV_COLUMNS)
        df.to_csv(path, index=False)

def ensure_category_csv(path: str = CATEGORY_CSV):
    if not os.path.exists(path):
        df = pd.DataFrame(columns=["name", "color", "budget"])
        df.to_csv(path, index=False)

def append_transaction_csv(path: str, tx: Transaction):
    """
    Appends a single transaction to CSV safely using a file lock.
    """
    ensure_csv(path)
    lock_path = path + ".lock"
    lock = FileLock(lock_path, timeout=5)
    with lock:
        df = pd.DataFrame([tx.to_dict()])
        df.to_csv(path, mode="a", index=False, header=not os.path.getsize(path)>0)
    return True

def read_transactions_csv(path: str) -> List[Transaction]:
    ensure_csv(path)
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.dropna(subset=["timestamp", "amount"], how="all")
    txs = []
    for _, row in df.iterrows():
        txs.append(Transaction.from_dict(row))
    return txs

# Category CRUD
def read_categories(path: str = CATEGORY_CSV) -> list:
    ensure_category_csv(path)
    df = pd.read_csv(path)
    return [Category.from_dict(row) for _, row in df.iterrows()]

def save_categories(categories: list, path: str = CATEGORY_CSV):
    df = pd.DataFrame([c.to_dict() for c in categories])
    df.to_csv(path, index=False)

def add_category(category: Category, path: str = CATEGORY_CSV):
    categories = read_categories(path)
    if any(c.name == category.name for c in categories):
        return False
    categories.append(category)
    save_categories(categories, path)
    return True

def update_category(category: Category, path: str = CATEGORY_CSV):
    categories = read_categories(path)
    for i, c in enumerate(categories):
        if c.name == category.name:
            categories[i] = category
            save_categories(categories, path)
            return True
    return False

def delete_category(name: str, path: str = CATEGORY_CSV):
    categories = read_categories(path)
    categories = [c for c in categories if c.name != name]
    save_categories(categories, path)
    return True
