import pandas as pd

from database import ROOT


CUSTOMERS_PATH = ROOT / "data" / "clean" / "customers.csv"
ORDERS_PATH = ROOT / "data" / "clean" / "orders.csv"
AS_OF = pd.Timestamp("2025-09-30")


def load_inputs():
    customers = pd.read_csv(CUSTOMERS_PATH, parse_dates=["signup_date"])
    orders = pd.read_csv(ORDERS_PATH, parse_dates=["order_ts"])
    completed = orders.loc[orders["status"].eq("completed")].copy()
    return customers, orders, completed


def build_rfm(customers, completed, as_of=AS_OF):
    features = (
        completed.groupby("customer_id")
        .agg(
            last_order_ts=("order_ts", "max"),
            frequency=("order_id", "count"),
            monetary=("order_amount", "sum"),
            average_order_value=("order_amount", "mean"),
        )
        .reset_index()
    )
    features["recency_days"] = (as_of - features["last_order_ts"].dt.normalize()).dt.days
    result = customers[["customer_id", "city", "age", "gender", "signup_date"]].merge(
        features, on="customer_id", how="left"
    )
    result["frequency"] = result["frequency"].fillna(0).astype(int)
    result["monetary"] = result["monetary"].fillna(0.0)
    result["average_order_value"] = result["average_order_value"].fillna(0.0)
    result["recency_days"] = result["recency_days"].fillna((as_of - result["signup_date"]).dt.days)
    return result
