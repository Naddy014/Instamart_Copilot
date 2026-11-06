import sqlite3

import pandas as pd

from database import DATABASE, ROOT


def load_queries():
    return [
        query.strip()
        for query in (ROOT / "sql" / "queries.sql").read_text(encoding="utf-8").split(";")
        if query.strip()
    ]


def main():
    with sqlite3.connect(DATABASE) as connection:
        results = [pd.read_sql_query(query, connection) for query in load_queries()]
        for number, result in enumerate(results, start=1):
            print(f"Q{number}: {len(result)} rows")
            print(result.head(3).to_string(index=False))

        customers = pd.read_sql_query("SELECT * FROM customers", connection)
        products = pd.read_sql_query("SELECT * FROM products", connection)
        orders = pd.read_sql_query(
            "SELECT * FROM orders WHERE status='completed'", connection
        )
    detail = orders.merge(customers[["customer_id", "city"]], on="customer_id").merge(
        products[["product_id", "category"]], on="product_id"
    )
    pivot = pd.pivot_table(
        detail,
        values="order_amount",
        index="city",
        columns="category",
        aggfunc="sum",
        fill_value=0,
        margins=True,
        margins_name="All",
    )
    city_summary = detail.groupby("city").agg(
        total=("order_amount", "sum"),
        mean=("order_amount", "mean"),
        count=("order_amount", "count"),
    )
    print("Q2 pandas pivot:")
    print(pivot.round(2).to_string())
    print("City aggregation:")
    print(city_summary.round(2).to_string())
    print("Completed order amount distribution:")
    print(
        detail["order_amount"]
        .agg(["mean", "median", "skew", "min", "max"])
        .to_string()
    )


if __name__ == "__main__":
    main()
