import sqlite3
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
DATABASE = ROOT / "data" / "instamart.db"


def create_database():
    customers = pd.read_csv(CLEAN / "customers.csv")
    products = pd.read_csv(CLEAN / "products.csv")
    orders = pd.read_csv(CLEAN / "orders.csv")
    orders["order_ts"] = pd.to_datetime(orders["order_ts"]).dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    DATABASE.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DATABASE) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        for table in ("orders", "products", "customers"):
            connection.execute(f"DROP TABLE IF EXISTS {table}")
        customers.to_sql("customers", connection, index=False)
        products.to_sql("products", connection, index=False)
        orders.to_sql("orders", connection, index=False)
        connection.executescript(
            """
            CREATE UNIQUE INDEX idx_customers_id ON customers(customer_id);
            CREATE UNIQUE INDEX idx_products_id ON products(product_id);
            CREATE UNIQUE INDEX idx_orders_id ON orders(order_id);
            CREATE INDEX idx_orders_customer ON orders(customer_id);
            CREATE INDEX idx_orders_product ON orders(product_id);
            """
        )
    print(f"database: {DATABASE.relative_to(ROOT)}")


if __name__ == "__main__":
    create_database()
