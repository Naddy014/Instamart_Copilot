import sqlite3

import pandas as pd

from database import DATABASE, ROOT


def main():
    customers = pd.read_csv(ROOT / "data" / "clean" / "customers.csv")
    products = pd.read_csv(ROOT / "data" / "clean" / "products.csv")
    orders = pd.read_csv(ROOT / "data" / "clean" / "orders.csv")
    completed = orders.loc[orders["status"].eq("completed")]
    cancelled = orders.loc[orders["status"].eq("cancelled")]

    assert len(customers) == 1200
    assert len(products) == 40
    assert len(orders) == 12309
    assert orders["order_id"].is_unique
    assert orders["status"].value_counts().to_dict() == {
        "completed": 11110,
        "cancelled": 1199,
    }
    assert orders["order_amount"].notna().all()
    assert completed["delivery_minutes"].notna().all()
    assert cancelled["delivery_minutes"].isna().all()
    assert round(completed["order_amount"].sum(), 2) == 2998668.80
    assert customers["age"].between(18, 60).all()

    with sqlite3.connect(DATABASE) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 1200
        assert connection.execute("SELECT COUNT(*) FROM products").fetchone()[0] == 40
        assert connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 12309

    print("ALL CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
