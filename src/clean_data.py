import pandas as pd

from database import ROOT


RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"


def clean_customers():
    customers = pd.read_csv(RAW / "customers.csv")
    customers["city"] = customers["city"].fillna("").str.strip().str.title()
    customers["gender"] = customers["gender"].fillna("").str.strip().str.casefold()
    customers["gender"] = customers["gender"].replace(
        {"m": "male", "male": "male", "f": "female", "female": "female", "": "unknown"}
    )
    customers["gender"] = customers["gender"].where(
        customers["gender"].isin(["male", "female"]), "unknown"
    )
    customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
    customers.loc[~customers["age"].between(10, 100), "age"] = pd.NA
    customers["age"] = customers["age"].fillna(
        customers.groupby("city")["age"].transform("median")
    )
    customers["age"] = customers["age"].astype("int64")
    customers["signup_date"] = pd.to_datetime(customers["signup_date"], errors="raise")
    return customers


def clean_products():
    products = pd.read_csv(RAW / "products.csv")
    products["price_inr"] = pd.to_numeric(products["price_inr"], errors="raise")
    return products


def clean_orders(products):
    orders = pd.read_csv(RAW / "orders.csv")
    orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="raise")
    orders = orders.drop_duplicates()
    orders = orders.sort_values(["order_ts"]).drop_duplicates(
        subset=["order_id"], keep="last"
    )

    orders["quantity"] = pd.to_numeric(orders["quantity"], errors="raise")
    orders["discount_pct"] = pd.to_numeric(orders["discount_pct"], errors="raise")
    orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
    orders["delivery_minutes"] = pd.to_numeric(
        orders["delivery_minutes"], errors="coerce"
    )
    orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = pd.NA

    product_details = products[["product_id", "price_inr", "category"]]
    orders = orders.merge(product_details, on="product_id", how="left", validate="many_to_one")
    missing_amount = orders["order_amount"].isna()
    orders.loc[missing_amount, "order_amount"] = (
        orders.loc[missing_amount, "price_inr"]
        * orders.loc[missing_amount, "quantity"]
        * (1 - orders.loc[missing_amount, "discount_pct"] / 100)
    ).round(2)

    completed = orders["status"].eq("completed")
    category_delivery_median = orders.loc[completed].groupby("category")[
        "delivery_minutes"
    ].transform("median")
    orders.loc[completed, "delivery_minutes"] = orders.loc[
        completed, "delivery_minutes"
    ].fillna(category_delivery_median)
    orders.loc[~completed, "delivery_minutes"] = pd.NA

    orders = orders.drop(columns=["price_inr", "category"])
    orders["quantity"] = orders["quantity"].astype("int64")
    orders["discount_pct"] = orders["discount_pct"].astype("int64")
    orders["order_amount"] = orders["order_amount"].round(2)
    orders = orders.sort_values("order_id").reset_index(drop=True)
    return orders


def main():
    CLEAN.mkdir(parents=True, exist_ok=True)
    customers = clean_customers()
    products = clean_products()
    orders = clean_orders(products)
    for name, frame in (
        ("customers.csv", customers),
        ("products.csv", products),
        ("orders.csv", orders),
    ):
        frame.to_csv(CLEAN / name, index=False, date_format="%Y-%m-%d %H:%M:%S")
    print(
        "cleaned: customers=%d products=%d orders=%d completed=%d cancelled=%d"
        % (
            len(customers),
            len(products),
            len(orders),
            int(orders["status"].eq("completed").sum()),
            int(orders["status"].eq("cancelled").sum()),
        )
    )


if __name__ == "__main__":
    main()
