import json

import pandas as pd

from database import ROOT


RAW = ROOT / "data" / "raw"
OUTPUTS = ROOT / "outputs"


def describe_frame(name, frame):
    missing = frame.isna().sum()
    return {
        "name": name,
        "shape": list(frame.shape),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "missing": {
            column: {
                "count": int(missing[column]),
                "percent": round(float(missing[column] / len(frame) * 100), 2),
            }
            for column in frame.columns
        },
    }


def main():
    customers = pd.read_csv(RAW / "customers.csv")
    products = pd.read_csv(RAW / "products.csv")
    orders = pd.read_csv(RAW / "orders.csv")
    tickets = pd.read_csv(RAW / "tickets.csv", keep_default_na=False)
    order_amount = pd.to_numeric(orders["order_amount"], errors="coerce")
    delivery = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    ages = pd.to_numeric(customers["age"], errors="coerce")
    report = {
        "tables": [
            describe_frame("customers", customers),
            describe_frame("products", products),
            describe_frame("orders", orders),
            describe_frame("tickets", tickets),
        ],
        "checkpoints": {
            "raw_order_rows": len(orders),
            "exact_duplicate_order_rows": int(orders.duplicated().sum()),
            "pending_order_rows": int(orders["status"].eq("pending").sum()),
            "distinct_order_ids": int(orders["order_id"].nunique()),
            "nonnumeric_order_amounts": int(order_amount.isna().sum()),
            "negative_delivery_minutes": int(delivery.lt(0).sum()),
            "blank_delivery_minutes": int(orders["delivery_minutes"].isna().sum()),
            "blank_customer_ages": int(customers["age"].isna().sum()),
            "customer_ages_outside_10_100": int(
                (ages.notna() & ~ages.between(10, 100)).sum()
            ),
            "distinct_raw_city_spellings": int(customers["city"].nunique()),
            "distinct_nonblank_gender_spellings": int(
                customers.loc[customers["gender"].ne(""), "gender"].nunique()
            ),
            "blank_customer_genders": int(customers["gender"].isna().sum()),
        },
    }
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    with (OUTPUTS / "raw_audit.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(json.dumps(report, indent=2))

    clean_customers = pd.read_csv(ROOT / "data" / "clean" / "customers.csv")
    clean_products = pd.read_csv(ROOT / "data" / "clean" / "products.csv")
    clean_orders = pd.read_csv(ROOT / "data" / "clean" / "orders.csv")
    completed = clean_orders.loc[clean_orders["status"].eq("completed")]
    city_summary = completed.merge(
        clean_customers[["customer_id", "city"]], on="customer_id"
    ).groupby("city")["order_amount"].agg(total="sum", mean="mean", count="count")
    order_summary = completed["order_amount"].describe(
        percentiles=[0.25, 0.5, 0.75]
    )
    q1 = order_summary["25%"]
    q3 = order_summary["75%"]
    iqr = q3 - q1
    print(
        json.dumps(
            {
                "city_order_amount": city_summary.round(2).to_dict(orient="index"),
                "completed_amount_statistics": {
                    "mean": float(completed["order_amount"].mean()),
                    "median": float(completed["order_amount"].median()),
                    "skewness": float(completed["order_amount"].skew()),
                    "q1": float(q1),
                    "q2": float(order_summary["50%"]),
                    "q3": float(q3),
                    "lower_fence": float(q1 - 1.5 * iqr),
                    "upper_fence": float(q3 + 1.5 * iqr),
                },
                "products": len(clean_products),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
