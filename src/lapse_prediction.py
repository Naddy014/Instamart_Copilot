import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from customer_features import ROOT, load_inputs
from database import ROOT as PROJECT_ROOT


OUTPUTS = PROJECT_ROOT / "outputs"
MODEL_PATH = OUTPUTS / "lapse_model.joblib"
FEATURES = ["recency_days", "frequency", "monetary", "average_order_value", "age", "city", "gender"]
TRAINING_CUTOFF = pd.Timestamp("2025-08-31 23:59:59")
LABEL_END = pd.Timestamp("2025-09-30 23:59:59")


def make_dataset(customers, completed):
    historical = completed.loc[completed["order_ts"] <= TRAINING_CUTOFF]
    rfm = (
        historical.groupby("customer_id")
        .agg(
            last_order=("order_ts", "max"),
            frequency=("order_id", "count"),
            monetary=("order_amount", "sum"),
            average_order_value=("order_amount", "mean"),
        )
        .reset_index()
    )
    dataset = customers[["customer_id", "city", "gender", "age"]].merge(
        rfm, on="customer_id", how="left"
    )
    dataset["frequency"] = dataset["frequency"].fillna(0)
    dataset["monetary"] = dataset["monetary"].fillna(0)
    dataset["average_order_value"] = dataset["average_order_value"].fillna(0)
    dataset["recency_days"] = (
        TRAINING_CUTOFF.normalize() - pd.to_datetime(dataset["last_order"]).dt.normalize()
    ).dt.days
    dataset["recency_days"] = dataset["recency_days"].fillna(183)
    future = completed.loc[
        (completed["order_ts"] > TRAINING_CUTOFF)
        & (completed["order_ts"] <= LABEL_END)
    ].groupby("customer_id")["order_id"].count()
    dataset["lapsed"] = (~dataset["customer_id"].isin(future.index)).astype(int)
    return dataset


def run():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    customers, _, completed = load_inputs()
    dataset = make_dataset(customers, completed)
    x = dataset[FEATURES].copy()
    x["monetary"] = np.log1p(x["monetary"])
    x["average_order_value"] = np.log1p(x["average_order_value"])
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        dataset["lapsed"],
        test_size=0.25,
        random_state=42,
        stratify=dataset["lapsed"],
    )
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    categorical = ["city", "gender"]
    encoder.fit(x_train[categorical])

    def transform(frame):
        numeric = frame[["recency_days", "frequency", "monetary", "average_order_value", "age"]].to_numpy(dtype=float)
        return np.hstack((numeric, encoder.transform(frame[categorical])))

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    model.fit(transform(x_train), y_train)
    probabilities = model.predict_proba(transform(x_test))[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "test_size": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "label": "no completed order during September 2025",
        "training_cutoff": TRAINING_CUTOFF.isoformat(),
    }

    all_x = dataset[FEATURES].copy()
    all_x["monetary"] = np.log1p(all_x["monetary"])
    all_x["average_order_value"] = np.log1p(all_x["average_order_value"])
    risk = model.predict_proba(transform(all_x))[:, 1]
    retention = dataset[["customer_id", "city"]].copy()
    retention["lapse_probability"] = risk
    retention["risk_level"] = pd.cut(
        risk, bins=[-0.001, 0.4, 0.7, 1.001], labels=["Low", "Medium", "High"]
    ).astype(str)
    retention = retention.sort_values(
        ["lapse_probability", "customer_id"], ascending=[False, True]
    )
    retention.to_csv(OUTPUTS / "retention_list.csv", index=False)
    joblib.dump(
        {"model": model, "encoder": encoder, "features": FEATURES, "metrics": metrics},
        MODEL_PATH,
    )
    with (OUTPUTS / "lapse_metrics.json").open("w", encoding="utf-8") as file:
        import json

        json.dump(metrics, file, indent=2)
    print("lapse metrics:", metrics)
    print("retention list rows:", len(retention))
    return retention


if __name__ == "__main__":
    run()
