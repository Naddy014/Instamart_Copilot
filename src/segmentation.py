import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from customer_features import build_rfm, load_inputs
from database import ROOT


OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"
FEATURE_COLUMNS = ["recency_days", "frequency", "monetary"]


def run():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    customers, _, completed = load_inputs()
    rfm = build_rfm(customers, completed)
    matrix = np.log1p(rfm[FEATURE_COLUMNS].clip(lower=0))
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_ids = model.fit_predict(scaled)
    rfm["cluster_id"] = cluster_ids
    cluster_order = (
        rfm.groupby("cluster_id")["monetary"].mean().sort_values().index.tolist()
    )
    labels = dict(zip(cluster_order, ["Rare", "Frequent", "Premium"]))
    rfm["segment"] = rfm["cluster_id"].map(labels)
    result = rfm[
        [
            "customer_id",
            "city",
            "recency_days",
            "frequency",
            "monetary",
            "average_order_value",
            "cluster_id",
            "segment",
        ]
    ].sort_values("customer_id")
    result.to_csv(OUTPUTS / "customer_segments.csv", index=False)
    joblib.dump(
        {"model": model, "scaler": scaler, "features": FEATURE_COLUMNS, "labels": labels},
        MODELS / "customer_segments.joblib",
    )
    print("segments:", result["segment"].value_counts().sort_index().to_dict())
    return result


if __name__ == "__main__":
    run()
