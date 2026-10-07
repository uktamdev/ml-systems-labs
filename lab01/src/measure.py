"""Lab 1: baseline models, accuracy, and system cost measurements."""
import csv
import os
import random
import statistics
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import psutil
from memory_profiler import memory_usage
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=ConvergenceWarning)

RESULTS = Path(__file__).resolve().parents[1] / "results"

N_TRAIN_RUNS = 5
N_INFER_RUNS = 100

MODELS = {
    "LogisticRegression": lambda: LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
}


def peak_mb(func):
    """Peak process memory (MiB) while func runs."""
    peak = memory_usage((func, (), {}), interval=0.01, max_usage=True)
    return float(np.max(peak))


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    random.seed(42)
    np.random.seed(42)
    RESULTS.mkdir(exist_ok=True)

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    sample = X_test[:1]

    accuracy_rows = []
    cost_rows = []

    for name, make in MODELS.items():
        print(f"== {name} ==")

        # Accuracy (final model, trained once)
        model = make().fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        accuracy_rows.append({"model": name, "test_accuracy": f"{acc:.4f}"})

        # Training time: 1 warm-up, then median of 5 runs
        make().fit(X_train, y_train)
        times = []
        for _ in range(N_TRAIN_RUNS):
            t0 = time.perf_counter()
            make().fit(X_train, y_train)
            times.append(time.perf_counter() - t0)
        train_median_s = statistics.median(times)

        # Single-sample inference: 1 warm-up, then 100 runs, median
        model.predict(sample)
        lat = []
        for _ in range(N_INFER_RUNS):
            t0 = time.perf_counter()
            model.predict(sample)
            lat.append((time.perf_counter() - t0) * 1000)
        infer_median_ms = statistics.median(lat)

        # Model size
        path = RESULTS / f"model_{name}.joblib"
        joblib.dump(model, path)
        size_bytes = os.path.getsize(path)
        size_kb = size_bytes / 1024

        # Peak memory during training and inference
        rss_before = psutil.Process().memory_info().rss / 1024**2
        train_peak = peak_mb(lambda: make().fit(X_train, y_train))

        def infer_loop():
            for _ in range(N_INFER_RUNS):
                model.predict(sample)

        infer_peak = peak_mb(infer_loop)

        cost_rows.append({
            "model": name,
            "train_time_median_s": f"{train_median_s:.4f}",
            "infer_latency_median_ms": f"{infer_median_ms:.4f}",
            "model_size_bytes": size_bytes,
            "model_size_kb": f"{size_kb:.2f}",
            "rss_before_mb": f"{rss_before:.1f}",
            "train_peak_mem_mb": f"{train_peak:.1f}",
            "infer_peak_mem_mb": f"{infer_peak:.1f}",
        })
        print(cost_rows[-1])

    write_csv(RESULTS / "baseline_accuracy.csv", accuracy_rows)
    write_csv(RESULTS / "system_cost.csv", cost_rows)
    print("Saved baseline_accuracy.csv and system_cost.csv")


if __name__ == "__main__":
    main()