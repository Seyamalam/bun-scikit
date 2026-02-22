import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


def u32(value: int) -> int:
    return value & 0xFFFFFFFF


def mulberry32(seed: int):
    state = u32(seed)

    def rand() -> float:
        nonlocal state
        state = u32(state + 0x6D2B79F5)
        t = u32((state ^ (state >> 15)) * (1 | state))
        t = u32(t ^ u32(t + u32((t ^ (t >> 7)) * (61 | t))))
        return u32(t ^ (t >> 14)) / 4294967296.0

    return rand


def train_test_split_indices(sample_count: int, test_count: int, seed: int):
    indices = list(range(sample_count))
    rand = mulberry32(seed)
    for i in range(sample_count - 1, 0, -1):
        j = int(rand() * (i + 1))
        indices[i], indices[j] = indices[j], indices[i]
    test_indices = set(indices[:test_count])
    return test_indices


def load_dataset(dataset_path: Path):
    with dataset_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or "target" not in reader.fieldnames:
            raise ValueError("Dataset must contain a 'target' column.")
        feature_names = [name for name in reader.fieldnames if name != "target"]
        x_rows = []
        y_values = []
        for row in reader:
            x_rows.append([float(row[name]) for name in feature_names])
            y_values.append(int(float(row["target"])))

    x = np.array(x_rows, dtype=np.float64)
    y = np.array(y_values, dtype=np.int64)
    return feature_names, x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    feature_names, x, y = load_dataset(dataset_path)

    sample_count = len(x)
    if sample_count < 2:
        raise ValueError("Dataset must contain at least 2 rows.")

    test_count = max(1, int(sample_count * args.test_size))
    test_indices = train_test_split_indices(sample_count, test_count, args.random_state)

    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for i in range(sample_count):
        if i in test_indices:
            x_test.append(x[i])
            y_test.append(y[i])
        else:
            x_train.append(x[i])
            y_train.append(y[i])

    x_train = np.array(x_train, dtype=np.float64)
    x_test = np.array(x_test, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.int64)
    y_test = np.array(y_test, dtype=np.int64)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    fit_times = []
    predict_times = []
    predictions_for_metrics = None

    total_loops = args.iterations + args.warmup
    for loop_index in range(total_loops):
        model = LogisticRegression(
            solver="lbfgs",
            max_iter=5000,
            random_state=args.random_state,
        )

        fit_start = time.perf_counter()
        model.fit(x_train_scaled, y_train)
        fit_ms = (time.perf_counter() - fit_start) * 1000

        predict_start = time.perf_counter()
        predictions = model.predict(x_test_scaled)
        predict_ms = (time.perf_counter() - predict_start) * 1000

        if loop_index >= args.warmup:
            fit_times.append(fit_ms)
            predict_times.append(predict_ms)
            predictions_for_metrics = predictions

    if predictions_for_metrics is None:
        raise RuntimeError("No benchmark iterations were recorded.")

    result = {
        "implementation": "python-scikit-learn",
        "model": "StandardScaler + LogisticRegression(lbfgs)",
        "iterations": args.iterations,
        "fitMsMedian": statistics.median(fit_times),
        "predictMsMedian": statistics.median(predict_times),
        "accuracy": accuracy_score(y_test, predictions_for_metrics),
        "f1": f1_score(y_test, predictions_for_metrics),
        "dataset": {
            "path": str(dataset_path).replace("\\", "/"),
            "samples": int(sample_count),
            "features": int(len(feature_names)),
            "trainSize": int(len(x_train)),
            "testSize": int(len(x_test)),
            "randomState": args.random_state,
            "testFraction": args.test_size,
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scikitLearn": sklearn.__version__,
        },
    }

    print(json.dumps(result))


if __name__ == "__main__":
    main()
