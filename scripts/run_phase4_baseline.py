"""Phase 4 baseline. Huấn luyện chỉ từ CSV train/validation thực tế của Phase 2."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml_core.preprocessing.feature_schema import FeatureSchema

EXPERIMENTS = ("exp1_full", "exp2_dedup", "exp3_capped")
TARGET_COLUMN = "diseases"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "results" / "phase4_baseline_377_raw"
REPORT_PATH = PROJECT_ROOT / "docs" / "phase4_baseline_report.md"


def top_k_accuracy(y_true: np.ndarray, scores: np.ndarray, classes: np.ndarray, k: int) -> float:
    indices = np.argpartition(scores, -k, axis=1)[:, -k:]
    return float(np.mean([label in classes[row] for label, row in zip(y_true, indices)]))


def top_confusions(matrix: np.ndarray, labels: np.ndarray, limit: int = 10) -> list[dict]:
    candidates = []
    for actual_index, predicted_index in zip(*np.where(~np.eye(matrix.shape[0], dtype=bool))):
        count = int(matrix[actual_index, predicted_index])
        if count:
            candidates.append({"actual": str(labels[actual_index]), "predicted": str(labels[predicted_index]), "count": count})
    return sorted(candidates, key=lambda item: item["count"], reverse=True)[:limit]


def run_experiment(experiment_id: str) -> dict:
    experiment_dir = PROJECT_ROOT / "data" / "processed" / experiment_id
    train = pd.read_csv(experiment_dir / "train.csv")
    validation = pd.read_csv(experiment_dir / "val.csv")

    # Schema và matrix đều xuất phát từ dataframe CSV thực tế của experiment này.
    schema = FeatureSchema.from_training_dataframe(train, TARGET_COLUMN)
    schema.validate_dataframe(validation, require_target=True)
    x_train = schema.transform(train)
    x_validation = schema.transform(validation)
    y_train = train[TARGET_COLUMN].to_numpy()
    y_validation = validation[TARGET_COLUMN].to_numpy()

    model = MultinomialNB(alpha=1.0)
    started = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - started
    validation_predictions = model.predict(x_validation)
    validation_scores = model.predict_proba(x_validation)
    labels = model.classes_
    matrix = confusion_matrix(y_validation, validation_predictions, labels=labels)

    return {
        "experiment_id": experiment_id,
        "data_source": {"train_csv": str(experiment_dir / "train.csv"), "validation_csv": str(experiment_dir / "val.csv")},
        "feature_source": "columns from this experiment's train.csv",
        "feature_count": len(schema.feature_names),
        "feature_names": list(schema.feature_names),
        "model": {"name": "MultinomialNB", "alpha": 1.0},
        "fit_seconds": round(fit_seconds, 3),
        "train_samples": len(train),
        "validation_samples": len(validation),
        "validation_metrics": {
            "top_1_accuracy": float(accuracy_score(y_validation, validation_predictions)),
            "top_3_accuracy": top_k_accuracy(y_validation, validation_scores, labels, 3),
            "top_5_accuracy": top_k_accuracy(y_validation, validation_scores, labels, 5),
            "macro_f1": float(f1_score(y_validation, validation_predictions, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_validation, validation_predictions, average="weighted", zero_division=0)),
        },
        "per_class_metrics": classification_report(y_validation, validation_predictions, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": {"labels": labels.tolist(), "values": matrix.tolist()},
        "top_confusions": top_confusions(matrix, labels),
    }


def write_report(results: dict) -> None:
    lines = [
        "# BÁO CÁO GIAI ĐOẠN 4 — BASELINE 377 FEATURE RAW", "",
        "## Liên kết dữ liệu", "",
        "Mỗi phiên chỉ đọc `train.csv` và `val.csv` thực tế do Phase 2 tạo ra. Không dùng manifest, JSON result, notebook output hoặc metadata làm input huấn luyện. Phase 3 chỉ xác thực cột và chuyển cùng giá trị nhị phân sang CSR sparse.", "",
        "## Cấu hình", "", "- Model: `MultinomialNB(alpha=1.0)`.", "- Feature: đầy đủ 377 cột raw, không feature selection, không giảm chiều.", "- Tập đánh giá: validation; test set không được đọc.", "",
        "## Kết quả validation", "", "| Experiment | Train | Validation | Top-1 | Macro-F1 | Weighted-F1 | Top-3 | Top-5 | Fit (s) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results["experiments"]:
        metric = item["validation_metrics"]
        lines.append(f"| {item['experiment_id']} | {item['train_samples']:,} | {item['validation_samples']:,} | {metric['top_1_accuracy']:.2%} | {metric['macro_f1']:.2%} | {metric['weighted_f1']:.2%} | {metric['top_3_accuracy']:.2%} | {metric['top_5_accuracy']:.2%} | {item['fit_seconds']:.3f} |")
    lines += ["", "## Phân tích lỗi", ""]
    for item in results["experiments"]:
        lines.append(f"### {item['experiment_id']}")
        lines.append("")
        lines.append("Các nhầm lẫn có tần suất cao nhất trên validation:")
        lines.append("")
        lines.append("| Nhãn thực | Dự đoán | Số mẫu |\n|---|---|---:|")
        lines += [f"| {row['actual']} | {row['predicted']} | {row['count']} |" for row in item["top_confusions"]]
        lines.append("")
    lines += ["## Giới hạn", "", "Baseline không phải đánh giá lâm sàng, không chọn model cuối và không sử dụng test set. Các bước feature selection, tuning và Phase 5 vẫn chưa được thực hiện.", ""]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Từ chối ghi đè artifact hiện có: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)
    results = {"phase": 4, "evaluation_set": "validation only", "experiments": [run_experiment(exp) for exp in EXPERIMENTS]}
    (OUTPUT_DIR / "baseline_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(results)
    print(f"Saved results to {OUTPUT_DIR}")
    print(f"Saved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
