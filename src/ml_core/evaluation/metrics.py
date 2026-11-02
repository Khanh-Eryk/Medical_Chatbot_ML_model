"""
Module metrics.py — Đánh giá mô hình phân loại đa lớp.

Cung cấp:
- Top-K Accuracy (K=1, 3, 5)
- Macro-F1, Weighted-F1
- Thời gian huấn luyện và dự đoán
- So sánh Train vs Validation để phát hiện Overfitting

Lưu ý:
- Tất cả metrics được tính trên tập Validation.
- Tập Test được giữ cho đánh giá cuối cùng (Phase 7).
- Score dự đoán KHÔNG được mặc định là xác suất đã calibrate.
"""

import time
import logging
from typing import Dict, Any, Optional, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

logger = logging.getLogger(__name__)


def compute_top_k_accuracy(
    y_true: np.ndarray,
    proba_matrix: np.ndarray,
    classes: np.ndarray,
    k: int = 5
) -> float:
    """
    Tính Top-K Accuracy từ probability matrix.
    
    Args:
        y_true: Nhãn thực (string hoặc int).
        proba_matrix: Ma trận xác suất (n_samples, n_classes).
        classes: Danh sách nhãn tương ứng với cột của proba_matrix.
        k: Số lượng dự đoán hàng đầu.
    
    Returns:
        Tỷ lệ mẫu có nhãn đúng nằm trong Top-K.
    """
    n_samples = len(y_true)
    if n_samples == 0:
        return 0.0

    # Lấy chỉ mục của Top-K classes (xử lý batch để tiết kiệm bộ nhớ)
    actual_k = min(k, proba_matrix.shape[1])
    top_k_indices = np.argpartition(proba_matrix, -actual_k, axis=1)[:, -actual_k:]
    top_k_classes = classes[top_k_indices]

    correct = 0
    for i in range(n_samples):
        if y_true[i] in top_k_classes[i]:
            correct += 1

    return correct / n_samples


def compute_top_k_from_decision_function(
    y_true: np.ndarray,
    decision_scores: np.ndarray,
    classes: np.ndarray,
    k: int = 5
) -> float:
    """
    Tính Top-K Accuracy từ decision function scores (cho LinearSVC).
    Tương tự compute_top_k_accuracy nhưng dùng raw scores thay vì probabilities.
    """
    return compute_top_k_accuracy(y_true, decision_scores, classes, k)


def evaluate_model_on_split(
    model,
    X: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
    top_k_values: List[int] = [1, 3, 5],
    use_decision_function: bool = False
) -> Dict[str, Any]:
    """
    Đánh giá model trên một tập dữ liệu (val hoặc train).
    
    Args:
        model: Model đã fit.
        X: Features.
        y: Labels.
        classes: Danh sách classes từ model.
        top_k_values: Danh sách K cho Top-K Accuracy.
        use_decision_function: True nếu model dùng decision_function thay vì predict_proba.
    
    Returns:
        Dict chứa các metrics.
    """
    t_start = time.time()
    y_pred = model.predict(X)
    predict_time = time.time() - t_start

    metrics: Dict[str, Any] = {}
    metrics["top_1_accuracy"] = float(accuracy_score(y, y_pred))
    metrics["macro_f1"] = float(f1_score(y, y_pred, average="macro", zero_division=0))
    metrics["weighted_f1"] = float(f1_score(y, y_pred, average="weighted", zero_division=0))
    metrics["predict_time_seconds"] = round(predict_time, 3)

    # Top-K Accuracy (K > 1) cần probability hoặc decision scores
    if len(top_k_values) > 0 and max(top_k_values) > 1:
        try:
            t_score = time.time()
            if use_decision_function and hasattr(model, "decision_function"):
                scores = model.decision_function(X)
            elif hasattr(model, "predict_proba"):
                scores = model.predict_proba(X)
            else:
                scores = None
            score_time = time.time() - t_score

            if scores is not None:
                for k in top_k_values:
                    if k == 1:
                        continue  # Đã tính ở trên
                    metrics[f"top_{k}_accuracy"] = float(
                        compute_top_k_accuracy(y, scores, classes, k)
                    )
                metrics["score_computation_time_seconds"] = round(score_time, 3)
            else:
                for k in top_k_values:
                    if k > 1:
                        metrics[f"top_{k}_accuracy"] = None
        except Exception as e:
            logger.warning(f"Không thể tính Top-K > 1: {e}")
            for k in top_k_values:
                if k > 1:
                    metrics[f"top_{k}_accuracy"] = None

    return metrics


def train_and_evaluate(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_name: str,
    use_decision_function: bool = False
) -> Dict[str, Any]:
    """
    Huấn luyện model và đánh giá trên cả Train và Validation.
    
    Returns:
        Dict chứa fit_time, train_metrics, val_metrics.
    """
    logger.info(f"  Đang huấn luyện {model_name}...")
    t_fit = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t_fit
    logger.info(f"  Fit time: {fit_time:.2f}s")

    classes = model.classes_ if hasattr(model, "classes_") else np.unique(y_train)

    # Đánh giá trên Train (để phát hiện overfitting)
    logger.info(f"  Đang đánh giá trên Train set...")
    train_metrics = evaluate_model_on_split(
        model, X_train, y_train, classes,
        top_k_values=[1],  # Chỉ cần Top-1 cho train (tiết kiệm thời gian)
        use_decision_function=use_decision_function
    )

    # Đánh giá trên Validation
    logger.info(f"  Đang đánh giá trên Validation set...")
    val_metrics = evaluate_model_on_split(
        model, X_val, y_val, classes,
        top_k_values=[1, 3, 5],
        use_decision_function=use_decision_function
    )

    return {
        "model_name": model_name,
        "fit_time_seconds": round(fit_time, 3),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "n_train_samples": len(y_train),
        "n_val_samples": len(y_val),
        "n_classes": len(classes),
    }

