"""
Script run_model_comparison.py — Phase 4+5: Baseline & So sánh thuật toán.

Mục tiêu:
- Huấn luyện và đánh giá 8 thuật toán trên 3 thí nghiệm xử lý trùng lặp.
- Tất cả đánh giá trên tập Validation (Test set giữ cho Phase 7).
- Ghi kết quả JSON + tạo báo cáo tóm tắt.

Thuật toán:
1. MultinomialNB (Baseline)
2. Logistic Regression
3. Decision Tree
4. Random Forest
5. Extra Trees
6. K-Nearest Neighbors
7. LinearSVC
8. XGBoost

Metrics:
- Top-1, Top-3, Top-5 Accuracy
- Macro-F1, Weighted-F1
- Fit time, Predict time
- Train Top-1 (phát hiện overfitting)

Lưu ý:
- KNN có thể rất chậm trên dataset lớn → giới hạn timeout.
- LinearSVC dùng decision_function thay vì predict_proba.
- Kết quả lưu tại experiments/results/.
"""

import os
import sys
import json
import time
import logging
import signal
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

# Tránh lỗi CP1252 khi chạy script trên Windows với log/trợ giúp tiếng Việt.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Thêm project root vào PYTHONPATH
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

from src.ml_core.evaluation.metrics import train_and_evaluate

# ============================================================================
# CẤU HÌNH
# ============================================================================

EXPERIMENTS = {
    "exp1_full": "data/processed/exp1_full",
    "exp2_dedup": "data/processed/exp2_dedup",
    "exp3_capped": "data/processed/exp3_capped",
}

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")
RANDOM_STATE = 42
TARGET_COL = "diseases"

# Timeout cho mỗi thuật toán (giây) — bảo vệ khỏi treo máy
ALGORITHM_TIMEOUT = 600  # 10 phút

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ============================================================================
# ĐỊNH NGHĨA THUẬT TOÁN
# ============================================================================

def get_algorithm_configs() -> List[Dict[str, Any]]:
    """
    Trả về danh sách cấu hình thuật toán cần thử nghiệm.
    Mỗi config gồm: name, model, use_decision_function, requires_label_encoding.
    """
    return [
        {
            "name": "MultinomialNB",
            "model": MultinomialNB(alpha=1.0),
            "use_decision_function": False,
            "requires_label_encoding": False,
            "notes": "Baseline. Alpha=1.0 (Laplace smoothing mặc định)."
        },
        {
            "name": "LogisticRegression",
            "model": LogisticRegression(
                max_iter=1000,
                solver="saga",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            "use_decision_function": False,
            "requires_label_encoding": False,
            "notes": "Logistic Regression, solver=saga cho dữ liệu lớn, max_iter=1000."
        },
        {
            "name": "DecisionTree",
            "model": DecisionTreeClassifier(
                random_state=RANDOM_STATE,
                max_depth=None
            ),
            "use_decision_function": False,
            "requires_label_encoding": False,
            "notes": "Decision Tree không giới hạn depth (unrestricted)."
        },
        {
            "name": "RandomForest",
            "model": RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                max_depth=None
            ),
            "use_decision_function": False,
            "requires_label_encoding": False,
            "notes": "Random Forest 100 trees, n_jobs=-1."
        },
        {
            "name": "ExtraTrees",
            "model": ExtraTreesClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                max_depth=None
            ),
            "use_decision_function": False,
            "requires_label_encoding": False,
            "notes": "Extra Trees 100 trees, n_jobs=-1."
        },
        {
            "name": "KNN",
            "model": KNeighborsClassifier(
                n_neighbors=5,
                metric="hamming",
                n_jobs=-1
            ),
            "use_decision_function": False,
            "requires_label_encoding": False,
            "notes": "KNN K=5, metric=hamming (phù hợp dữ liệu nhị phân). Có thể chậm."
        },
        {
            "name": "LinearSVC",
            "model": LinearSVC(
                max_iter=2000,
                random_state=RANDOM_STATE,
                dual="auto"
            ),
            "use_decision_function": True,
            "requires_label_encoding": False,
            "notes": "LinearSVC, dùng decision_function cho Top-K."
        },
        {
            "name": "XGBoost",
            "model": XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.3,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                tree_method="hist",
                eval_metric="mlogloss",
                verbosity=0
            ),
            "use_decision_function": False,
            "requires_label_encoding": True,
            "notes": "XGBoost 100 trees, tree_method=hist (nhanh). Cần LabelEncoder."
        },
    ]


# ============================================================================
# TẢI DỮ LIỆU
# ============================================================================

def load_experiment_data(
    exp_dir: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Tải train và val CSV cho một thí nghiệm.
    Returns: X_train, y_train, X_val, y_val
    """
    train_path = os.path.join(PROJECT_ROOT, exp_dir, "train.csv")
    val_path = os.path.join(PROJECT_ROOT, exp_dir, "val.csv")

    logger.info(f"  Đang đọc train: {train_path}")
    train_df = pd.read_csv(train_path)
    logger.info(f"  Đang đọc val: {val_path}")
    val_df = pd.read_csv(val_path)

    feature_cols = [c for c in train_df.columns if c != TARGET_COL]

    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET_COL].values
    X_val = val_df[feature_cols].values
    y_val = val_df[TARGET_COL].values

    return X_train, y_train, X_val, y_val


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_comparison(
    selected_algorithms: List[str] | None = None,
    output_dir: str | None = None,
) -> Dict[str, Any]:
    """Chạy các thuật toán được chọn trên ba split thí nghiệm hiện có."""
    
    run_output_dir = output_dir or OUTPUT_DIR
    os.makedirs(run_output_dir, exist_ok=True)

    algorithms = get_algorithm_configs()
    if selected_algorithms is not None:
        requested = set(selected_algorithms)
        algorithms = [config for config in algorithms if config["name"] in requested]
        unknown_algorithms = requested - {config["name"] for config in get_algorithm_configs()}
        if unknown_algorithms:
            raise ValueError(f"Thuật toán không được hỗ trợ: {', '.join(sorted(unknown_algorithms))}")
        if not algorithms:
            raise ValueError("Cần chọn ít nhất một thuật toán để chạy.")

    all_results: Dict[str, Any] = {
        "metadata": {
            "description": "Phase 4+5: Baseline & Algorithm Comparison",
            "timestamp": datetime.now().isoformat(),
            "random_state": RANDOM_STATE,
            "target_col": TARGET_COL,
            "evaluation_set": "validation (test set reserved for Phase 7)",
            "experiments": list(EXPERIMENTS.keys()),
            "algorithms": [config["name"] for config in algorithms],
            "output_dir": os.path.relpath(run_output_dir, PROJECT_ROOT),
        },
        "results": {}
    }

    for exp_id, exp_dir in EXPERIMENTS.items():
        logger.info(f"\n{'='*70}")
        logger.info(f"EXPERIMENT: {exp_id}")
        logger.info(f"{'='*70}")

        X_train, y_train, X_val, y_val = load_experiment_data(exp_dir)
        logger.info(f"  Train: {X_train.shape}, Val: {X_val.shape}")
        logger.info(f"  Unique classes in train: {len(np.unique(y_train))}")
        logger.info(f"  Unique classes in val: {len(np.unique(y_val))}")

        exp_results: Dict[str, Any] = {
            "experiment_id": exp_id,
            "train_shape": list(X_train.shape),
            "val_shape": list(X_val.shape),
            "feature_count": X_train.shape[1],
            "feature_names": [
                column
                for column in pd.read_csv(
                    os.path.join(PROJECT_ROOT, exp_dir, "train.csv"), nrows=0
                ).columns
                if column != TARGET_COL
            ],
            "algorithms": {}
        }

        for algo_config in algorithms:
            algo_name = algo_config["name"]
            logger.info(f"\n--- {algo_name} on {exp_id} ---")

            try:
                # Clone model để tránh state leak giữa các experiments
                from sklearn.base import clone
                model = clone(algo_config["model"])

                # MultinomialNB hỗ trợ CSR; biểu diễn thưa giữ nguyên 377 feature
                # nhị phân nhưng tránh nhân ma trận dense rất lớn khi đánh giá train.
                X_train_model = X_train
                X_val_model = X_val
                input_representation = "dense"
                if algo_name == "MultinomialNB":
                    X_train_model = sparse.csr_matrix(X_train)
                    X_val_model = sparse.csr_matrix(X_val)
                    input_representation = "sparse_csr"

                # XGBoost cần LabelEncoder cho y
                le = None
                y_train_encoded = y_train
                y_val_encoded = y_val
                if algo_config["requires_label_encoding"]:
                    le = LabelEncoder()
                    y_train_encoded = le.fit_transform(y_train)
                    y_val_encoded = le.transform(y_val)

                result = train_and_evaluate(
                    model=model,
                    X_train=X_train_model,
                    y_train=y_train_encoded,
                    X_val=X_val_model,
                    y_val=y_val_encoded,
                    model_name=algo_name,
                    use_decision_function=algo_config["use_decision_function"]
                )

                result["notes"] = algo_config["notes"]
                result["status"] = "success"
                result["input_representation"] = input_representation

                # Log tóm tắt
                val_m = result["val_metrics"]
                train_m = result["train_metrics"]
                logger.info(
                    f"  Val Top-1: {val_m['top_1_accuracy']:.4f} | "
                    f"Top-3: {val_m.get('top_3_accuracy', 'N/A')} | "
                    f"Top-5: {val_m.get('top_5_accuracy', 'N/A')} | "
                    f"Macro-F1: {val_m['macro_f1']:.4f}"
                )
                logger.info(
                    f"  Train Top-1: {train_m['top_1_accuracy']:.4f} | "
                    f"Fit: {result['fit_time_seconds']:.2f}s"
                )

                # Phát hiện overfitting
                gap = train_m["top_1_accuracy"] - val_m["top_1_accuracy"]
                result["overfitting_gap"] = round(gap, 4)
                if gap > 0.1:
                    logger.warning(f"  ⚠ Overfitting lớn: Train-Val gap = {gap:.4f}")

            except Exception as e:
                logger.error(f"  LỖI: {algo_name} thất bại: {e}")
                result = {
                    "model_name": algo_name,
                    "status": "failed",
                    "error": str(e),
                    "notes": algo_config["notes"]
                }

            exp_results["algorithms"][algo_name] = result

            # Lưu kết quả giữa chừng để không mất dữ liệu nếu crash
            all_results["results"][exp_id] = exp_results
            interim_path = os.path.join(run_output_dir, "model_comparison_results.json")
            with open(interim_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    # Lưu kết quả cuối
    final_path = os.path.join(run_output_dir, "model_comparison_results.json")
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\nĐã lưu kết quả vào: {final_path}")

    return all_results


def print_summary_table(results: Dict[str, Any]) -> None:
    """In bảng tóm tắt kết quả."""
    print("\n" + "=" * 100)
    print("BẢNG TÓM TẮT KẾT QUẢ — Phase 4+5: Baseline & Algorithm Comparison")
    print("=" * 100)
    
    header = f"{'Algorithm':<18} {'Experiment':<14} {'Val Top-1':>10} {'Val Top-3':>10} {'Val Top-5':>10} {'Macro-F1':>10} {'Train Top-1':>12} {'Gap':>8} {'Fit(s)':>8}"
    print(header)
    print("-" * len(header))

    for exp_id, exp_data in results.get("results", {}).items():
        for algo_name, algo_result in exp_data.get("algorithms", {}).items():
            if algo_result.get("status") != "success":
                print(f"{algo_name:<18} {exp_id:<14} {'FAILED':>10}")
                continue
            
            val_m = algo_result["val_metrics"]
            train_m = algo_result["train_metrics"]
            
            top3 = val_m.get("top_3_accuracy")
            top5 = val_m.get("top_5_accuracy")
            top3_str = f"{top3:.4f}" if top3 is not None else "N/A"
            top5_str = f"{top5:.4f}" if top5 is not None else "N/A"
            
            gap = algo_result.get("overfitting_gap", "N/A")
            gap_str = f"{gap:.4f}" if isinstance(gap, (int, float)) else gap
            
            print(
                f"{algo_name:<18} {exp_id:<14} "
                f"{val_m['top_1_accuracy']:>10.4f} "
                f"{top3_str:>10} "
                f"{top5_str:>10} "
                f"{val_m['macro_f1']:>10.4f} "
                f"{train_m['top_1_accuracy']:>12.4f} "
                f"{gap_str:>8} "
                f"{algo_result['fit_time_seconds']:>8.1f}"
            )
        print()  # Dòng trống giữa experiments

    print("=" * 100)


def parse_args() -> argparse.Namespace:
    """Đọc lựa chọn thuật toán và thư mục output cho một phiên thí nghiệm."""
    parser = argparse.ArgumentParser(
        description="Huấn luyện baseline hoặc so sánh thuật toán trên các split Phase 2."
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=[config["name"] for config in get_algorithm_configs()],
        help="Một hoặc nhiều thuật toán cần chạy. Mặc định chạy toàn bộ danh sách.",
    )
    parser.add_argument(
        "--output-dir",
        help="Thư mục lưu kết quả. Nếu bỏ trống, dùng experiments/results/.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("Phase 4+5: Baseline & Algorithm Comparison")
    logger.info("=" * 70)

    args = parse_args()
    results = run_comparison(
        selected_algorithms=args.algorithms,
        output_dir=args.output_dir,
    )
    print_summary_table(results)

    logger.info("\nHoàn thành Phase 4+5!")

