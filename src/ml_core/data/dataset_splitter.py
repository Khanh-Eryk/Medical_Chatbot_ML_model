"""
Module dataset_splitter.py
Nhiệm vụ:
- Quản lý quy trình phân chia dữ liệu có kiểm soát rò rỉ (Leakage-resistant split).
- Hỗ trợ 3 thí nghiệm xử lý trùng lặp:
  + Exp 1: Giữ nguyên toàn bộ dataset (Full).
  + Exp 2: Loại bỏ hoàn toàn trùng lặp (Full Deduplication).
  + Exp 3: Giới hạn tần suất lặp tối đa K=3 (Frequency Capping K=3).
- Bảo toàn tập Test độc lập dùng chung (Common Test Set) cho cả 3 thí nghiệm để so sánh công bằng.
- Xuất manifest và metadata chi tiết để tái lập.
"""

import os
import json
import logging
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_raw_dataset(data_path: str) -> pd.DataFrame:
    """Tải dữ liệu thô từ đường dẫn đã cho."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại {data_path}")
    logger.info(f"Đang đọc dữ liệu từ: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    logger.info(f"Tải thành công: {len(df):,} dòng, {len(df.columns)} cột")
    return df


def get_feature_and_target_columns(df: pd.DataFrame, target_col: str = "diseases") -> Tuple[List[str], str]:
    """Phân tách cột features và target."""
    if target_col not in df.columns:
        raise ValueError(f"Không tìm thấy cột target '{target_col}' trong dataset.")
    feature_cols = [c for c in df.columns if c != target_col]
    return feature_cols, target_col


def split_common_test_set(
    df: pd.DataFrame,
    target_col: str = "diseases",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Tách tập Test và Validation từ các tổ hợp duy nhất (Unique Combinations) để:
    1. Tránh rò rỉ dữ liệu hoàn toàn giữa Train, Validation và Test.
    2. Cung cấp một tập Test và Validation chuẩn dùng chung cho cả 3 thí nghiệm.
    """
    feature_cols, _ = get_feature_and_target_columns(df, target_col)
    logger.info("Đang trích xuất các tổ hợp đặc trưng và nhãn duy nhất (Unique instances)...")
    unique_df = df.drop_duplicates(subset=[target_col] + feature_cols).copy()
    logger.info(f"Tổng số tổ hợp duy nhất: {len(unique_df):,}")

    class_counts = unique_df[target_col].value_counts()
    eligible_classes = class_counts[class_counts >= 3].index
    rare_classes = class_counts[class_counts < 3].index

    logger.info(f"Số lớp đủ điều kiện chia tầng (>= 3 mẫu): {len(eligible_classes)}")
    logger.info(f"Số lớp cực hiếm (< 3 mẫu, được giữ trong Train): {len(rare_classes)}")

    df_eligible = unique_df[unique_df[target_col].isin(eligible_classes)].copy()
    df_rare = unique_df[unique_df[target_col].isin(rare_classes)].copy()

    # Bước 1: Tách Test set
    train_val_df, test_df = train_test_split(
        df_eligible,
        test_size=test_size,
        stratify=df_eligible[target_col],
        random_state=random_state
    )

    # Bước 2: Tách Validation set từ train_val
    adjusted_val_size = val_size / (1.0 - test_size)
    train_unique_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_size,
        stratify=train_val_df[target_col],
        random_state=random_state
    )

    # Đưa các lớp cực hiếm vào train_unique
    train_unique_df = pd.concat([train_unique_df, df_rare], ignore_index=True)

    logger.info(f"Phân chia Unique pool: Train={len(train_unique_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")
    return train_unique_df, val_df, test_df


def build_experiment_datasets(
    raw_df: pd.DataFrame,
    train_unique_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "diseases",
    max_cap_k: int = 3,
    output_base_dir: str = "data/processed"
) -> Dict[str, Any]:
    """
    Xây dựng dữ liệu cho 3 thí nghiệm từ tập Train:
    - Exp 1: Giữ nguyên toàn bộ các bản ghi thuộc Train pool (Full).
    - Exp 2: Giữ đúng 1 bản ghi cho mỗi tổ hợp (Full Deduplication).
    - Exp 3: Giới hạn tối đa K bản ghi cho mỗi tổ hợp (Frequency Capping K=3).
    """
    feature_cols, _ = get_feature_and_target_columns(raw_df, target_col)
    subset_cols = [target_col] + feature_cols

    logger.info("Đang ánh xạ các bản ghi raw_df vào phân vùng...")
    
    val_keys = set(tuple(x) for x in val_df[subset_cols].values)
    test_keys = set(tuple(x) for x in test_df[subset_cols].values)
    train_unique_keys = set(tuple(x) for x in train_unique_df[subset_cols].values)

    assert len(val_keys & test_keys) == 0, "LỖI LEAKAGE: Val và Test có phần tử trùng lặp!"
    assert len(train_unique_keys & test_keys) == 0, "LỖI LEAKAGE: Train và Test có phần tử trùng lặp!"
    assert len(train_unique_keys & val_keys) == 0, "LỖI LEAKAGE: Train và Val có phần tử trùng lặp!"

    raw_tuples = [tuple(x) for x in raw_df[subset_cols].values]
    train_mask = [t in train_unique_keys for t in raw_tuples]
    raw_train_df = raw_df[train_mask].copy()
    logger.info(f"Tổng số bản ghi trong Raw Train Pool: {len(raw_train_df):,}")

    # Exp 1: Full Train
    exp1_train = raw_train_df.copy()

    # Exp 2: Dedup Train
    exp2_train = train_unique_df.copy()

    # Exp 3: Capped Train (Tối ưu hóa vector hóa cumcount để chạy trong 0.1s thay vì groupby.apply)
    logger.info(f"Đang thực hiện Capping K={max_cap_k} trên Train Pool (vectorized cumcount)...")
    cum_counts = raw_train_df.groupby(subset_cols).cumcount()
    exp3_train = raw_train_df[cum_counts < max_cap_k].copy()
    logger.info(f"Số mẫu sau khi Capping K={max_cap_k}: {len(exp3_train):,} dòng")

    experiments = {
        "exp1_full": {
            "name": "Exp 1: Full Dataset (No Deduplication)",
            "train": exp1_train,
            "val": val_df,
            "test": test_df,
            "capping_k": None,
        },
        "exp2_dedup": {
            "name": "Exp 2: Fully Deduplicated Dataset",
            "train": exp2_train,
            "val": val_df,
            "test": test_df,
            "capping_k": 1,
        },
        "exp3_capped": {
            "name": f"Exp 3: Frequency Capped Dataset (K={max_cap_k})",
            "train": exp3_train,
            "val": val_df,
            "test": test_df,
            "capping_k": max_cap_k,
        }
    }

    manifest = {}

    for exp_id, exp_data in experiments.items():
        exp_dir = os.path.join(output_base_dir, exp_id)
        os.makedirs(exp_dir, exist_ok=True)
        
        train_path = os.path.join(exp_dir, "train.csv")
        val_path = os.path.join(exp_dir, "val.csv")
        test_path = os.path.join(exp_dir, "test.csv")
        meta_path = os.path.join(exp_dir, "metadata.json")

        logger.info(f"Đang ghi file CSV cho {exp_id}...")
        exp_data["train"].to_csv(train_path, index=False)
        exp_data["val"].to_csv(val_path, index=False)
        exp_data["test"].to_csv(test_path, index=False)

        meta = {
            "experiment_id": exp_id,
            "description": exp_data["name"],
            "capping_k": exp_data["capping_k"],
            "train_samples": len(exp_data["train"]),
            "val_samples": len(exp_data["val"]),
            "test_samples": len(exp_data["test"]),
            "total_samples": len(exp_data["train"]) + len(exp_data["val"]) + len(exp_data["test"]),
            "num_features": len(feature_cols),
            "num_classes_train": int(exp_data["train"][target_col].nunique()),
            "num_classes_val": int(exp_data["val"][target_col].nunique()),
            "num_classes_test": int(exp_data["test"][target_col].nunique()),
            "files": {
                "train": train_path,
                "val": val_path,
                "test": test_path
            }
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        manifest[exp_id] = meta
        logger.info(f"Hoàn thành {exp_id}: Train={len(exp_data['train']):,}, Val={len(exp_data['val']):,}, Test={len(exp_data['test']):,}")

    summary_path = os.path.join(output_base_dir, "experiments_manifest.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info(f"Đã lưu tóm tắt toàn bộ 3 thí nghiệm vào {summary_path}")

    return manifest
