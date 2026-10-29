"""
Script thực thi chuẩn bị dữ liệu cho 3 thí nghiệm (Phase 2):
1. Exp 1: Giữ nguyên toàn bộ dataset (Full).
2. Exp 2: Loại bỏ hoàn toàn trùng lặp (Fully Deduplicated).
3. Exp 3: Giảm số lần lặp ở những tổ hợp bị lặp quá nhiều (Frequency Capped với K=3).
"""

import sys
import os
import time

# Đảm bảo đường dẫn import từ src
sys.path.insert(0, os.path.abspath("src"))

from ml_core.data.dataset_splitter import (
    load_raw_dataset,
    split_common_test_set,
    build_experiment_datasets
)

RAW_DATA_PATH = os.path.join("data", "raw", "Disease and symptoms dataset.csv")
OUTPUT_DIR = os.path.join("data", "processed")

def main():
    print("=" * 60)
    print("BẮT ĐẦU CHUẨN BỊ DỮ LIỆU CHO 3 THÍ NGHIỆM (PHASE 2)")
    print("=" * 60)
    start_time = time.time()

    # 1. Tải raw dataset
    df = load_raw_dataset(RAW_DATA_PATH)

    # 2. Tách Test set và Val set chuẩn dùng chung (chống leakage)
    train_unique_df, val_df, test_df = split_common_test_set(
        df=df,
        target_col="diseases",
        test_size=0.15,
        val_size=0.15,
        random_state=42
    )

    # 3. Tạo 3 tập thí nghiệm (Exp 1, Exp 2, Exp 3 với K=3)
    manifest = build_experiment_datasets(
        raw_df=df,
        train_unique_df=train_unique_df,
        val_df=val_df,
        test_df=test_df,
        target_col="diseases",
        max_cap_k=3,
        output_base_dir=OUTPUT_DIR
    )

    duration = time.time() - start_time
    print("=" * 60)
    print(f"HOÀN THÀNH CHUẨN BỊ 3 THÍ NGHIỆM TRONG {duration:.2f} GIÂY!")
    print("=" * 60)
    for exp_id, meta in manifest.items():
        print(f"[{exp_id}] - {meta['description']}")
        print(f"   Train samples: {meta['train_samples']:,}")
        print(f"   Val samples:   {meta['val_samples']:,}")
        print(f"   Test samples:  {meta['test_samples']:,}")
        print("-" * 60)

if __name__ == "__main__":
    main()

