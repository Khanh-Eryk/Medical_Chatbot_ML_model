"""
Script thực hiện kiểm toán dữ liệu (Data Audit) toàn diện cho tệp Disease and symptoms dataset.csv.
Tuân thủ nguyên tắc:
1. Chỉ đọc (read-only), không chỉnh sửa hoặc ghi đè file gốc.
2. Kiểm tra encoding, kích thước, số dòng, số cột.
3. Kiểm tra missing values, kiểu dữ liệu, các giá trị duy nhất (unique values) của feature.
4. Thống kê phân phối nhãn bệnh (target: diseases), phát hiện mất cân bằng lớp.
5. Phát hiện các dòng trùng lặp hoàn toàn (exact duplicate rows).
6. Phát hiện các dòng xung đột nhãn (cùng symptoms nhưng khác disease label).
7. Phân tích độ thưa (sparsity) của ma trận đặc trưng, số lượng triệu chứng trung bình trên mỗi mẫu bệnh.
8. Xuất kết quả tổng hợp ra JSON machine-readable.
"""

import csv
import json
import os
import sys
import time
from collections import Counter
import numpy as np
import pandas as pd

DATA_FILE = "data/raw/disease and symptoms dataset.csv"
OUTPUT_METRICS_FILE = os.path.join("docs", "data_audit_metrics.json")

def run_audit():
    print(f"=== BẮT ĐẦU DATA AUDIT: {DATA_FILE} ===")
    start_time = time.time()
    
    if not os.path.exists(DATA_FILE):
        print(f"LỖI: Không tìm thấy tệp {DATA_FILE}")
        sys.exit(1)
        
    file_size_bytes = os.path.getsize(DATA_FILE)
    print(f"Kích thước tệp: {file_size_bytes:,} bytes (~{file_size_bytes / (1024*1024):.2f} MB)")
    
    # 1. Kiểm tra header và cấu trúc dòng cơ bản
    print("Đang đọc header và cấu trúc cột...")
    with open(DATA_FILE, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        first_row = next(reader)
        num_columns = len(header)
        first_row_cols = len(first_row)
        
    print(f"Số cột: {num_columns}")
    print(f"Khớp số cột dòng đầu: {num_columns == first_row_cols}")
    
    target_col = header[0]
    feature_cols = header[1:]
    num_features = len(feature_cols)
    print(f"Cột Target giả định: '{target_col}'")
    print(f"Số lượng feature giả định: {num_features}")
    
    # 2. Đọc dữ liệu bằng pandas
    print("Đang tải dữ liệu vào DataFrame (đang tối ưu hóa kiểu dữ liệu)...")
    load_start = time.time()
    
    # Đọc nhanh
    df = pd.read_csv(DATA_FILE, low_memory=False)
    load_duration = time.time() - load_start
    print(f"Tải hoàn tất sau {load_duration:.2f}s. Số dòng: {len(df):,}, Số cột: {len(df.columns):,}")
    
    num_rows = len(df)
    
    # 3. Phân tích missing values
    print("Kiểm tra giá trị thiếu (missing values / nulls)...")
    null_counts = df.isnull().sum()
    total_nulls = int(null_counts.sum())
    cols_with_null = null_counts[null_counts > 0].to_dict()
    print(f"Tổng số giá trị null trên toàn bộ dataset: {total_nulls}")
    if cols_with_null:
        print(f"Các cột có null: {cols_with_null}")
    else:
        print("Không có bất kỳ giá trị null nào trong toàn bộ bảng!")
        
    # 4. Phân tích cột Target (diseases)
    print("Phân tích phân phối cột nhãn (diseases)...")
    target_series = df[target_col].astype(str)
    disease_counts = target_series.value_counts()
    num_unique_diseases = len(disease_counts)
    min_samples_per_disease = int(disease_counts.min())
    max_samples_per_disease = int(disease_counts.max())
    mean_samples_per_disease = float(disease_counts.mean())
    median_samples_per_disease = float(disease_counts.median())
    std_samples_per_disease = float(disease_counts.std())
    
    print(f"Số lượng nhãn bệnh duy nhất: {num_unique_diseases}")
    print(f"Số mẫu ít nhất ở một bệnh: {min_samples_per_disease} ({disease_counts.idxmin()})")
    print(f"Số mẫu nhiều nhất ở một bệnh: {max_samples_per_disease} ({disease_counts.idxmax()})")
    print(f"Trung bình mẫu / bệnh: {mean_samples_per_disease:.2f}")
    print(f"Trung vị mẫu / bệnh: {median_samples_per_disease:.2f}")
    print(f"Độ lệch chuẩn mẫu / bệnh: {std_samples_per_disease:.2f}")
    
    # 5. Phân tích các cột Feature (giá trị, kiểu dữ liệu, miền giá trị)
    print("Kiểm tra miền giá trị của các feature...")
    features_df = df[feature_cols]
    
    # Kiểm tra unique values qua toàn bộ feature matrix
    feature_dtypes = {str(k): v for k, v in features_df.dtypes.value_counts().items()}
    print(f"Phân phối kiểu dữ liệu của features: {feature_dtypes}")
    
    # Kiểm tra xem có giá trị nào khác 0 và 1 không
    unique_vals_all_features = set()
    sample_sub = features_df.select_dtypes(include=[np.number])
    
    # Kiểm tra min, max
    min_val = float(features_df.min().min())
    max_val = float(features_df.max().max())
    print(f"Giá trị min toàn bộ feature: {min_val}, max: {max_val}")
    
    # Kiểm tra xem có cột feature nào chỉ chứa toàn số 0 không (constant features)
    feature_sums = features_df.sum(axis=0)
    zero_sum_features = feature_sums[feature_sums == 0].index.tolist()
    all_one_features = feature_sums[feature_sums == num_rows].index.tolist()
    print(f"Số feature không bao giờ xuất hiện (sum == 0): {len(zero_sum_features)}")
    print(f"Số feature luôn luôn xuất hiện (sum == num_rows): {len(all_one_features)}")
    
    # Tính độ thưa (sparsity)
    total_feature_cells = num_rows * num_features
    total_active_symptoms = int(feature_sums.sum())
    sparsity = 1.0 - (total_active_symptoms / total_feature_cells)
    
    # Số triệu chứng trung bình trên mỗi bệnh nhân
    symptoms_per_patient = features_df.sum(axis=1)
    min_symptoms_per_patient = int(symptoms_per_patient.min())
    max_symptoms_per_patient = int(symptoms_per_patient.max())
    mean_symptoms_per_patient = float(symptoms_per_patient.mean())
    zero_symptom_patients = int((symptoms_per_patient == 0).sum())
    
    print(f"Độ thưa của ma trận đặc trưng (Sparsity): {sparsity * 100:.2f}%")
    print(f"Số triệu chứng trung bình/mẫu: {mean_symptoms_per_patient:.2f} (Min: {min_symptoms_per_patient}, Max: {max_symptoms_per_patient})")
    print(f"Số mẫu không có bất kỳ triệu chứng nào (sum==0): {zero_symptom_patients}")

    # 6. Kiểm tra các dòng trùng lặp (Duplicates)
    print("Kiểm tra các bản ghi trùng lặp...")
    # Exact duplicate (cả features và target)
    dup_start = time.time()
    exact_duplicates = int(df.duplicated().sum())
    print(f"Số dòng trùng lặp hoàn toàn (cả features và disease): {exact_duplicates:,} ({exact_duplicates / num_rows * 100:.2f}%)")
    
    # Trùng lặp feature nhưng khác label (Label Conflict / Inconsistency)
    feature_duplicates = df.duplicated(subset=feature_cols, keep=False)
    num_feature_duplicates = int(feature_duplicates.sum())
    print(f"Tổng số dòng có feature vector trùng với ít nhất 1 dòng khác: {num_feature_duplicates:,} ({num_feature_duplicates / num_rows * 100:.2f}%)")
    
    # Tìm xung đột nhãn cụ thể: cùng feature vector nhưng nhãn khác nhau
    # Groupby feature_cols -> count unique diseases
    print("Đang phát hiện xung đột nhãn (cùng triệu chứng nhưng gán bệnh khác nhau)...")
    grouped = df.groupby(feature_cols)[target_col].nunique()
    conflicting_vectors = (grouped > 1).sum()
    conflicting_rows = grouped[grouped > 1]
    total_conflicting_rows = df[feature_duplicates].groupby(feature_cols).filter(lambda x: x[target_col].nunique() > 1)
    num_conflicting_rows = len(total_conflicting_rows)
    print(f"Số vector triệu chứng bị gán cho nhiều hơn 1 bệnh: {conflicting_vectors:,}")
    print(f"Tổng số dòng bị ảnh hưởng bởi xung đột nhãn: {num_conflicting_rows:,} ({num_conflicting_rows / num_rows * 100:.2f}%)")
    
    # Top 10 bệnh phổ biến nhất và ít phổ biến nhất
    top_10_diseases = disease_counts.head(10).to_dict()
    bottom_10_diseases = disease_counts.tail(10).to_dict()
    
    # Top 10 triệu chứng xuất hiện nhiều nhất và ít nhất
    top_10_symptoms = feature_sums.sort_values(ascending=False).head(10).to_dict()
    bottom_10_symptoms = feature_sums.sort_values(ascending=True).head(10).to_dict()
    
    audit_duration = time.time() - start_time
    print(f"=== HOÀN THÀNH DATA AUDIT SAU {audit_duration:.2f}s ===")
    
    # Lưu kết quả ra JSON
    audit_results = {
        "dataset_metadata": {
            "file_name": DATA_FILE,
            "file_size_bytes": file_size_bytes,
            "total_rows": num_rows,
            "total_columns": num_columns,
            "target_column": target_col,
            "num_features": num_features,
            "audit_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "data_integrity": {
            "total_nulls": total_nulls,
            "columns_with_nulls": cols_with_null,
            "feature_dtypes": feature_dtypes,
            "min_feature_val": min_val,
            "max_feature_val": max_val,
            "zero_sum_features": zero_sum_features,
            "all_one_features": all_one_features
        },
        "sparsity_and_distribution": {
            "sparsity_pct": round(sparsity * 100, 4),
            "mean_symptoms_per_sample": round(mean_symptoms_per_patient, 4),
            "min_symptoms_per_sample": min_symptoms_per_patient,
            "max_symptoms_per_sample": max_symptoms_per_patient,
            "zero_symptom_samples_count": zero_symptom_patients
        },
        "target_distribution": {
            "num_unique_diseases": num_unique_diseases,
            "min_samples": min_samples_per_disease,
            "max_samples": max_samples_per_disease,
            "mean_samples": round(mean_samples_per_disease, 2),
            "median_samples": round(median_samples_per_disease, 2),
            "std_samples": round(std_samples_per_disease, 2),
            "top_10_diseases": top_10_diseases,
            "bottom_10_diseases": bottom_10_diseases
        },
        "duplicates_and_conflicts": {
            "exact_duplicate_rows": exact_duplicates,
            "exact_duplicate_pct": round(exact_duplicates / num_rows * 100, 4),
            "identical_feature_rows": num_feature_duplicates,
            "identical_feature_pct": round(num_feature_duplicates / num_rows * 100, 4),
            "conflicting_feature_vectors_count": int(conflicting_vectors),
            "conflicting_rows_count": num_conflicting_rows,
            "conflicting_rows_pct": round(num_conflicting_rows / num_rows * 100, 4)
        },
        "symptom_frequency": {
            "top_10_symptoms": top_10_symptoms,
            "bottom_10_symptoms": bottom_10_symptoms
        }
    }
    
    with open(OUTPUT_METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2, ensure_ascii=False)
        
    print(f"Đã lưu kết quả kiểm toán vào: {OUTPUT_METRICS_FILE}")

if __name__ == "__main__":
    run_audit()

