"""
Unit tests kiểm tra tính toàn vẹn của dữ liệu đầu vào (Data Integrity Tests).
Chạy bằng pytest: pytest tests/test_data_integrity.py
"""

import os
import pytest
import pandas as pd
import numpy as np

# Định vị đường dẫn file dữ liệu
RAW_DATA_PATH = os.path.join("data", "raw", "Disease and symptoms dataset.csv")
ROOT_DATA_PATH = "Disease and symptoms dataset.csv"

def get_data_path():
    if os.path.exists(RAW_DATA_PATH):
        return RAW_DATA_PATH
    elif os.path.exists(ROOT_DATA_PATH):
        return ROOT_DATA_PATH
    else:
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại {RAW_DATA_PATH} hoặc {ROOT_DATA_PATH}")

@pytest.fixture(scope="module")
def data_path():
    return get_data_path()

@pytest.fixture(scope="module")
def dataset_df(data_path):
    """Tải một phần dữ liệu (5,000 dòng) để kiểm tra nhanh tính toàn vẹn."""
    df = pd.read_csv(data_path, nrows=5000, low_memory=False)
    return df

def test_file_exists_and_non_empty(data_path):
    """Kiểm tra file dữ liệu tồn tại và có kích thước hợp lệ (> 100MB)."""
    assert os.path.exists(data_path), f"File không tồn tại: {data_path}"
    file_size = os.path.getsize(data_path)
    assert file_size > 100 * 1024 * 1024, "File dữ liệu phải lớn hơn 100MB"

def test_schema_columns(dataset_df):
    """Kiểm tra số cột và cột target bắt buộc."""
    assert "diseases" in dataset_df.columns, "Cột 'diseases' (Target) phải tồn tại"
    assert len(dataset_df.columns) == 378, f"Kỳ vọng 378 cột, nhận được {len(dataset_df.columns)}"

def test_no_null_values(dataset_df):
    """Kiểm tra không có giá trị missing (NaN/null)."""
    assert dataset_df.isnull().sum().sum() == 0, "Dữ liệu không được chứa giá trị null"

def test_binary_feature_values(dataset_df):
    """Kiểm tra tất cả các cột triệu chứng chỉ chứa giá trị nhị phân {0, 1}."""
    feature_cols = [c for c in dataset_df.columns if c != "diseases"]
    unique_values = np.unique(dataset_df[feature_cols].values)
    for val in unique_values:
        assert val in [0, 1], f"Phát hiện giá trị bất thường không phải 0 hoặc 1: {val}"

def test_sample_has_at_least_one_symptom(dataset_df):
    """Kiểm tra mỗi mẫu phải có ít nhất 1 triệu chứng."""
    feature_cols = [c for c in dataset_df.columns if c != "diseases"]
    symptom_sum = dataset_df[feature_cols].sum(axis=1)
    assert (symptom_sum >= 1).all(), "Mỗi bản ghi phải có ít nhất 1 triệu chứng được kích hoạt"
