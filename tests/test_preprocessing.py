"""
Unit tests kiểm tra tính đúng đắn của pipeline phân chia dữ liệu (Data Split & Leakage Tests).
Chạy bằng pytest: pytest tests/test_preprocessing.py
"""

import os
import json
import pytest
import pandas as pd

PROCESSED_DIR = os.path.join("data", "processed")
MANIFEST_PATH = os.path.join(PROCESSED_DIR, "experiments_manifest.json")

def test_experiments_manifest_exists():
    """Kiểm tra file manifest tổng hợp tồn tại."""
    assert os.path.exists(MANIFEST_PATH), f"Không tìm thấy manifest tại {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert "exp1_full" in manifest
    assert "exp2_dedup" in manifest
    assert "exp3_capped" in manifest

@pytest.mark.parametrize("exp_id", ["exp1_full", "exp2_dedup", "exp3_capped"])
def test_experiment_files_exist(exp_id):
    """Kiểm tra từng thí nghiệm có đầy đủ train.csv, val.csv, test.csv, metadata.json."""
    exp_dir = os.path.join(PROCESSED_DIR, exp_id)
    assert os.path.exists(os.path.join(exp_dir, "train.csv"))
    assert os.path.exists(os.path.join(exp_dir, "val.csv"))
    assert os.path.exists(os.path.join(exp_dir, "test.csv"))
    assert os.path.exists(os.path.join(exp_dir, "metadata.json"))

def test_identical_test_and_val_across_experiments():
    """Kiểm tra tập Test và Val của cả 3 thí nghiệm hoàn toàn giống nhau để đảm bảo so sánh công bằng."""
    exp1_test = pd.read_csv(os.path.join(PROCESSED_DIR, "exp1_full", "test.csv"))
    exp2_test = pd.read_csv(os.path.join(PROCESSED_DIR, "exp2_dedup", "test.csv"))
    exp3_test = pd.read_csv(os.path.join(PROCESSED_DIR, "exp3_capped", "test.csv"))

    assert len(exp1_test) == len(exp2_test) == len(exp3_test) == 28432
    assert exp1_test.equals(exp2_test)
    assert exp1_test.equals(exp3_test)

def test_zero_leakage_between_train_and_test():
    """Kiểm tra triệt để: KHÔNG CÓ BẤT KỲ TỔ HỢP (Disease, Symptoms) NÀO BỊ RÒ RỈ giữa Train và Test."""
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "exp1_full", "test.csv"))
    test_tuples = set(tuple(x) for x in test_df.values)

    for exp_id in ["exp1_full", "exp2_dedup", "exp3_capped"]:
        train_df = pd.read_csv(os.path.join(PROCESSED_DIR, exp_id, "train.csv"))
        train_tuples = set(tuple(x) for x in train_df.values)
        overlap = train_tuples & test_tuples
        assert len(overlap) == 0, f"Phát hiện rò rỉ dữ liệu ({len(overlap)} mẫu trùng lặp) giữa Train và Test ở {exp_id}!"

def test_capping_k_rule_in_exp3():
    """Kiểm tra ở Exp 3 (K=3), không có tổ hợp nào xuất hiện vượt quá 3 lần."""
    exp3_train = pd.read_csv(os.path.join(PROCESSED_DIR, "exp3_capped", "train.csv"))
    max_count = exp3_train.groupby(list(exp3_train.columns)).size().max()
    assert max_count <= 3, f"Phát hiện tổ hợp xuất hiện {max_count} lần trong exp3_capped (vượt quá K=3)"

def test_full_dedup_rule_in_exp2():
    """Kiểm tra ở Exp 2 (Deduplication), không có bất kỳ dòng nào bị lặp lại."""
    exp2_train = pd.read_csv(os.path.join(PROCESSED_DIR, "exp2_dedup", "train.csv"))
    assert exp2_train.duplicated().sum() == 0, "Exp 2 còn chứa các dòng trùng lặp hoàn toàn!"

