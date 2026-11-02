# BÁO CÁO GIAI ĐOẠN 3 — FEATURE SCHEMA VÀ TIỀN XỬ LÝ

## Phạm vi

Phase 3 tạo feature schema từ cột thực tế của `data/processed/exp1_full/train.csv`, là dữ liệu train do Phase 2 tạo ra. Schema không được dùng như dữ liệu huấn luyện; dữ liệu đưa vào mô hình luôn được đọc trực tiếp từ `train.csv` và `val.csv` của từng experiment.

## Quy tắc đã triển khai

- Target: `diseases`.
- Feature: 377 cột còn lại, giữ nguyên tên và thứ tự từ CSV thực tế.
- Giá trị feature phải là nhị phân `0` hoặc `1`; giá trị thiếu, cột thiếu, cột thừa hoặc sai thứ tự đều bị từ chối.
- Chuyển đổi sang CSR sparse chỉ là biểu diễn tính toán của cùng giá trị nhị phân; không loại feature, không giảm chiều và không fit tham số trên validation/test.
- 49 feature zero-variance được giữ nguyên theo quyết định hiện tại.

## Đầu ra

- Mã nguồn: `src/ml_core/preprocessing/feature_schema.py`.
- Artifact kiểm tra/tái lập: `docs/phase3_feature_schema.json`.
- Tests: `tests/test_feature_schema.py`.

## Liên kết giai đoạn

Phase 4 tự tạo schema trong bộ nhớ từ CSV train thực tế của từng experiment và dùng schema chỉ để xác thực dataframe trước khi transform. Phase 4 không đọc `experiments_manifest.json`, artifact baseline cũ hay file JSON schema làm input huấn luyện.
