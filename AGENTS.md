# AGENTS.md — Medical Chatbot ML Core

## 1. Mục tiêu và phạm vi hiện tại

Dự án hiện tập trung vào ML Core phân loại bệnh từ các triệu chứng nhị phân.

Phạm vi đã có trong repository:

- Python package tại `src/ml_core/`.
- Dataset CSV với target `diseases` và 377 cột triệu chứng.
- Data audit, chia dữ liệu và thử nghiệm mô hình classical ML.
- Tests tại `tests/`.
- Scripts vận hành tại `scripts/`.

Không xem đầu ra mô hình là chẩn đoán y khoa chắc chắn. Không tự bổ sung tư vấn điều trị, kê đơn hoặc khẳng định giá trị lâm sàng.

## 2. Công nghệ và cấu trúc hiện có

- Ngôn ngữ: Python.
- Thư viện hiện có: numpy, pandas, scikit-learn, pytest, ipykernel, matplotlib, seaborn; xgboost đang được thêm vào trong thay đổi chưa commit.
- Package chính: `src/ml_core/`.
- Dữ liệu gốc: `data/raw/disease and symptoms dataset.csv`.
- Dữ liệu thí nghiệm đã xử lý: `data/processed/`.
- Artifacts mô hình: `models/`.
- Kết quả thí nghiệm: `experiments/`.
- Báo cáo và tài liệu: `docs/`.
- Script tác vụ: `scripts/`.

Không tự thay đổi công nghệ, cấu trúc package hoặc di chuyển dữ liệu nếu chưa được người dùng chấp thuận.

## 3. Quy tắc dữ liệu và thí nghiệm

- Không sửa, xóa, ghi đè hoặc di chuyển dữ liệu gốc.
- Không commit dữ liệu raw, processed, model artifact hoặc kết quả thí nghiệm lớn khi `.gitignore` đã loại trừ chúng.
- Giữ độc lập các tập train, validation và test; không dùng test set để chọn mô hình hoặc điều chỉnh feature.
- Ghi lại seed, dữ liệu đầu vào, chính sách split, feature schema, cấu hình mô hình và metric cho mọi thí nghiệm có kết quả được báo cáo.
- Các phiên thử nghiệm Phase 2 được quản lý độc lập theo ba tập dữ liệu hiện có:
  - `exp1_full`
  - `exp2_dedup`
  - `exp3_capped`
- Chỉ so sánh các mô hình giữa các phiên khi dùng cùng chiến lược đánh giá đã được ghi nhận.
- Baseline đầu tiên dùng đầy đủ 377 đặc trưng raw. Không tự loại 49 zero-variance features hoặc áp dụng feature selection trước khi có chỉ đạo tiếp theo.
- Duy trì Macro-F1, Weighted-F1, Top-1 và Top-k khi mô hình hỗ trợ score/probability; không dùng Accuracy đơn lẻ để kết luận chất lượng.

## 4. Ranh giới thay đổi và quyền quyết định

Không tự ý:

- Đổi target column, schema dữ liệu, chiến lược split, seed hoặc tiêu chí đánh giá.
- Thay đổi chính sách xử lý duplicate, conflict label hoặc rare label.
- Loại bỏ/chọn lọc feature, giảm chiều, resampling hoặc tuning hyperparameter ngoài phạm vi người dùng đã xác nhận.
- Thay kiến trúc, framework, API công khai hoặc công nghệ dự án.
- Reset, revert, xóa hoặc ghi đè các thay đổi Git hiện có.
- Commit, push, tạo release, hoặc thay đổi cấu hình remote khi chưa được yêu cầu.

Nếu yêu cầu chưa đủ rõ để ảnh hưởng đến các quyết định trên, dừng lại và nêu rõ thông tin cần xác nhận.

## 5. Quy trình phát triển

Trước khi sửa mã:

1. Kiểm tra `git status`, các thay đổi chưa commit và file liên quan.
2. Đọc tài liệu, script và test liên quan trực tiếp.
3. Xác định tác động đến dữ liệu, split, metric và khả năng tái lập.

Khi triển khai:

1. Thực hiện thay đổi nhỏ nhất đáp ứng yêu cầu đã xác nhận.
2. Không ghi đè artifacts hoặc dữ liệu thí nghiệm có sẵn.
3. Ghi kết quả thí nghiệm vào vị trí phù hợp dưới `experiments/` hoặc `docs/` theo cấu trúc hiện có.
4. Cập nhật tài liệu khi thay đổi làm khác hành vi, dữ liệu đầu vào, quy trình chạy hoặc kết quả được chọn làm mốc.

## 6. Kiểm thử và xác minh

- Chạy các test liên quan trước và sau thay đổi.
- Lệnh test hiện dùng: `\.venv\Scripts\python.exe -m pytest -q`
- Với script xử lý dữ liệu hoặc huấn luyện, xác minh:
  - đường dẫn đầu vào/đầu ra;
  - schema và số cột;
  - số mẫu và số lớp mỗi split;
  - không có overlap giữa train/validation/test theo chính sách đang áp dụng;
  - metric được tính trên đúng split.
- Báo cáo rõ nếu một test, dependency hoặc script không chạy được; không che giấu lỗi bằng cách bỏ qua test.

## 7. Tài liệu và bàn giao

Khi hoàn tất một hạng mục, bàn giao ngắn gọn:

- File đã thay đổi.
- Mục tiêu đã hoàn thành.
- Lệnh xác minh đã chạy và kết quả.
- Kết quả hoặc artifact được tạo, nếu có.
- Hạn chế, rủi ro và quyết định còn chờ phê duyệt.

Không tuyên bố mô hình sẵn sàng production hoặc có giá trị lâm sàng nếu chưa có đánh giá và phê duyệt tương ứng.
