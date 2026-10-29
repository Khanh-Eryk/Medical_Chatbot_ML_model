# BÁO CÁO KIỂM TOÁN DỮ LIỆU (DATA AUDIT REPORT)
## Dự án: Medical Chatbot — Phân loại bệnh dựa trên triệu chứng (ML Core)

**Ngày thực hiện:** 29/10/2026  
**Tệp dữ liệu kiểm toán:** `Disease and symptoms dataset.csv`  
**File kết quả JSON định lượng:** [data_audit_metrics.json](file:///c:/Users/Admin/.vscode/ProjectCV/Medical_chatbot/docs/data_audit_metrics.json)  
**Trạng thái tài liệu:** Báo cáo kiểm toán dữ liệu & Đề xuất phương án xử lý — Chờ người dùng xem xét và phê duyệt.

---

## 1. Tổng quan kỹ thuật bộ dữ liệu

| Thuộc tính | Giá trị kiểm toán | Ghi chú kỹ thuật |
|---|---|---|
| Dung lượng tệp | 190,786,867 bytes (~181.95 MB) | Dữ liệu gốc được bảo toàn nguyên vẹn |
| Định dạng | CSV, mã hóa UTF-8 | Phân cách dấu phẩy, có escape quotes |
| Tổng số dòng (Records) | **246,945** | Khá lớn đối với bài toán tabular phân loại |
| Tổng số cột (Columns) | **378** | 1 cột nhãn mục tiêu + 377 cột đặc trưng |
| Cột mục tiêu (Target) | `diseases` | Nhãn phân loại dạng chuỗi ký tự (string) |
| Cột đặc trưng (Features) | 377 cột | Tên triệu chứng y khoa |
| Kiểu dữ liệu đặc trưng | 100% `int64` | Giá trị nhị phân {0, 1} |
| Missing / Null Values | **0 (0.00%)** | Không có bất kỳ ô trống nào |
| Độ thưa đặc trưng (Sparsity) | **98.59%** | Ma trận triệu chứng rất thưa |
| Số triệu chứng trung bình/mẫu | **5.33** triệu chứng | Min = 1, Max = 12 |
| Mẫu có 0 triệu chứng | **0 mẫu** | Mọi bản ghi đều có ít nhất 1 triệu chứng |

---

## 2. Các phát hiện quan trọng từ dữ liệu thực tế

### 2.1. Phân phối nhãn bệnh (`diseases`) — Mất cân bằng lớp nghiêm trọng (Severe Imbalance)
- **Tổng số nhãn bệnh:** **773** bệnh duy nhất.
- **Biên độ số mẫu rất lớn:**
  - Bệnh có nhiều mẫu nhất: `cystitis` (1,219 mẫu).
  - Bệnh có ít mẫu nhất: Có **19 bệnh chỉ có đúng 1 mẫu duy nhất** (ví dụ: `diabetes`, `thalassemia`, `heat stroke`, `open wound of the head`, `myocarditis`, `kaposi sarcoma`,...).
  - **61 bệnh** có $\le$ 5 mẫu.
  - **96 bệnh** có $<$ 10 mẫu.
  - Trung vị: 168 mẫu/bệnh. Trung bình: 319.46 mẫu/bệnh.
- **Hệ quả kỹ thuật:**
  - Không thể thực hiện phân chia tầng trực tiếp (`StratifiedSplit` chuẩn) cho các nhãn chỉ có 1 hoặc 2 mẫu sang cả 3 tập Train/Validation/Test nếu không có chính sách xử lý rõ ràng.

### 2.2. Đặc trưng tĩnh (Constant Zero Features)
- Có **49 / 377 cột triệu chứng có tổng giá trị bằng 0** (không bao giờ xuất hiện ở bất kỳ mẫu nào trong 246,945 dòng).
- *Ví dụ:* `pus in sputum`, `underweight`, `arm cramps or spasms`, `pallor`, `joint stiffness or tightness`, `eye strain`, `pus in urine`, `throat irritation`, `disturbance of smell or taste`, `jaw pain`,...
- **Hệ quả kỹ thuật:** 49 cột này không mang thông tin phân biệt (zero variance). Giữ lại sẽ làm tăng chiều không gian đặc trưng vô ích và tốn bộ nhớ tính toán.

### 2.3. Trùng lặp hoàn toàn (Exact Duplicates)
- **57,298 dòng trùng lặp hoàn toàn** (chiếm **23.20%** toàn bộ bộ dữ liệu).
- Nghĩa là cả 377 triệu chứng và nhãn bệnh đều giống hệt với các dòng khác.
- **Hệ quả kỹ thuật:** Nếu chia tập ngẫu nhiên không xử lý trùng lặp, các mẫu trùng lặp này sẽ xuất hiện đồng thời ở cả tập Train và Test $\rightarrow$ **Data Leakage nghiêm trọng**, làm Accuracy ảo rất cao nhưng không phản ánh khả năng tổng quát hóa thực tế.

### 2.4. Trùng lặp đặc trưng & Xung đột nhãn (Label Inconsistency / Contradiction)
- Có **121,344 dòng** (chiếm **49.14%**) chia sẻ vector triệu chứng giống với ít nhất một dòng khác.
- Đáng chú ý, có **12,634 vector triệu chứng duy nhất** bị gán cho $\ge 2$ bệnh khác nhau, ảnh hưởng tới **39,728 dòng** (**16.09%** dataset).
- *Nguyên nhân y khoa thực tế:* Nhiều bệnh lý khác nhau có cùng tập hợp các triệu chứng cơ bản (ví dụ sốt + ho + đau đầu có thể là Cúm, Viêm phế quản, hoặc COVID). Trong dữ liệu nhị phân không có thuộc tính phụ trợ (tuổi, giới tính, tiền sử, mức độ nặng), các ca này biểu diễn giống hệt nhau nhưng nhãn khác nhau.

---

## 3. Đề xuất các phương án xử lý (Cần Người dùng phê duyệt)

Căn cứ vào `Technical_document.md` và `Technical_specification.md`, Coding Agent không tự quyết định các chính sách này mà lập phân tích để bạn xem xét:

### Vấn đề 1: Xử lý 49 cột triệu chứng bằng 0 (Zero-variance features)
- **Phương án A (Khuyến nghị):** Loại bỏ 49 cột này trong Pipeline tiền xử lý của ML Core (chỉ giữ lại 328 triệu chứng thực sự xuất hiện). Lưu metadata danh sách 328 cột đã chọn để áp dụng nhất quán cho suy luận (Inference).
  - *Ưu điểm:* Tiết kiệm RAM, tăng tốc độ huấn luyện mô hình, tránh nhiễu ma trận.
  - *Nhược điểm:* Khi chatbot tương tác, nếu người dùng đề cập đến 1 trong 49 triệu chứng này, ML Core cần biết triệu chứng đó không nằm trong active feature set.
- **Phương án B:** Giữ nguyên 377 cột.
  - *Ưu điểm:* Không thay đổi kích thước vector đầu vào so với CSV gốc.
  - *Nhược điểm:* Tốn tài nguyên vô ích, một số thuật toán tuyến tính hoặc tính khoảng cách sẽ mang trọng số 0.

### Vấn đề 2: Xử lý 57,298 dòng trùng lặp hoàn toàn (Exact Duplicates)
- **Phương án A (Khuyến nghị):** Loại bỏ các dòng trùng lặp hoàn toàn trước khi chia tập dữ liệu (Deduplication). Giữ lại 1 bản ghi đại diện cho mỗi tổ hợp (Symptoms, Disease).
  - *Ưu điểm:* Loại bỏ triệt để rò rỉ dữ liệu giữa train/val/test; đánh giá mô hình khách quan và trung thực.
  - *Nhược điểm:* Giảm số lượng dòng huấn luyện (từ ~247k xuống ~189k dòng), nhưng 189k mẫu phân biệt vẫn là cỡ mẫu rất lớn cho Classical ML.
- **Phương án B:** Giữ nguyên trùng lặp, chia ngẫu nhiên.
  - *Nhược điểm:* Rò rỉ dữ liệu nghiêm trọng, vi phạm nguyên tắc kỹ thuật cốt lõi của dự án. Không khuyến nghị.

### Vấn đề 3: Xử lý 39,728 dòng xung đột nhãn (Label Conflicts)
- **Phương án A (Khuyến nghị - Giữ nguyên & Sử dụng Top-k/Probabilistic Evaluation):**
  Giữ lại các dòng này. Chấp nhận đây là đặc thù vốn có của y khoa (các bệnh trùng triệu chứng). Đánh giá mô hình bằng **Top-3 và Top-5 Accuracy**, Macro F1 và Log-loss/Brier Score thay vì chỉ Top-1 Accuracy.
  - *Ưu điểm:* Bảo toàn tính chân thực của phân phối bệnh thực tế. Phù hợp cho kịch bản Chatbot tư vấn (đưa ra danh sách các bệnh có khả năng nhất kèm xác suất).
  - *Nhược điểm:* Điểm Top-1 Accuracy thuần túy trên tập test sẽ bị trần (không thể đạt 100% do có nhiễu Bayes cố hữu).
- **Phương án B:** Lọc bỏ các vector xung đột nhãn.
  - *Ưu điểm:* Tăng Top-1 Accuracy của mô hình.
  - *Nhược điểm:* Mất đi 16% dữ liệu thực tế; mô hình trở nên phi thực tế vì trong lâm sàng các bệnh luôn có triệu chứng chồng lấn.

### Vấn đề 4: Chiến lược chia tập dữ liệu với 19 bệnh chỉ có 1 mẫu (Severe Tail Imbalance)
- **Phương án A (Khuyến nghị - Lọc ngưỡng thiểu số kết hợp Stratified Split):**
  Đặt ngưỡng mẫu tối thiểu cho các bệnh đưa vào bài toán phân loại chính của ML Core (ví dụ: chỉ phân loại các bệnh có $\ge 10$ hoặc $\ge 20$ mẫu để đảm bảo có ít nhất vài mẫu trong train, val và test). Các bệnh cực hiếm ($\le 5$ mẫu) được ghi nhận trong từ điển metadata là "Rare Diseases - Chuyển tuyến/Từ chối dự đoán (Abstention Rule)".
  - *Ưu điểm:* Phân chia tầng (Stratified 3-way split: 70% Train / 15% Val / 15% Test) hoàn toàn khả thi, kiểm định mô hình có ý nghĩa thống kê. Phù hợp nguyên tắc An toàn Y tế (không dự đoán bệnh khi dữ liệu chỉ có 1 ca duy nhất).
  - *Nhược điểm:* Số lớp phân loại giảm từ 773 xuống khoảng 677 bệnh (vẫn bao phủ $\ge 99.8\%$ số ca bệnh trong dataset).
- **Phương án B:** Giữ nguyên toàn bộ 773 bệnh, các bệnh có 1 mẫu bắt buộc phải xếp vào tập Train (không thể test).
  - *Ưu điểm:* Giữ đủ 773 nhãn.
  - *Nhược điểm:* Không thể đánh giá được hiệu năng của mô hình trên các bệnh hiếm đó (do không có mẫu trong test).

---

## 4. Tóm tắt câu hỏi cần Người dùng phê duyệt

Xin vui lòng xem xét và cho ý kiến phê duyệt đối với các lựa chọn sau để tiến hành **Giai đoạn 2 (Chuẩn bị dữ liệu và chia tập)**:

1. **Xử lý cột triệu chứng:** Có đồng ý loại bỏ **49 cột triệu chứng có tổng bằng 0** trong feature schema không?
2. **Xử lý trùng lặp:** Có đồng ý loại bỏ **57,298 dòng trùng lặp hoàn toàn** trước khi chia tập dữ liệu để chống Data Leakage không?
3. **Xung đột nhãn:** Có đồng ý **giữ lại các mẫu trùng triệu chứng nhưng khác bệnh** và áp dụng đánh giá xác suất / Top-k Accuracy (Top-3, Top-5) thay vì chỉ đo Top-1 không?
4. **Ngưỡng nhãn thiểu số (Rare labels):** Bạn đồng ý với phương án nào cho các bệnh có $\le 5$ hoặc $\le 10$ mẫu (loại khỏi tập phân loại chính và gán vào chính sách Abstention/Cảnh báo bệnh hiếm, hay giữ nguyên và chỉ đưa vào Train)?

