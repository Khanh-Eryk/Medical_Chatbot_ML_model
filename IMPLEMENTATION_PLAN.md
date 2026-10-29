# KẾ HOẠCH TRIỂN KHAI — MEDICAL CHATBOT

**Phiên bản:** 0.1  
**Trạng thái:** Bản dự thảo chờ phê duyệt  
**Định hướng:** Ưu tiên nghiên cứu ML, xây dựng ML Core độc lập và tích hợp chatbot MVP sau.

## 1. Nguyên tắc dự án

1. Ưu tiên nghiên cứu ML và chất lượng mô hình dự đoán.
2. ML Core phải là một Python package độc lập, không phụ thuộc vào chatbot, FastAPI hay framework Agent.
3. Lựa chọn mô hình dựa trên thực nghiệm có hệ thống, không mặc định thuật toán nào tốt nhất.
4. Đánh giá mô hình phải hạn chế rò rỉ dữ liệu và không chỉ dựa vào Accuracy.
5. Giữ nguyên dữ liệu gốc; mọi dữ liệu đã xử lý phải được tạo thành đầu ra riêng.
6. Chatbot không được trình bày dự đoán của mô hình như một chẩn đoán y khoa chắc chắn.
7. Các quy tắc an toàn phải được triển khai bằng logic ứng dụng rõ ràng, không chỉ dựa vào prompt của LLM.
8. Các thí nghiệm, quyết định tiền xử lý, cấu hình và kết quả đánh giá phải có khả năng tái lập.
9. **Coding Agent không được tự ý thay đổi yêu cầu, kiến trúc, schema dữ liệu, giao diện công khai hoặc công nghệ đã được phê duyệt.**

## 2. Phạm vi dự án

### 2.1. Trong phạm vi

- Kiểm tra và đánh giá chất lượng dataset.
- Xây dựng pipeline tiền xử lý và biểu diễn đặc trưng.
- Xây dựng mô hình cơ sở (Baseline).
- So sánh nhiều thuật toán phân loại phù hợp.
- Thử nghiệm xử lý mất cân bằng lớp và kỹ thuật tạo đặc trưng.
- Tối ưu siêu tham số.
- Đánh giá mô hình cuối cùng trên tập kiểm tra độc lập.
- Đóng gói mô hình thành Python package có thể tái sử dụng.
- Cung cấp giao diện dự đoán bằng Python với hợp đồng đầu vào/đầu ra rõ ràng.
- Tích hợp ML Core vào chatbot tư vấn triệu chứng ở mức MVP.
- Xây dựng kiểm thử đơn vị, kiểm thử tích hợp và kiểm thử hồi quy.
- Tài liệu hóa giới hạn và kết quả nghiên cứu của mô hình.

### 2.2. Ngoài phạm vi MVP ban đầu

- Chẩn đoán y khoa xác định.
- Kê đơn hoặc tạo phác đồ điều trị cá nhân hóa.
- Khẳng định mô hình có giá trị lâm sàng chỉ dựa trên Accuracy.
- Tự động huấn luyện lại mô hình trong môi trường production.
- Xây dựng hạ tầng ML phân tán phức tạp hoặc dịch vụ suy luận riêng khi chưa có nhu cầu.
- Hỗ trợ dự đoán bệnh ngoài phạm vi dataset đã được xác minh.

## 3. Các giai đoạn triển khai

### Giai đoạn 0 — Khảo sát repository và thiết lập dự án

**Mục tiêu:** Xác định hiện trạng trước khi thay đổi mã nguồn.

Công việc:
- Kiểm tra cấu trúc thư mục, mã nguồn, dependencies và trạng thái Git.
- Đọc các quy tắc dự án và tài liệu hiện có.
- Xác định vị trí thực tế của dataset.
- Kiểm tra giấy phép và tài liệu nguồn dữ liệu.
- Xác định mã nguồn, thành phần và tài nguyên có thể tái sử dụng.
- Đề xuất cấu trúc repository và chiến lược quản lý dependencies để người dùng phê duyệt.
- Thiết lập môi trường phát triển có thể tái lập.

Đầu ra:
- Báo cáo khảo sát repository.
- Cấu trúc triển khai được phê duyệt.
- Cấu hình dependencies ban đầu.
- Danh mục tài liệu dự án.

Tiêu chí hoàn thành:
- Đã khảo sát mã nguồn trước khi chỉnh sửa.
- Dữ liệu gốc được bảo toàn.
- Không có thay đổi kiến trúc ngoài phạm vi được phê duyệt.

### Giai đoạn 1 — Kiểm toán dữ liệu (Data Audit)

**Mục tiêu:** Hiểu dữ liệu thực tế trước khi tiền xử lý và huấn luyện.

Công việc:
- Kiểm tra encoding, dấu phân cách, header, số dòng, số cột và kiểu dữ liệu.
- Xác minh cột target và các cột feature.
- Thống kê số nhãn bệnh và số mẫu của từng nhãn.
- Kiểm tra giá trị thiếu, sai định dạng và không hợp lệ.
- Xác minh ý nghĩa và miền giá trị của các feature triệu chứng.
- Phát hiện các dòng trùng lặp hoàn toàn.
- Phát hiện các mẫu có cùng vector đặc trưng nhưng nhãn bệnh khác nhau.
- Phân tích độ thưa của feature, tần suất triệu chứng và mất cân bằng lớp.
- Kiểm tra nguy cơ rò rỉ nhãn và các dấu hiệu đặc thù của dữ liệu tổng hợp.
- Ghi nhận giới hạn và các câu hỏi chưa được giải quyết.

Đầu ra:
- `docs/data_audit_report.md`
- Thống kê kiểm toán ở định dạng máy có thể đọc.
- Báo cáo phân phối nhãn.
- Danh sách vấn đề dữ liệu và phương án xử lý đề xuất.

Tiêu chí hoàn thành:
- Mỗi đề xuất làm sạch dữ liệu đều có lý do.
- Không tự ý loại bỏ hoặc sửa nhãn dữ liệu.
- Các quyết định chưa được thống nhất đều được ghi nhận.

### Giai đoạn 2 — Chuẩn bị dữ liệu và chia tập

**Mục tiêu:** Xây dựng pipeline chuẩn bị dữ liệu có khả năng tái lập.

Công việc:
- Xây dựng pipeline tiền xử lý xác định.
- Xác định chính sách xử lý dữ liệu trùng lặp và xung đột nhãn.
- Lựa chọn chiến lược chia train/validation/test dựa trên kết quả kiểm toán.
- Sử dụng chia tầng theo nhãn khi phù hợp và khả thi.
- Ngăn các mẫu giống nhau hoặc có liên quan bị rò rỉ giữa các tập.
- Chỉ fit các bước tiền xử lý có học tham số trên tập huấn luyện.
- Ghi lại random seed, số lượng mẫu, phân phối nhãn và nguồn gốc dữ liệu.
- Giữ tập test độc lập cho đến khi hoàn tất lựa chọn mô hình.

Đầu ra:
- Mã nguồn chuẩn bị dữ liệu có thể tái sử dụng.
- Manifest hoặc metadata mô tả các tập dữ liệu.
- Tài liệu chiến lược chia tập.

Tiêu chí hoàn thành:
- Train và inference sử dụng cùng feature schema.
- Các nguy cơ rò rỉ dữ liệu đã được xử lý hoặc ghi nhận.
- Có thể tái tạo các tập dữ liệu đã chia.

### Giai đoạn 3 — Tạo đặc trưng và tiền xử lý

**Mục tiêu:** Xây dựng biểu diễn triệu chứng ổn định cho mô hình.

Công việc:
- Xây dựng feature schema chuẩn từ dataset đã xác minh.
- Xác định định dạng đầu vào và thứ tự feature.
- Kiểm tra giá trị nhị phân và cách xử lý feature bị thiếu.
- So sánh các phương án tiền xử lý hoặc lựa chọn đặc trưng có cơ sở.
- Chỉ xem xét giảm chiều khi phù hợp và có đánh giá.
- Lưu feature mapping cùng mô hình đã huấn luyện.
- Không sử dụng target hoặc thông tin không có sẵn khi dự đoán.

Đầu ra:
- Đặc tả feature schema.
- Mã nguồn tiền xử lý có thể tái sử dụng.
- Kiểm thử xác thực feature.
- Báo cáo các thí nghiệm tạo đặc trưng.

Tiêu chí hoàn thành:
- Cùng một đầu vào luôn tạo ra cùng biểu diễn đặc trưng.
- Feature thiếu, không xác định hoặc sai định dạng được xử lý rõ ràng.
- Không fit preprocessing trên dữ liệu validation hoặc test.

### Giai đoạn 4 — Xây dựng mô hình cơ sở (Baseline)

**Mục tiêu:** Thiết lập mốc hiệu năng để so sánh.

Công việc:
- Xây dựng baseline đơn giản phù hợp với bài toán phân loại đa lớp.
- Thiết lập quy trình huấn luyện và đánh giá có thể tái lập.
- Ghi nhận cấu hình và kết quả baseline.
- Tạo confusion matrix và các metric theo từng lớp.
- Xác định các lớp thường bị dự đoán nhầm.

Các hướng baseline có thể gồm mô hình dự đoán đơn giản, mô hình tuyến tính hoặc mô hình cây, tùy theo kết quả kiểm toán dữ liệu.

Đầu ra:
- Script huấn luyện baseline.
- Báo cáo đánh giá baseline.
- Cấu hình và metadata của baseline.

Tiêu chí hoàn thành:
- Kết quả có thể tái lập.
- Metric được tính trên đúng tập dữ liệu.
- Giới hạn của baseline được ghi nhận.

### Giai đoạn 5 — So sánh thuật toán

**Mục tiêu:** Tìm các thuật toán tiềm năng bằng thực nghiệm có kiểm soát.

Công việc:
- Chọn thuật toán phù hợp với cách biểu diễn feature, kích thước dataset và tài nguyên tính toán.
- So sánh các ứng viên trên cùng chiến lược chia tập và quy trình đánh giá.
- Ghi lại cấu hình, random seed, thời gian chạy và tài nguyên sử dụng.
- So sánh train và validation để phát hiện overfitting.
- Phân tích hiệu năng từng lớp và các kiểu dự đoán nhầm.
- Duy trì baseline làm mốc tham chiếu.

Các thuật toán có thể thử nghiệm:
- K-Nearest Neighbors (K-NN).
- Decision Tree.
- Random Forest.
- Extra Trees.
- Support Vector Machine.
- Các thuật toán Gradient Boosting phù hợp.

Danh sách trên chỉ là đề xuất ban đầu. Danh sách chính thức phụ thuộc vào kết quả kiểm toán và giới hạn tài nguyên.

Đầu ra:
- Cấu hình các thí nghiệm.
- Báo cáo so sánh mô hình.
- Danh sách mô hình tiềm năng.

Tiêu chí hoàn thành:
- Các phép so sánh có thể tái lập và công bằng.
- Không lựa chọn mô hình dựa trên kết quả test.
- Quyết định lựa chọn dựa trên metric và đánh đổi đã ghi nhận.

### Giai đoạn 6 — Tối ưu mô hình

**Mục tiêu:** Cải thiện khả năng tổng quát hóa và hiệu năng theo từng lớp.

Công việc:
- Tối ưu siêu tham số bằng validation hoặc cross-validation.
- Đánh giá chiến lược gán trọng số cho lớp.
- Thử nghiệm lấy mẫu lại khi có cơ sở.
- Đánh giá lựa chọn feature và các phương án tiền xử lý liên quan.
- So sánh metric macro và weighted.
- Đo thời gian huấn luyện, độ trễ suy luận và kích thước mô hình.
- Kiểm tra calibration nếu score dự đoán được dùng để quyết định từ chối dự đoán.
- Tránh điều chỉnh mô hình theo từng mẫu test.

Đầu ra:
- Cấu hình tuning và kết quả thí nghiệm.
- Báo cáo tối ưu.
- Cấu hình mô hình được đề xuất.

Tiêu chí hoàn thành:
- Cải thiện được chứng minh so với baseline.
- Cấu hình có thể tái lập.
- Các đánh đổi về hiệu năng được ghi nhận.

### Giai đoạn 7 — Đánh giá cuối và lựa chọn mô hình

**Mục tiêu:** Chọn mô hình cuối cùng bằng tập test độc lập.

Metric chính:
- Macro-F1.
- Precision và Recall theo từng lớp.
- Weighted-F1.
- Confusion matrix.

Metric bổ sung khi phù hợp:
- Accuracy.
- Balanced Accuracy.
- Metric đánh giá calibration.
- Độ trễ suy luận và kích thước mô hình.

Công việc:
- Cố định ứng viên và cấu hình trước khi đánh giá cuối.
- Đánh giá trên test set chưa được sử dụng để lựa chọn mô hình.
- So sánh với baseline.
- Phân tích các lớp có hiệu năng thấp hoặc ít mẫu.
- Ghi nhận độ bất định, giới hạn dữ liệu và trường hợp thất bại.
- Quyết định mô hình có phù hợp với MVP hay chưa.

Đầu ra:
- Báo cáo đánh giá cuối.
- Lý do lựa chọn mô hình.
- Model card.
- Danh sách giới hạn và khuyến nghị triển khai.

Tiêu chí hoàn thành:
- Không đưa ra tuyên bố về giá trị lâm sàng khi chưa có bằng chứng.
- Báo cáo có phân tích theo từng lớp.
- Không phê duyệt mô hình chỉ vì đạt một ngưỡng Accuracy sơ bộ.
- Nếu bằng chứng chưa đủ, mô hình tiếp tục ở trạng thái thử nghiệm.

### Giai đoạn 8 — Đóng gói ML Core

**Mục tiêu:** Chuyển mô hình thành Python package độc lập và tái sử dụng được.

Trách nhiệm:
- Tải mô hình đã được phê duyệt và preprocessing tương ứng.
- Xác thực đầu vào có cấu trúc.
- Chuyển đổi đầu vào theo feature schema đã được phê duyệt.
- Trả về dự đoán và metadata theo hợp đồng đã định nghĩa.
- Xử lý đầu vào sai và lỗi tải mô hình.
- Không phụ thuộc vào chatbot hoặc framework web.

Giao diện công khai dự kiến:
- `load_model(...)`
- `predict(...)`
- `predict_proba(...)`, chỉ khi estimator hỗ trợ xác suất có ý nghĩa và hợp đồng đầu ra được định nghĩa rõ.

Hợp đồng API chính thức cần được xem xét trước khi triển khai.

Đầu ra dự đoán dự kiến:
- Nhãn bệnh được dự đoán.
- Danh sách nhãn ứng viên được xếp hạng và score, nếu được hỗ trợ.
- Phiên bản mô hình.
- Trạng thái xác thực đầu vào.
- Metadata phục vụ chẩn đoán kỹ thuật khi cần.

ML Core không tạo lời khuyên y tế hoặc câu trả lời hội thoại.

Đầu ra:
- Python package có thể cài đặt.
- Model artifact và feature schema có phiên bản.
- Tài liệu API công khai.
- Kiểm thử đơn vị và kiểm thử hợp đồng.
- Ví dụ sử dụng từ một script Python độc lập.

Tiêu chí hoàn thành:
- Có thể import package mà không cần khởi động chatbot.
- Inference sử dụng đúng preprocessing.
- Đầu vào không hợp lệ tạo lỗi rõ ràng, không âm thầm làm sai dữ liệu.
- Artifact và metadata được quản lý phiên bản đồng bộ.

### Giai đoạn 9 — Tích hợp chatbot

**Mục tiêu:** Tích hợp ML Core vào chatbot tư vấn triệu chứng giới hạn ở mức MVP.

Luồng đề xuất:

1. Nhận tin nhắn từ người dùng.
2. Kiểm tra triệu chứng nguy hiểm và tình huống cần xử lý khẩn cấp.
3. Xác định yêu cầu có nằm trong phạm vi được hỗ trợ hay không.
4. Dùng LLM trích xuất triệu chứng thành dữ liệu có cấu trúc.
5. Kiểm tra dữ liệu trích xuất và xác định thông tin thiếu hoặc mâu thuẫn.
6. Hỏi thêm khi cần thiết.
7. Gọi ML Core khi đầu vào đáp ứng yêu cầu.
8. Qua lớp quyết định để xem xét chất lượng đầu vào, độ bất định và nguy cơ dữ liệu ngoài phân phối.
9. Trả lời thận trọng hoặc từ chối đưa ra dự đoán.
10. Đưa ra hướng dẫn bước tiếp theo an toàn, không khẳng định chẩn đoán hoặc tự tạo chỉ định điều trị không có căn cứ.

Yêu cầu:
- Câu hỏi ngoài phạm vi phải được chuyển hướng lịch sự.
- Triệu chứng nguy hiểm phải kích hoạt hướng dẫn tìm trợ giúp y tế khẩn cấp, không bị xử lý như câu hỏi ngoài phạm vi thông thường.
- LLM không được tự bịa triệu chứng hoặc thông tin chưa được cung cấp.
- Không ép đưa ra dự đoán bệnh khi bằng chứng không đủ.
- Score của mô hình không mặc nhiên là xác suất hoặc độ tin cậy đã được hiệu chuẩn.
- Mục tiêu Accuracy 85% ban đầu chưa phải ngưỡng lâm sàng đã được phê duyệt.

Đầu ra:
- Thành phần kiểm soát phạm vi.
- Schema trích xuất dữ liệu.
- Điều phối backend.
- Logic an toàn và xử lý độ bất định.
- Giao diện người dùng tối thiểu.
- Kiểm thử tích hợp.

Tiêu chí hoàn thành:
- Chatbot gọi ML Core qua giao diện đã tài liệu hóa.
- Các luồng an toàn và từ chối dự đoán được kiểm thử.
- Chatbot không trình bày đầu ra mô hình như chẩn đoán xác định.

### Giai đoạn 10 — Kiểm thử hệ thống và hoàn thiện tài liệu

Công việc:
- Kiểm thử preprocessing, tiện ích huấn luyện và inference.
- Kiểm thử đầu vào sai, feature thiếu và thông tin mâu thuẫn.
- Kiểm thử từ chối ngoài phạm vi và tình huống triệu chứng nguy hiểm.
- Kiểm thử tích hợp chatbot với ML Core.
- Xác minh việc tải artifact và tương thích phiên bản.
- Tài liệu hóa cách cài đặt, chạy, huấn luyện, đánh giá và suy luận.
- Ghi lại giới hạn và các trường hợp mô hình có thể thất bại.
- Rà soát bảo mật và cách xử lý tin nhắn sức khỏe nhạy cảm.

Đầu ra:
- Báo cáo kiểm thử.
- Tài liệu cài đặt và sử dụng.
- Tài liệu kiến trúc cuối.
- Model card và ghi chú phiên bản phát hành.

Tiêu chí hoàn thành:
- Các kiểm thử quan trọng đều đạt.
- Giới hạn đã biết được trình bày rõ ràng.
- Có thể tái lập dự án theo hướng dẫn đã viết.

## 4. Cấu trúc repository và phân công vai trò

Dự án áp dụng quy trình phát triển rõ ràng từ nghiên cứu thực nghiệm sang đóng gói module sản phẩm:
- **Notebooks (
otebooks/):** Không gian tương tác trực quan dùng để thực hiện Data Audit, EDA (khảo sát phân phối, tương quan, trực quan hóa), thử nghiệm nhanh các thuật toán baseline, so sánh mô hình và phân tích lỗi (error analysis).
- **Modules .py (src/ml_core/):** Sau khi các giả thuyết và logic được kiểm chứng thực nghiệm ổn định trong notebook, tiến hành chuẩn hóa, tái cấu trúc và đưa các thành phần vào package ML Core độc lập, sạch, có type hints và docstrings.
- **Tests (	ests/):** Đảm bảo chất lượng bằng kiểm thử tự động với pytest: kiểm tra tính toàn vẹn của dữ liệu đầu vào (schema, null, range), pipeline tiền xử lý (chống rò rỉ dữ liệu), kết quả dự đoán (predict contract, output format) và các kịch bản ngoại lệ/dữ liệu lỗi.

\\	ext
Medical_chatbot/
├── data/
│   ├── raw/                  # Lưu trữ hoặc liên kết dữ liệu thô (được bảo vệ, không ghi đè)
│   └── processed/            # Dữ liệu sau tiền xử lý, phân chia train/val/test
├── docs/                     # Tài liệu thiết kế, báo cáo kiểm toán, experiment logs
├── notebooks/                # Nghiên cứu, Data Audit, EDA, thử nghiệm thuật toán
│   ├── 01_data_audit_eda.ipynb
│   ├── 02_model_experiments.ipynb
│   └── 03_error_analysis.ipynb
├── src/
│   └── ml_core/              # Package ML Core độc lập (Python modules)
│       ├── __init__.py
│       ├── data/             # DataLoader, data validation
│       ├── preprocessing/    # Pipeline làm sạch và xử lý đặc trưng
│       ├── models/           # Định nghĩa mô hình, baseline, training logic
│       ├── evaluation/       # Metrics (Top-k, Macro-F1, Calibration)
│       └── inference/        # Predictor interface, abstention rules
├── tests/                    # Kiểm thử tự động với pytest
│   ├── test_data_integrity.py   # Kiểm tra schema, missing, dữ liệu đầu vào
│   ├── test_preprocessing.py    # Kiểm tra biến đổi đặc trưng, leakage
│   ├── test_inference.py        # Kiểm tra contract dự đoán, output format
│   └── test_error_handling.py   # Kiểm tra các trường hợp ngoại lệ, edge cases
├── .gitignore                # Bảo vệ dữ liệu lớn, bỏ qua venv và cache
├── requirements.txt          # Khai báo dependencies chính
├── requirements.lock         # Cố định phiên bản tái lập
└── README.md
\
Lưu ý:
- Dữ liệu gốc phải được bảo toàn. Không đưa dữ liệu lên repository công khai nếu giấy phép hoặc trạng thái riêng tư không cho phép.
- Dữ liệu đã xử lý và artifact cần có chính sách lưu trữ, quản lý phiên bản rõ ràng.
- Không thêm lớp trừu tượng hoặc hạ tầng khi chưa có yêu cầu cụ thể.

## 5. Chính sách đánh giá và phê duyệt mô hình

Dự án phải phân biệt ba khái niệm:

1. **Đánh giá trên dataset:** metric tính trên tập validation hoặc test cụ thể.
2. **Score dự đoán:** score hoặc xác suất mà mô hình trả về cho một nhãn bệnh.
3. **Ngưỡng quyết định:** quy tắc đã được đánh giá để quyết định chấp nhận, trì hoãn hoặc từ chối đưa ra dự đoán.

Ba khái niệm này không thể thay thế cho nhau.

Mục tiêu Accuracy 85% ban đầu chỉ là giả thuyết phục vụ lập kế hoạch cho đến khi có kết quả baseline, phân phối lớp và đánh giá giới hạn dữ liệu. Tiêu chí chấp nhận cuối cùng cần được thống nhất trước khi chọn mô hình.

Mô hình huấn luyện trên dữ liệu tổng hợp không được xem là đã được xác nhận lâm sàng chỉ vì đạt điểm số benchmark cao.

## 6. Quy tắc làm việc của Coding Agent

Trước mỗi nhiệm vụ triển khai, Coding Agent phải:

1. Kiểm tra các file và dependencies liên quan.
2. Đọc yêu cầu và kiến trúc đã được phê duyệt.
3. Nêu kế hoạch thay đổi và các quyết định còn chưa rõ.
4. Chỉ triển khai trong phạm vi được giao.
5. Tái sử dụng mã nguồn hiện có khi phù hợp.
6. Giữ mã nguồn đơn giản, dễ bảo trì và kiểm thử.
7. Bổ sung xác thực dữ liệu và xử lý lỗi phù hợp.
8. Chạy các kiểm thử liên quan và báo cáo kết quả thực tế.
9. Không tuyên bố kiểm thử thành công nếu chưa thực sự chạy thành công.
10. Không tự ý thay đổi yêu cầu, giao diện công khai, schema dữ liệu, kiến trúc hoặc công nghệ đã được phê duyệt.

Sau mỗi nhiệm vụ, phải báo cáo:
- Các file được tạo hoặc sửa.
- Chức năng đã triển khai.
- Các kiểm thử đã chạy và kết quả thực tế.
- Các giả định đã sử dụng.
- Vấn đề còn tồn tại và công việc tiếp theo.
- Các quyết định cần người dùng phê duyệt.

## 7. Quy trình kiểm soát thay đổi

Các thay đổi sau bắt buộc phải được người dùng phê duyệt:
- Yêu cầu chức năng và use case.
- Thay dataset hoặc thay đổi target label.
- Schema dữ liệu và hợp đồng feature.
- Quy trình đánh giá và chia train/validation/test.
- API công khai của ML Core.
- Kiến trúc component hoặc kiến trúc triển khai.
- Công nghệ và cấu trúc cốt lõi của repository.
- Tiêu chí chọn và phê duyệt mô hình.

Agent được quyền đề xuất phương án thay thế nhưng không được triển khai các thay đổi này khi chưa được phê duyệt.

## 8. Nhiệm vụ tiếp theo

Bắt đầu với **Giai đoạn 0: Khảo sát repository**, sau đó chuyển sang **Giai đoạn 1: Kiểm toán dữ liệu**.

Không bắt đầu so sánh thuật toán, tối ưu, lựa chọn mô hình cuối hoặc tích hợp chatbot trước khi kết quả kiểm toán dữ liệu và chiến lược chia tập được xem xét.

Nhiệm vụ triển khai đầu tiên là khảo sát repository và tạo báo cáo. Không tái cấu trúc diện rộng và không sửa dữ liệu gốc trong nhiệm vụ này.
### Quy tắc xử lý quyết định kỹ thuật chưa được xác định

1. Không tự biến giả định, ví dụ minh họa hoặc phương án đề xuất thành yêu cầu chính thức.
2. Khi thiếu thông tin, trước tiên phải khảo sát mã nguồn, dữ liệu và tài liệu liên quan.
3. Nếu vẫn chưa đủ căn cứ, hãy lập danh sách vấn đề chưa xác định, giải thích tác động và đề xuất phương án.
4. Phân biệt rõ quyết định đã được phê duyệt, đề xuất của agent và vấn đề đang chờ quyết định.
5. Không triển khai các quyết định ảnh hưởng đến kiến trúc, hợp đồng API, schema dữ liệu, phương pháp đánh giá hoặc tiêu chí nghiệm thu khi chưa được phê duyệt.
6. Có thể triển khai các phần độc lập đã rõ yêu cầu, miễn là không tạo ra quyết định ngầm hoặc gây khó khăn cho việc thay đổi thiết kế sau này.
7. Trước khi bắt đầu một giai đoạn phụ thuộc vào quyết định chưa được thống nhất, phải trình người dùng xem xét và phê duyệt.