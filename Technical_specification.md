# TÀI LIỆU ĐẶC TẢ KỸ THUẬT (TECHNICAL SPECIFICATION)
## Dự án dự đoán bệnh dựa trên triệu chứng — Ưu tiên ML Core

**Trạng thái tài liệu:** Bản dự thảo chờ phê duyệt  
**Giai đoạn triển khai đầu tiên:** Khảo sát dữ liệu và xây dựng ML Core độc lập  
**Nguyên tắc ra quyết định:** Các quyết định đã được phê duyệt là yêu cầu bắt buộc. Những quyết định phụ thuộc dữ liệu phải được khảo sát, đề xuất và chờ người dùng phê duyệt trước khi triển khai.

---

## 1. Mục đích dự án

Xây dựng hệ thống dự đoán nhãn bệnh dựa trên các triệu chứng được biểu diễn dưới dạng dữ liệu có cấu trúc, sử dụng mô hình học máy để thực hiện phân loại.

Trong các giai đoạn sau, hệ thống có thể được mở rộng bằng giao diện hội thoại, cho phép người dùng mô tả triệu chứng bằng ngôn ngữ tự nhiên. Mô hình ngôn ngữ lớn (LLM) có thể hỗ trợ trích xuất thông tin triệu chứng, đặt câu hỏi bổ sung và trình bày kết quả dự đoán theo cách thận trọng, dễ hiểu.

Giai đoạn đầu tiên tập trung vào ML Core và quy trình từ dữ liệu đến mô hình. Chưa triển khai chatbot, giao diện người dùng hoặc tích hợp LLM trong giai đoạn này.

Dự án phục vụ mục đích học tập và thử nghiệm. Kết quả dự đoán không được trình bày như một chẩn đoán y khoa xác định. Hiệu suất trên dữ liệu tổng hợp không đồng nghĩa với việc mô hình đã được kiểm chứng lâm sàng.

## 2. Trạng thái quyết định và quyền phê duyệt

Mọi quyết định kỹ thuật phải được phân loại theo một trong ba nhóm sau.

### 2.1. Các quyết định đã được phê duyệt

Đây là những yêu cầu đã được người dùng xác nhận. Agent không được tự ý thay đổi nếu chưa có sự chấp thuận rõ ràng.

- Giai đoạn triển khai đầu tiên là xây dựng ML Core độc lập.
- Python là ngôn ngữ lập trình chính.
- Dự án sử dụng một môi trường Python ảo riêng.
- Phạm vi mô hình ban đầu là các thuật toán học máy cổ điển (Classical ML).
- scikit-learn là thư viện học máy chính.
- Sử dụng pip để cài đặt và quản lý các gói phụ thuộc.
- Sử dụng `requirements.txt` để khai báo các gói phụ thuộc của dự án.
- ML Core phải độc lập với chatbot, giao diện người dùng, nhà cung cấp LLM và framework điều phối agent.
- Quy trình phải bắt đầu bằng việc khảo sát repository và thực hiện Data Audit trước khi chốt các thiết kế phụ thuộc dữ liệu.
- Dữ liệu thô ban đầu phải được bảo toàn.
- Phải ngăn chặn rò rỉ dữ liệu (Data Leakage).
- Tập kiểm thử cuối cùng phải độc lập với quá trình lựa chọn mô hình và tinh chỉnh siêu tham số.
- Quy trình xử lý dữ liệu và thực nghiệm phải có khả năng tái lập.
- Những thay đổi ảnh hưởng đến yêu cầu chức năng, kiến trúc, schema dữ liệu, hợp đồng API công khai hoặc công nghệ đã được duyệt đều cần sự phê duyệt rõ ràng của người dùng.

### 2.2. Các nguyên tắc kỹ thuật bắt buộc

Những nguyên tắc sau phải được tuân thủ trong toàn bộ quá trình triển khai.

- Phân biệt rõ dữ kiện quan sát được, giả thuyết và quyết định đã được phê duyệt.
- Không tự tạo yêu cầu mới hoặc coi phương án đề xuất là quyết định đã được duyệt.
- Không ghi đè dữ liệu thô hoặc âm thầm loại bỏ bản ghi.
- Ghi lại các phép biến đổi dữ liệu và cấu hình thực nghiệm.
- Báo cáo các giả định, giới hạn, vấn đề chưa giải quyết và lỗi phát sinh.
- Phân biệt mã phục vụ khảo sát, thử nghiệm với mã triển khai có thể tái sử dụng.
- Không tuyên bố mô hình có giá trị lâm sàng chỉ dựa trên kết quả đánh giá một bộ dữ liệu.
- Không mặc định xem điểm số đầu ra của mô hình là xác suất đáng tin cậy trong thực tế nếu chưa đánh giá khả năng hiệu chuẩn.

### 2.3. Các quyết định phải chờ Data Audit và phê duyệt

Các nội dung dưới đây chưa được chốt. Agent không được tự quyết định trước khi có đủ bằng chứng và sự phê duyệt của người dùng.

- Schema đặc trưng (Feature Schema) và cách biểu diễn đặc trưng.
- Cách xử lý dữ liệu thiếu, không hợp lệ, trùng lặp hoặc có nhãn mâu thuẫn.
- Chất lượng nhãn và cách xử lý sự mất cân bằng giữa các lớp.
- Chiến lược chia tập huấn luyện, kiểm định và kiểm thử.
- Thiết kế mô hình baseline và danh sách thuật toán cần so sánh, ngoài phạm vi Classical ML đã được duyệt.
- Các chỉ số đánh giá, tiêu chí lựa chọn mô hình và ngưỡng chấp nhận.
- Phương pháp tiền xử lý đặc trưng, mã hóa hoặc chuẩn hóa dữ liệu.
- Không gian tìm kiếm siêu tham số và chiến lược tối ưu hóa.
- Hợp đồng đầu vào/đầu ra công khai của ML Core, bao gồm chữ ký chính xác của hàm `predict()`.
- Định dạng lưu mô hình, cơ chế quản lý phiên bản và giao diện suy luận.
- Quy tắc từ chối dự đoán (Abstention), phát hiện dữ liệu ngoài phân phối (Out-of-Distribution — OOD) và các ngưỡng bất định.
- Hợp đồng tích hợp với chatbot và backend trong tương lai.

Agent không được tự chốt các nội dung trên chỉ vì một phương án mặc định thuận tiện cho việc lập trình.

## 3. Công nghệ đã được phê duyệt

### 3.1. Công nghệ nền tảng

| Hạng mục | Công nghệ hoặc yêu cầu |
|---|---|
| Ngôn ngữ lập trình | Python |
| Cô lập môi trường | Môi trường Python ảo riêng |
| Phạm vi mô hình ban đầu | Classical ML |
| Thư viện học máy | scikit-learn |
| Cài đặt gói phụ thuộc | pip |
| Khai báo gói phụ thuộc | `requirements.txt` |
| Quản lý phiên bản mã nguồn | Git |

Agent phải kiểm tra repository và môi trường Python hiện tại trước khi tạo hoặc chỉnh sửa các tệp cấu hình môi trường.

Không cài đặt thư viện trực tiếp vào môi trường Python toàn cục khi dự án cần sử dụng môi trường ảo riêng.

Phiên bản Python và phiên bản cụ thể của các thư viện phải được lựa chọn sau khi kiểm tra môi trường và khả năng tương thích. Sau khi xác minh, các phiên bản phụ thuộc phải được ghi lại trong `requirements.txt` để hỗ trợ tái lập môi trường.

### 3.2. Thư viện và công cụ hỗ trợ

Các thư viện như pandas, NumPy, pytest và Jupyter Notebook có thể được đề xuất để phục vụ phân tích dữ liệu, tính toán số học, kiểm thử tự động và thử nghiệm khám phá.

Trừ khi được phê duyệt riêng, các công cụ hỗ trợ này được xem là phương án triển khai đề xuất, chưa phải yêu cầu công nghệ bắt buộc đã được chốt.

Mỗi thư viện bổ sung phải có mục đích rõ ràng và lý do sử dụng. Agent cần tránh đưa vào các dependency không cần thiết.

Các framework học sâu như PyTorch hoặc TensorFlow nằm ngoài phạm vi mô hình ban đầu đã được duyệt. Không được tự ý đưa chúng vào dự án. Nếu có bằng chứng cho thấy Classical ML không đáp ứng yêu cầu, agent phải trình bày lý do và chờ người dùng phê duyệt trước khi mở rộng phạm vi.

## 4. Phạm vi và thứ tự triển khai

### Giai đoạn 0 — Khảo sát repository và môi trường

Kiểm tra trạng thái hiện tại của dự án trước khi thực hiện bất kỳ thay đổi nào.

Các nội dung cần khảo sát:

- Danh sách tệp, thư mục và cấu trúc repository.
- Trạng thái Git và các thay đổi hiện có.
- Các phiên bản Python đã cài đặt.
- Môi trường ảo và các tệp khai báo dependency hiện có.
- Vị trí các tệp dữ liệu.
- Tài liệu, script, notebook và model artifact hiện có.

Không xóa, ghi đè hoặc tái cấu trúc các tài nguyên hiện có nếu chưa có lý do rõ ràng và sự phê duyệt phù hợp.

**Sản phẩm bàn giao:** báo cáo khảo sát repository, môi trường, rủi ro và các bước tiếp theo được đề xuất.

### Giai đoạn 1 — Kiểm toán dữ liệu (Data Audit)

Khảo sát bộ dữ liệu thực tế trước khi quyết định cách chuẩn bị dữ liệu.

Các nội dung cần xem xét, tùy theo cấu trúc dữ liệu:

- Định dạng tệp, dung lượng, số dòng, số cột và ý nghĩa các cột.
- Nhãn mục tiêu và các cột có thể là đặc trưng đầu vào.
- Kiểu dữ liệu và miền giá trị hợp lệ.
- Dữ liệu thiếu, sai định dạng hoặc chứa giá trị bất thường.
- Bản ghi trùng lặp và các trường hợp có nhãn mâu thuẫn.
- Tần suất xuất hiện của từng nhãn và mức độ mất cân bằng lớp.
- Bản ghi trùng hoặc gần trùng có thể gây rò rỉ dữ liệu.
- Các đặc trưng không liên quan, dư thừa hoặc đáng nghi.
- Tài liệu mô tả bộ dữ liệu, nguồn gốc và giấy phép sử dụng.
- Bộ dữ liệu có phải dữ liệu tổng hợp hay không và điều đó ảnh hưởng thế nào đến việc đánh giá mô hình.

Báo cáo phải dựa trên dữ liệu thực tế đã kiểm tra. Không suy đoán số liệu từ ảnh chụp màn hình hoặc thông tin mô tả trước đây khi có thể kiểm tra trực tiếp tệp dữ liệu.

**Sản phẩm bàn giao:** báo cáo Data Audit, bao gồm các phát hiện, câu hỏi chưa được giải quyết, rủi ro và các phương án đề xuất.

Agent có thể thực hiện khảo sát không phá hủy trước khi xin phê duyệt. Tuy nhiên, các quy tắc xử lý dữ liệu có ảnh hưởng đáng kể, định nghĩa đặc trưng và chiến lược chia tập dữ liệu phải được đề xuất để người dùng xem xét trước khi triển khai.

### Giai đoạn 2 — Phê duyệt thiết kế dữ liệu

Sau Data Audit, agent phải trình bày các quyết định dự kiến liên quan đến:

- Schema và cách biểu diễn đặc trưng.
- Định nghĩa nhãn mục tiêu.
- Chính sách xử lý chất lượng dữ liệu và bản ghi trùng.
- Dữ liệu thiếu và không hợp lệ.
- Mất cân bằng lớp và các lớp hiếm.
- Phương pháp chia tập dữ liệu và phòng chống leakage.
- Thiết kế đánh giá mô hình.

Với mỗi quyết định quan trọng, cần giải thích các phương án khả thi, ưu điểm, rủi ro và ảnh hưởng dự kiến đến kết quả thực nghiệm.

Agent phải chờ người dùng phê duyệt rõ ràng trước khi triển khai chính sách được đề xuất.

Mọi quyết định đã được duyệt cùng lý do lựa chọn phải được ghi lại trong tài liệu phù hợp.

### Giai đoạn 3 — Chuẩn bị dữ liệu và xây dựng baseline

Chỉ thực hiện sau khi các quyết định thiết kế dữ liệu liên quan đã được phê duyệt.

- Triển khai các quy tắc chuẩn bị dữ liệu đã được duyệt.
- Xây dựng quy trình chia tập và tiền xử lý theo thiết kế được duyệt.
- Thiết lập mô hình baseline theo giao thức đánh giá đã thống nhất.
- Ghi lại phiên bản dữ liệu, cấu hình và kết quả thực nghiệm.

Không tự mặc định tỷ lệ chia tập, phương pháp tiền xử lý, thuật toán baseline hoặc schema đặc trưng trước khi có phê duyệt.

Các phép biến đổi có học tham số từ dữ liệu phải được fit trên tập dữ liệu được phép dùng để huấn luyện. Dữ liệu kiểm định và kiểm thử không được tác động không phù hợp đến quá trình fit hoặc lựa chọn mô hình.

### Giai đoạn 4 — Thực nghiệm Classical ML

So sánh các thuật toán Classical ML phù hợp bằng giao thức thực nghiệm đã được duyệt.

Các thuật toán có thể được xem xét gồm Logistic Regression, Decision Tree, Random Forest, Extra Trees, SVM hoặc những bộ phân loại scikit-learn khác có lý do phù hợp.

Danh sách trên chỉ mang tính tham khảo, không phải danh sách bắt buộc phải triển khai toàn bộ.

Với mỗi thực nghiệm, cần ghi lại:

- Mô hình và cấu hình.
- Phiên bản dữ liệu và đặc trưng.
- Phương pháp tiền xử lý và kiểm định.
- Các chỉ số đánh giá.
- Chi phí tính toán và các trường hợp mô hình dự đoán sai đáng chú ý.
- Giới hạn và cách diễn giải kết quả.

Không lựa chọn mô hình cuối cùng chỉ dựa trên Accuracy cao nhất.

### Giai đoạn 5 — Lựa chọn mô hình và đánh giá cuối cùng

Sau khi kế hoạch đánh giá được phê duyệt:

- So sánh mô hình theo các chỉ số và tiêu chí đã thống nhất.
- Kiểm tra hiệu suất trên từng lớp và các trường hợp dự đoán sai quan trọng.
- Đánh giá mô hình có đáp ứng tiêu chí chấp nhận đã duyệt hay không.
- Duy trì tính độc lập của tập kiểm thử cuối cùng.
- Báo cáo giới hạn, bao gồm tính tổng hợp của dữ liệu nếu phù hợp.

Không có ngưỡng hiệu suất, tuyên bố lâm sàng hoặc quy tắc lựa chọn mô hình cuối cùng nào được coi là đã duyệt nếu chưa được ghi nhận rõ ràng.

### Giai đoạn 6 — Xây dựng ML Core độc lập

Đóng gói quy trình đã được phê duyệt và kiểm chứng thành mã Python có thể tái sử dụng.

ML Core cần phân tách các trách nhiệm logic như:

- Truy cập và kiểm tra dữ liệu.
- Chuẩn bị dữ liệu theo chính sách đã duyệt.
- Biến đổi đặc trưng.
- Huấn luyện và thực nghiệm.
- Đánh giá và lựa chọn mô hình.
- Quản lý model artifact.
- Suy luận.
- Kiểm thử và tài liệu.

Đây là các trách nhiệm logic, không phải yêu cầu tạo một module hoặc class riêng cho từng mục. Cấu trúc package thực tế cần đơn giản và dễ bảo trì.

Tên module, giao diện công khai, chữ ký hàm `predict()` và schema đầu vào/đầu ra phải được đề xuất và phê duyệt khi đã có đủ bằng chứng thiết kế.

ML Core không được phụ thuộc vào kiểu dữ liệu riêng của chatbot, nhà cung cấp LLM hoặc framework điều phối agent.

### Giai đoạn 7 — Chuẩn bị tích hợp

Chỉ sau khi hợp đồng và hành vi của ML Core đã được phê duyệt, dự án mới xác định cách chatbot hoặc backend gọi ML Core.

Việc tích hợp không được khiến ML Core phải thiết kế lại chỉ để phụ thuộc vào một LLM hoặc framework web cụ thể.

Chatbot và frontend là các giai đoạn sau, không thuộc phạm vi triển khai ban đầu của Data Audit.

## 5. Tính toàn vẹn dữ liệu và mô hình

### 5.1. Bảo toàn dữ liệu thô

- Xem dữ liệu gốc là đầu vào bất biến.
- Lưu dữ liệu đã biến đổi riêng biệt với dữ liệu thô.
- Ghi lại các phép biến đổi và vị trí lưu đầu ra.
- Không âm thầm loại bỏ bản ghi hoặc thay đổi nhãn.

### 5.2. Ngăn chặn rò rỉ dữ liệu

- Khảo sát bản ghi trùng và gần trùng trước khi chia tập.
- Chỉ fit các phép biến đổi học tham số trên dữ liệu huấn luyện được phép sử dụng.
- Giữ dữ liệu kiểm định tách biệt khỏi quá trình fit khi giao thức yêu cầu.
- Không sử dụng kết quả test cuối cùng để lựa chọn đặc trưng, tinh chỉnh siêu tham số hoặc lựa chọn mô hình.
- Nếu dữ liệu có cấu trúc nhóm hoặc sự phụ thuộc đặc biệt, phải khảo sát xem chia ngẫu nhiên thông thường có phù hợp không trước khi chọn phương pháp.

Chính sách chia tập chính thức chỉ được chốt sau Data Audit và phê duyệt của người dùng.

### 5.3. Khả năng tái lập

Ghi lại đủ thông tin để tái lập thực nghiệm, bao gồm random seed liên quan, cấu hình, phiên bản dữ liệu, phương pháp tiền xử lý và phiên bản dependency.

Không tuyên bố khả năng tái lập tuyệt đối nếu môi trường thực thi hoặc thư viện sử dụng không bảo đảm điều đó.

### 5.4. An toàn y tế và giới hạn mô hình

Đây là nguyên mẫu phục vụ học tập và dự đoán thử nghiệm, không phải hệ thống chẩn đoán đã được kiểm chứng lâm sàng.

- Nhãn được dự đoán không phải chẩn đoán xác định.
- Điểm số của mô hình không mặc nhiên là xác suất đã được hiệu chuẩn.
- Không ép dự đoán một nhãn bệnh với độ chắc chắn cao khi đầu vào không thuộc phạm vi dữ liệu hoặc nằm ngoài phân phối.
- Đầu vào không đầy đủ hoặc kém chất lượng có thể cần hỏi thêm hoặc từ chối dự đoán.
- Các triệu chứng khẩn cấp hoặc có khả năng đe dọa tính mạng không được xử lý như một bài toán phân loại thông thường.
- Không tự tạo đơn thuốc hoặc kế hoạch điều trị không có căn cứ.
- Đánh giá trên dữ liệu tổng hợp không chứng minh mô hình hoạt động tốt trên bệnh nhân thực tế.

Logic từ chối dự đoán, các ngưỡng và cách tích hợp an toàn phải được thiết kế, phê duyệt trước khi triển khai.

## 6. Chính sách cấu trúc repository và môi trường

Cấu trúc dưới đây là định hướng đề xuất, không phải yêu cầu phải tạo tất cả thư mục ngay lập tức:

- `docs/` — tài liệu đặc tả, báo cáo kiểm toán dữ liệu và quyết định đã duyệt.
- `data/raw/` — dữ liệu gốc, được bảo toàn và không chỉnh sửa trực tiếp.
- `data/processed/` — dữ liệu dẫn xuất đã qua xử lý, nếu cần.
- `ml_core/` — package ML Core có thể tái sử dụng.
- `experiments/` — script hoặc notebook phục vụ thực nghiệm.
- `tests/` — kiểm thử tự động.
- `requirements.txt` — khai báo dependency của dự án.
- `.venv/` — thư mục môi trường Python ảo, cần được loại trừ khỏi Git.

Agent phải kiểm tra repository trước và đề xuất các thay đổi cấu trúc cần thiết. Không áp đặt cấu trúc này lên repository hiện có khi chưa được xem xét.

Không commit môi trường ảo lên Git. Việc đưa dữ liệu, model artifact và đầu ra thực nghiệm vào Git phải được cân nhắc theo dung lượng, giấy phép, tính nhạy cảm và yêu cầu tái lập. Không tự động commit mọi tệp được tạo ra.

## 7. Quy trình kiểm soát thay đổi

Cần có sự phê duyệt rõ ràng của người dùng trước khi thay đổi:

- Yêu cầu chức năng hoặc phạm vi đã được duyệt.
- Kiến trúc lõi hoặc ranh giới giữa các thành phần.
- Schema dữ liệu hoặc đặc trưng.
- Chính sách làm sạch và tiền xử lý có ảnh hưởng đến kết quả thực nghiệm.
- Giao thức chia tập và đánh giá.
- Hợp đồng đầu vào/đầu ra công khai của ML Core.
- Công nghệ nền tảng đã được duyệt.
- Tiêu chí lựa chọn mô hình hoặc ngưỡng chấp nhận.
- Thứ tự ưu tiên triển khai ban đầu.

Với quyết định chưa được giải quyết, agent phải khảo sát, trình bày các lựa chọn và đánh đổi, đưa ra khuyến nghị, sau đó chờ phê duyệt.

Agent có thể thực hiện các hoạt động khảo sát không phá hủy mà không cần xin phép trước. Tuy nhiên, không được biến một đề xuất chưa được duyệt thành quyết định triển khai chính thức.

## 8. Tiêu chí hoàn thành nhiệm vụ đầu tiên

Nhiệm vụ ban đầu được xem là hoàn thành khi:

- Repository và môi trường đã được khảo sát.
- Các tệp dữ liệu hiện có đã được xác định.
- Trạng thái dự án và rủi ro tiềm ẩn đã được báo cáo.
- Quy trình Data Audit đã được đề xuất.
- Chưa triển khai quyết định thiết kế dữ liệu hoặc kiến trúc chưa được duyệt.
- Chưa huấn luyện mô hình khi chưa đến giai đoạn phù hợp.
- Các quyết định tiếp theo cần người dùng phê duyệt đã được liệt kê rõ ràng.

Agent phải gửi báo cáo hoàn thành bao gồm:

1. Các tệp đã kiểm tra hoặc thay đổi.
2. Các phát hiện có bằng chứng.
3. Các lệnh hoặc bước kiểm tra đã thực hiện.
4. Vấn đề và câu hỏi chưa giải quyết.
5. Các bước tiếp theo được đề xuất.
6. Những quyết định cần người dùng phê duyệt.

**Hành động đầu tiên:** khảo sát repository và các tệp dữ liệu hiện có, sau đó báo cáo kết quả. Không lập tức tái cấu trúc, làm sạch dữ liệu, chốt schema đặc trưng hoặc huấn luyện mô hình.