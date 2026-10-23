### Quy trình thiết kế và phê duyệt các quyết định phụ thuộc dữ liệu

Các quyết định liên quan đến feature schema, xử lý dữ liệu thiếu hoặc không hợp lệ, chính sách xử lý mẫu trùng lặp và chiến lược chia tập dữ liệu không được mặc định trước khi hoàn thành Data Audit.

Sau Data Audit, Coding Agent phải:
1. Trình bày kết quả kiểm toán và các vấn đề phát hiện được.
2. Đề xuất các phương án xử lý có căn cứ từ dữ liệu thực tế.
3. Phân tích ưu điểm, nhược điểm và ảnh hưởng của từng phương án đến quá trình huấn luyện và đánh giá.
4. Chờ người dùng xem xét và phê duyệt các quyết định cần thiết.
5. Chỉ triển khai các thay đổi dữ liệu và pipeline sau khi thiết kế tương ứng được phê duyệt.

Các nguyên tắc chống data leakage, bảo toàn dữ liệu gốc, khả năng tái lập và tách biệt tập test phải được tuân thủ xuyên suốt quá trình nghiên cứu.

Mọi quyết định đã được phê duyệt phải được ghi nhận trong tài liệu thiết kế tương ứng để bảo đảm tính nhất quán giữa dữ liệu, huấn luyện, đánh giá và suy luận.