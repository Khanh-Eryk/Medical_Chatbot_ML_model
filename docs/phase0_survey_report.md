# BÁO CÁO KHẢO SÁT REPOSITORY VÀ MÔI TRƯỜNG (PHASE 0)

**Dự án:** Dự đoán bệnh dựa trên triệu chứng — ML Core  
**Thời gian thực hiện:** 29/10/2026  
**Trạng thái:** Hoàn thành khảo sát — Chờ người dùng phê duyệt đề xuất thiết lập môi trường và cấu trúc thư mục  

---

## 1. Hiện trạng Repository và Phiên bản mã nguồn

- **Cấu trúc hiện tại của thư mục gốc:**
  - `.git/`: Kho lưu trữ Git.
  - `Disease and symptoms dataset.csv` (~181.95 MB, 190,786,867 bytes).
  - `IMPLEMENTATION_PLAN.md` (23,960 bytes).
  - `Technical_document.md` (1,342 bytes).
  - `Technical_specification.md` (22,727 bytes).
  - `docs/`: Thư mục tài liệu (mới tạo để chứa báo cáo).
- **Trạng thái Git:**
  - Branch: `main`
  - Remote: `origin https://github.com/Khanh-Eryk/Medical_Chatbot_ML_model.git`
  - Lịch sử commit: Commit gần nhất `fc9cdeb Add documents`
  - Tệp chưa theo dõi (untracked): `Disease and symptoms dataset.csv` (Kích thước ~182 MB, vượt giới hạn 100 MB mặc định của GitHub nếu commit trực tiếp mà không dùng Git LFS).
  - Chưa có tệp `.gitignore`.

---

## 2. Khảo sát dữ liệu gốc ban đầu

- **Tệp dữ liệu:** `Disease and symptoms dataset.csv` nằm ở thư mục gốc.
- **Kích thước:** 190,786,867 bytes (~182 MB).
- **Cấu trúc sơ bộ (đọc header và 2 dòng đầu):**
  - Cột đầu tiên: `diseases` (nhãn mục tiêu / target label).
  - Các cột tiếp theo: Các triệu chứng dạng nhị phân/số (ví dụ: `anxiety and nervousness`, `depression`, `shortness of breath`, ..., giá trị quan sát ban đầu là 0 và 1).
  - Dòng header có chứa dấu ngoặc kép phân cách cho các triệu chứng có dấu phẩy (ví dụ: `"muscle cramps, contractures, or spasms"`).
  - Dữ liệu gốc hoàn toàn được bảo toàn nguyên vẹn, không chỉnh sửa hoặc di chuyển.

---

## 3. Khảo sát Môi trường Python

Qua rà soát hệ thống Windows, các phiên bản Python khả dụng gồm:
1. `C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe` (Python 3.12.9, pip 24.3.1) — **Bản chính thức tiêu chuẩn Windows**.
2. `C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe` (Python 3.10.8, pip 22.2.2).
3. `C:\msys64\mingw64\bin\python.exe` (Python 3.11.10 - MSYS2 Mingw64, không ưu tiên dùng cho Python package Windows chuẩn).

- **Hiện trạng venv:** Hiện tại trong repo chưa có môi trường ảo nào (`.venv` chưa được tạo).
- **Hiện trạng dependency:** Chưa có tệp `requirements.txt` hoặc `pyproject.toml`.

---

## 4. Các đề xuất kỹ thuật cần Người dùng phê duyệt

Để chuẩn bị chuyển sang **Giai đoạn 1 (Data Audit)** theo đúng các nguyên tắc trong `Technical_specification.md` và `IMPLEMENTATION_PLAN.md`, đề xuất các hạng mục sau:

### 4.1. Đề xuất phiên bản Python và tạo Môi trường ảo (`.venv`)
- **Phiên bản đề xuất:** Sử dụng **Python 3.12.9** (từ `AppData\Local\Programs\Python\Python312`) để tạo môi trường ảo tại thư mục `.venv` trong dự án:
  ```powershell
  & "C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe" -m venv .venv
  ```
- **Lý do:** Python 3.12 là phiên bản ổn định, hiện đại, hỗ trợ đầy đủ các bản build wheel mới nhất của `numpy`, `scikit-learn`, `pandas`, `pytest`.

### 4.2. Đề xuất danh sách dependencies ban đầu (`requirements.txt`)
Dự án đã duyệt Classical ML + `scikit-learn`. Đề xuất cài đặt tối thiểu cho Giai đoạn 1 (Data Audit & EDA):
- `numpy`
- `pandas` (hỗ trợ đọc, phân tích dữ liệu dạng bảng lớn ~182MB)
- `scikit-learn` (thư viện học máy chính)
- `pytest` (hỗ trợ unit test)

### 4.3. Đề xuất cấu hình `.gitignore`
Tạo `.gitignore` để tránh commit các file không cần thiết hoặc file dữ liệu vượt quá giới hạn của GitHub:
```gitignore
# Môi trường ảo
.venv/
venv/
__pycache__/
*.pyc

# Dữ liệu lớn / Data artifacts
# Lưu ý: Không commit trực tiếp file ~182MB lên GitHub repo thông thường
Disease and symptoms dataset.csv
data/raw/
data/processed/

# Thư mục build, output, cache
models/
experiments/
.pytest_cache/
```

### 4.4. Đề xuất cấu trúc thư mục dự án chuẩn (theo mục 4 của IMPLEMENTATION_PLAN)
```text
Medical_chatbot/
├── data/
│   ├── raw/                  # Lưu dữ liệu gốc hoặc liên kết dữ liệu
│   └── processed/            # Lưu các tập dữ liệu sau tiền xử lý (ở Phase 2)
├── docs/                     # Báo cáo kỹ thuật, Data Audit, Experiment logs
│   └── phase0_survey_report.md
├── src/
│   └── ml_core/              # Package ML Core độc lập
├── tests/                    # Kiểm thử đơn vị và tích hợp
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 5. Kết luận Giai đoạn 0

Giai đoạn 0 đã hoàn thành việc khảo sát hiện trạng:
- Dữ liệu gốc được bảo toàn.
- Không có thay đổi ngoài phạm vi.
- Báo cáo đã ghi nhận đầy đủ hiện trạng.
- Sẵn sàng khởi tạo môi trường và cấu trúc sau khi được người dùng phê duyệt các đề xuất tại Mục 4.

