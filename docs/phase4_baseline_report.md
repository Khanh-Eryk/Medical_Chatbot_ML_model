# BÁO CÁO GIAI ĐOẠN 4 — BASELINE 377 FEATURE RAW

## Liên kết dữ liệu

Mỗi phiên chỉ đọc `train.csv` và `val.csv` thực tế do Phase 2 tạo ra. Không dùng manifest, JSON result, notebook output hoặc metadata làm input huấn luyện. Phase 3 chỉ xác thực cột và chuyển cùng giá trị nhị phân sang CSR sparse.

## Cấu hình

- Model: `MultinomialNB(alpha=1.0)`.
- Feature: đầy đủ 377 cột raw, không feature selection, không giảm chiều.
- Tập đánh giá: validation; test set không được đọc.

## Kết quả validation

| Experiment | Train | Validation | Top-1 | Macro-F1 | Weighted-F1 | Top-3 | Top-5 | Fit (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exp1_full | 173,219 | 28,432 | 82.50% | 66.46% | 81.96% | 94.30% | 96.98% | 4.320 |
| exp2_dedup | 132,783 | 28,432 | 80.42% | 55.40% | 79.26% | 93.09% | 96.04% | 3.295 |
| exp3_capped | 166,308 | 28,432 | 82.26% | 63.62% | 81.61% | 94.12% | 96.85% | 4.483 |

## Phân tích lỗi

### exp1_full

Các nhầm lẫn có tần suất cao nhất trên validation:

| Nhãn thực | Dự đoán | Số mẫu |
|---|---|---:|
| infectious gastroenteritis | noninfectious gastroenteritis | 52 |
| noninfectious gastroenteritis | infectious gastroenteritis | 51 |
| temporary or benign blood in urine | cystitis | 46 |
| cataract | macular degeneration | 37 |
| bladder disorder | cystitis | 36 |
| schizophrenia | psychotic disorder | 32 |
| skin pigmentation disorder | skin polyp | 30 |
| spontaneous abortion | vaginal cyst | 28 |
| pyelonephritis | kidney stone | 27 |
| seborrheic keratosis | skin polyp | 26 |

### exp2_dedup

Các nhầm lẫn có tần suất cao nhất trên validation:

| Nhãn thực | Dự đoán | Số mẫu |
|---|---|---:|
| temporary or benign blood in urine | cystitis | 54 |
| corneal disorder | dry eye of unknown cause | 53 |
| infectious gastroenteritis | noninfectious gastroenteritis | 53 |
| noninfectious gastroenteritis | infectious gastroenteritis | 53 |
| personality disorder | marijuana abuse | 46 |
| bladder disorder | cystitis | 44 |
| cataract | macular degeneration | 38 |
| gallstone | cholecystitis | 37 |
| tooth disorder | gum disease | 36 |
| conjunctivitis due to virus | conjunctivitis | 33 |

### exp3_capped

Các nhầm lẫn có tần suất cao nhất trên validation:

| Nhãn thực | Dự đoán | Số mẫu |
|---|---|---:|
| infectious gastroenteritis | noninfectious gastroenteritis | 52 |
| noninfectious gastroenteritis | infectious gastroenteritis | 51 |
| temporary or benign blood in urine | cystitis | 49 |
| bladder disorder | cystitis | 37 |
| cataract | macular degeneration | 37 |
| schizophrenia | psychotic disorder | 32 |
| skin pigmentation disorder | skin polyp | 30 |
| spontaneous abortion | vaginal cyst | 28 |
| pyelonephritis | kidney stone | 27 |
| diabetic retinopathy | macular degeneration | 26 |

## Giới hạn

Baseline không phải đánh giá lâm sàng, không chọn model cuối và không sử dụng test set. Các bước feature selection, tuning và Phase 5 vẫn chưa được thực hiện.
