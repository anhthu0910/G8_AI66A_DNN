# Bảng so sánh các mô hình

Tài liệu này lưu cấu trúc và kết quả của từng lần chạy để so sánh các thay đổi kiến trúc. Baseline Model 1 lấy từ `src/Model_1/Model1_Simple_CNN.ipynb`; metric lấy từ `src/Model_1/outputs/model1_metrics.json`.

 --- 

# Thu: Model 1 - Simple CNN 

## Cau truc 1

### Cấu trúc mạng

Đầu vào là ảnh RGB kích thước `128 × 128`. Các convolution dùng kernel `3 × 3`, padding `1`, giữ nguyên chiều rộng/cao trước khi MaxPool giảm một nửa.

| Tầng | Cấu hình | Kích thước đầu ra | Số tham số |
|---|---|---:|---:|
| Input | RGB image | `3 × 128 × 128` | 0 |
| Conv block 1 | Conv2d `3 → 32`, kernel `3 × 3`, padding 1 → ReLU → MaxPool `2 × 2` | `32 × 64 × 64` | 896 |
| Conv block 2 | Conv2d `32 → 64`, kernel `3 × 3`, padding 1 → ReLU → MaxPool `2 × 2` | `64 × 32 × 32` | 18,496 |
| Conv block 3 | Conv2d `64 → 128`, kernel `3 × 3`, padding 1 → ReLU → MaxPool `2 × 2` | `128 × 16 × 16` | 73,856 |
| Flatten | `128 × 16 × 16` → vector | `32,768` | 0 |
| Fully connected 1 | Linear `32,768 → 256` → ReLU → Dropout `0.5` | `256` | 8,388,864 |
| Classifier | Linear `256 → 4` (CNV, DME, DRUSEN, NORMAL) | `4` logits | 1,028 |
| **Tổng** | 3 convolution blocks, 2 fully connected layers; không BatchNorm/skip connection |  | **8,483,140** |

### Cấu hình thực nghiệm

| Thuộc tính | Giá trị |
|---|---|
| Dataset | `data/cut` nếu tồn tại, nếu không thì `data/raw` |
| Tiền xử lý | Resize `128 × 128`; normalize theo mean/std tính từ train |
| Augmentation train | Horizontal flip, rotation `±15°`, affine translate/scale, ColorJitter |
| Batch size | 64 |
| Optimizer | Adam, learning rate `0.001`, weight decay `0.0001` |
| Loss | CrossEntropyLoss; class weights được bật có điều kiện khi tỉ lệ mất cân bằng > 1.5 |
| LR scheduler | ReduceLROnPlateau, factor `0.5`, patience `2` |
| Epoch | Tối đa 30; lần chạy ghi nhận đủ 30 epoch |
| Chọn checkpoint | Validation loss thấp nhất; best checkpoint được lưu tại `src/Model_1/outputs/model1_simple_cnn.pt` |

### Kết quả lần chạy hiện tại

| Metric | Kết quả |
|---|---:|
| Số tham số | 8,483,140 |
| Best validation loss | 0.1673 |
| Test accuracy | 0.9441 (94.41%) |
| Test macro precision | 0.8960 |
| Test macro recall | 0.9182 |
| Test macro-F1 | 0.9062 |
| Test weighted-F1 | 0.9449 |
| Thời gian huấn luyện | 38.77 phút |

Các metric được cập nhật từ lần chạy mới nhất và lưu tại `src/Model_1/outputs/model1_metrics.json`. Lịch sử train/validation nằm ở `src/Model_1/outputs/model1_history.csv`.

## Bảng so sánh tổng quan

| Model | Kiến trúc / thay đổi chính | Tham số | Best val loss | Test accuracy | Macro-F1 | Weighted-F1 | Thời gian (phút) | Ghi chú |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Cấu trúc 1 | 3 Conv blocks (32/64/128), FC 256, Dropout 0.5 | 8,483,140 | 0.1673 | 0.9441 | 0.9062 | 0.9449 | 38.77 | 128 × 128, 30 epoch |
| Cấu trúc 2 | Điền sau khi chạy |  |  |  |  |  |  |  |
| Cấu trúc 3 | Điền sau khi chạy |  |  |  |  |  |  |  |

Để so sánh công bằng, giữ cố định train/val/test split, preprocessing, seed và cách chọn checkpoint; chỉ thay đổi kiến trúc hoặc yếu tố thực nghiệm cần đánh giá. Ghi rõ mọi thay đổi cấu hình trong cột ghi chú.

## Cau truc 2

## Cau truc 3

---

# Lan: Model 2 - Complex CNN

---

# Hien: Model 3 - Transfer Learning / Fine-tuning