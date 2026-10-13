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
| Model 2 - ResNetSmall | Residual blocks, BatchNorm, skip connections; class weights | 2,796,068 | 0.2759 | **0.9716** | **0.9522** | **0.9721** | 84.8 | `raw`, 128 × 128, 40 epoch; test 10,848 ảnh |
| Model 2 - Complex CNN balanced | CNN tuần tự 4 block; class weights | 423,268 | 0.1943 | 0.9293 | 0.8905 | 0.9312 | Chưa lưu | `raw`, 224 × 224, 30 epoch; test 10,933 ảnh |
| Model 2 - Parallel CNN balanced | Nhánh Conv song song ghép bằng Concatenate; class weights | 453,092 | 0.2472 | 0.9122 | 0.8568 | 0.9103 | Chưa lưu | `raw`, 224 × 224, 13 epoch; test 10,933 ảnh |
| Model 2 - Residual CNN balanced | Residual blocks, BatchNorm, skip connections; class weights | 1,253,572 | 0.1371 | 0.9535 | 0.9251 | 0.9546 | 856.28 | `raw`, 224 × 224, 40 epoch; test 10,933 ảnh |
| Model 3 | Điền sau khi chạy |  |  |  |  |  |  |  |

Để so sánh công bằng, giữ cố định train/val/test split, preprocessing, seed và cách chọn checkpoint; chỉ thay đổi kiến trúc hoặc yếu tố thực nghiệm cần đánh giá. Ghi rõ mọi thay đổi cấu hình trong cột ghi chú.

## Cau truc 2

## Cau truc 3

---

# Lan: Model 2 - Các kiến trúc CNN

## Kết quả bốn lần chạy trên data/raw

| File đã chạy | Test accuracy | Macro precision | Macro recall | Macro-F1 | Weighted-F1 | Best val loss | Ghi chú |
|---|---:|---:|---:|---:|---:|---:|---|
| `Model2_ResNet.ipynb` | 0.9716 | 0.9391 | 0.9674 | 0.9522 | 0.9721 | 0.2759 | 84.8 phút; 128 × 128; test 10,848 ảnh |
| `complex_cnn_balanced.ipynb` | 0.9293 | 0.8725 | 0.9170 | 0.8905 | 0.9312 | 0.1943 | 224 × 224; 30 epoch; thời gian không được lưu |
| `parallel_cnn_balanced.ipynb` | 0.9122 | 0.8778 | 0.8393 | 0.8568 | 0.9103 | 0.2472 | 224 × 224; 13 epoch; thời gian không được lưu |
| `residual_cnn_balanced.ipynb` | 0.9535 | 0.9097 | 0.9443 | 0.9251 | 0.9546 | 0.1371 | 856.28 phút; 224 × 224; test 10,933 ảnh |

Các kết quả cho ba notebook Keras được đọc từ báo cáo phân loại và confusion matrix đã lưu trong notebook; các metric được tính lại từ số đếm trong confusion matrix và làm tròn 4 chữ số. Best validation loss lấy từ log epoch. Thời gian huấn luyện của Complex CNN và Parallel CNN không còn trong artifacts đã lưu, nên không ước lượng thay.

## Các hướng thử nghiệm Model 2 và cơ sở lý thuyết

Các kiến trúc đều dựa trên CNN: kernel tích chập học đặc trưng cục bộ từ ảnh; qua các tầng, đặc trưng thường đi từ cạnh và cấu trúc đơn giản đến texture và vùng tổn thương phức tạp hơn. Chia sẻ trọng số giúp CNN không cần học một bộ lọc riêng cho từng vị trí ảnh. Pooling hoặc stride giảm kích thước feature map, còn lớp phân loại cuối ánh xạ đặc trưng sang bốn lớp bệnh.

### 1. CNN tuần tự — `complex_cnn_balanced.ipynb`

Các khối Conv2D được xếp nối tiếp, số kênh tăng dần; mỗi khối dùng Batch Normalization, ReLU và MaxPooling. Đây là cách tổ chức CNN cơ bản, dễ đọc và giải thích: tầng sau nhận đặc trưng từ tầng trước rồi tiếp tục trích xuất đặc trưng ở mức cao hơn. Global Average Pooling ở cuối gộp không gian trước lớp phân loại.

- **Ưu điểm:** luồng dữ liệu thẳng, cấu trúc quen thuộc, ít nhánh nên dễ triển khai và theo dõi.
- **Hạn chế:** mỗi tầng chỉ xử lý biểu diễn mà tầng trước truyền sang; không có đường tắt nên mạng sâu có thể khó tối ưu hơn mạng nông.
- **Kết quả lần chạy:** accuracy 0.9293, macro-F1 0.8905. Macro-recall 0.9170 cho thấy trung bình các lớp được nhận diện khá tốt, nhưng kết quả giữa từng lớp vẫn khác nhau.

### 2. CNN nhiều nhánh song song — `parallel_cnn_balanced.ipynb`

Các nhánh xử lý cùng feature map bằng những phép tích chập khác nhau (ví dụ kernel 1×1, kernel 3×3 và pooling), sau đó ghép đặc trưng bằng Concatenate. Ý tưởng là học đặc trưng ở nhiều quy mô trong cùng một giai đoạn thay vì chỉ dùng một đường tuần tự.

- **Ưu điểm:** các nhánh có thể bổ sung cho nhau; kernel nhỏ và lớn/pooling nhạy với kiểu đặc trưng hoặc phạm vi không gian khác nhau.
- **Hạn chế:** cần giữ các nhánh đồng kích thước để ghép, tốn thêm bộ nhớ và làm kiến trúc khó theo dõi hơn CNN tuần tự. Nhiều nhánh không tự động đồng nghĩa với độ chính xác cao hơn.
- **Kết quả lần chạy:** accuracy 0.9122 và macro-F1 0.8568 là thấp nhất trong bốn lần chạy; macro-recall 0.8393 cũng thấp hơn các cấu hình còn lại. Kết quả này chỉ mô tả lần chạy cụ thể, không chứng minh rằng kiến trúc song song luôn kém.

### 3. CNN có kết nối tắt (Residual CNN) — `residual_cnn_balanced.ipynb`

Mỗi residual block tính một biến đổi bằng các lớp tích chập rồi cộng kết quả với đầu vào của block: `output = F(x) + x`. Nếu số kênh không khớp, phép tích chập 1×1 điều chỉnh nhánh tắt. Đường tắt tạo lối truyền thông tin và gradient trực tiếp hơn qua nhiều tầng, giúp việc tối ưu mạng sâu thuận lợi hơn.

- **Ưu điểm:** giảm phụ thuộc vào việc mỗi block phải tự học lại toàn bộ biểu diễn; có thể hỗ trợ huấn luyện mạng sâu.
- **Hạn chế:** cần căn chỉnh kích thước/kênh giữa hai nhánh; nhiều block làm tăng chi phí tính toán. Kết nối tắt không đảm bảo mô hình thắng nếu dữ liệu và siêu tham số chưa phù hợp.
- **Kết quả lần chạy:** accuracy 0.9535, macro-F1 0.9251 và best validation loss 0.1371. Trong ba mô hình Keras, đây là kết quả tốt nhất theo accuracy và macro-F1, nhưng thời gian đã ghi nhận rất lâu: 856.28 phút.

### 4. ResNet nhỏ `Model2_ResNet.ipynb`

Đây cũng là CNN residual nhưng được cài bằng PyTorch. Mô hình có stem tích chập, các stage gồm ResBlock với BatchNorm và skip connection, rồi Global Average Pooling và lớp phân loại. Cấu hình lần chạy bật class weights, label smoothing, AdamW, OneCycleLR và mixed precision trên GPU NVIDIA.

- **Ưu điểm:** residual block và BatchNorm hỗ trợ tối ưu mạng sâu; Global Average Pooling giữ phần phân loại gọn; các stage và đường tắt thể hiện rõ ý tưởng ResNet.
- **Hạn chế:** cách cài đặt có nhiều thành phần huấn luyện hơn CNN tuần tự; notebook hiện yêu cầu GPU NVIDIA CUDA nên không tiện chạy trực tiếp trên Mac không có CUDA.
- **Kết quả lần chạy:** accuracy 0.9716 và macro-F1 0.9522, cao nhất trong bốn lần chạy đã ghi nhận. Mô hình có 2,796,068 tham số, ảnh đầu vào 128 × 128 và thời gian train 84.8 phút.

### Vai trò của cân bằng lớp

Ba notebook có hậu tố `balanced` dùng class weights trong loss để tăng ảnh hưởng của các lớp ít mẫu. Đây là thay đổi trong cách huấn luyện, không phải một kiến trúc CNN riêng. Nó có thể giúp mô hình chú ý hơn đến lớp hiếm và cải thiện macro-recall/macro-F1, nhưng cũng có thể làm precision hoặc accuracy của lớp phổ biến giảm; không đảm bảo mọi metric đều tăng. Nên đánh giá bằng macro-F1 và recall từng lớp, không chỉ nhìn accuracy.

### Vì sao kết quả train khác nhau?

Sự khác nhau có thể đến từ nhiều yếu tố cùng lúc, không chỉ riêng hình dạng kiến trúc:

1. **Khả năng biểu diễn của kiến trúc:** số tầng, số kênh, nhánh song song, skip connection và cách gộp đặc trưng làm thay đổi những mẫu mô hình học được. Số tham số của các lần chạy cũng khác đáng kể, từ 423 nghìn đến 2.8 triệu.
2. **Độ phân giải và dữ liệu đánh giá:** ba notebook Keras dùng 224 × 224 và có 10,933 ảnh test; ResNet nhỏ dùng 128 × 128 và có 10,848 ảnh test. Do số lượng ảnh test khác nhau, không thể mặc định các lần chạy dùng chính xác cùng test split.
3. **Tiền xử lý và augmentation:** resize, crop, lật, xoay, dịch chuyển, zoom, thay đổi màu và che một phần ảnh làm thay đổi phân phối dữ liệu huấn luyện. Augmentation có thể giúp mô hình bớt phụ thuộc vào biến thể cụ thể, nhưng phép biến đổi quá mạnh hoặc không phù hợp ảnh OCT có thể làm mất đặc trưng hữu ích.
4. **Chiến lược tối ưu hóa:** learning rate, optimizer, scheduler, label smoothing, weight decay, batch size, số epoch và quy tắc chọn checkpoint tác động đến tốc độ/hướng hội tụ. ResNet nhỏ dùng AdamW + OneCycleLR; các notebook Keras dùng pipeline huấn luyện riêng, nên đây không chỉ là phép thử thay kiến trúc.
5. **Tính ngẫu nhiên và môi trường chạy:** khởi tạo trọng số, thứ tự batch, augmentation ngẫu nhiên, framework, GPU và cách thực thi có thể làm thay đổi cả điểm số lẫn thời gian. Vì vậy, không nên kết luận chênh lệch thời gian 84.8 và 856.28 phút chỉ do kiến trúc; các lần chạy còn khác framework/phần cứng.

Do các lần chạy khác nhau về độ phân giải, số lượng test, framework và cấu hình huấn luyện, bảng hiện tại là **so sánh kết quả thực nghiệm của các cấu hình đã chạy**, chưa phải thí nghiệm cô lập tác động của riêng kiến trúc. Muốn kết luận công bằng hơn, cần cố định cùng split, độ phân giải, tiền xử lý/augmentation, seed, batch size, số epoch và quy tắc chọn checkpoint; sau đó chỉ thay kiến trúc, tốt nhất chạy nhiều seed và báo cáo trung bình.

Giữ [`Model2_ResNet.ipynb`](../src/Model_2/src/Model2_ResNet.ipynb): trong bốn lần chạy này, nó có accuracy và macro-F1 cao nhất, đồng thời thời gian chạy được ghi nhận là 84.8 phút. Kiến trúc dùng các residual block tiêu chuẩn (`ResBlock` và `ResNetSmall`) với BatchNorm và skip connection; notebook có các mục giải thích theo từng bước. Đây không phải mô hình đơn giản nhất, nhưng là lựa chọn tốt nhất khi cân bằng kết quả và khả năng giải thích trong bốn lần chạy đã có.

Lưu ý: `Model2_ResNet.ipynb` dùng ảnh 128 × 128 và tập test có 10,848 ảnh; ba notebook Keras dùng ảnh 224 × 224 và tập test có 10,933 ảnh. Vì vậy, các điểm số hiện tại không phải so sánh hoàn toàn cùng điều kiện và chưa chứng minh chênh lệch chỉ đến từ kiến trúc. Notebook ResNet hiện yêu cầu GPU NVIDIA CUDA; không chạy trực tiếp được trên Mac không có CUDA.

---

# Hien: Model 3 - Transfer Learning / Fine-tuning