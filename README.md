# G8_AI66A_DNN
# Tổng quan Dự án: So sánh các Mô hình Deep Learning trong Chẩn đoán Bệnh lý về Mắt (OCT)

> **Tên đề tài:** Comparative Deep Learning Models for Human Eye Disease Prediction  
> **Lĩnh vực:** Thị giác máy tính (Computer Vision), Mạng nơ-ron sâu (Deep Neural Networks), Y tế số (Digital Health)  
> **Mục tiêu:** Xây dựng, huấn luyện và so sánh hiệu năng 3 kiến trúc Deep Learning trong việc tự động phân loại 4 trạng thái bệnh lý võng mạc từ ảnh chụp cắt lớp quang học (OCT), đồng thời tích hợp mô hình tối ưu vào ứng dụng web chẩn đoán tự động.
> **Dataset**: https://www.kaggle.com/datasets/anirudhcv/labeled-optical-coherence-tomography-oct

---

## 1. Bối cảnh & Mục tiêu Dự án

Chẩn đoán sớm các bệnh lý võng mạc đóng vai trò quyết định trong việc ngăn ngừa nguy cơ suy giảm thị lực và mù lụa ở bệnh nhân. Phương pháp chụp cắt lớp quang học võng mạc (**Optical Coherence Tomography - OCT**) cung cấp hình ảnh lát cắt vi mô chi tiết của các lớp võng mạc. Tuy nhiên, việc đánh giá thủ công hàng nghìn bức ảnh đòi hỏi thời gian và chuyên môn cao từ bác sĩ nhãn khoa.

Dự án được xây dựng nhằm tự động hóa quy trình phân loại ảnh OCT vào **4 nhóm trạng thái**:

1. **CNV (Choroidal Neovascularization - Tân mạch màng mạch):** Tình trạng phát triển bất thường của các mạch máu dưới võng mạc, gây tổn thương cấu trúc tế bào.
2. **DME (Diabetic Macular Edema - Phù hoàng điểm do tiểu đường):** Sự tích tụ dịch ở hoàng điểm do biến chứng mạch máu từ bệnh tiểu đường.
3. **Drusen (Lắng đọng dịch dưới võng mạc):** Các đốm lắng đọng màu vàng dưới võng mạc, là dấu hiệu sớm của thoái hóa hoàng điểm liên quan đến tuổi tác (AMD).
4. **Normal (Mắt bình thường):** Cấu trúc võng mạc khỏe mạnh, các lớp tế bào xếp chồng đồng đều, không phát hiện dấu hiệu bệnh lý.

---

## 2. Dữ liệu & Chiến lược Tiền xử lý

### 2.1. Tập dữ liệu Retinal OCT
* **Quy mô:** Thư mục `data/raw` hiện có **109.309 ảnh** trong ba tập train/validation/test.
* **Tỷ lệ phân chia:** Dữ liệu được phân chia thành 3 tập độc lập:
  * **Tập Huấn luyện (Train set):** Dùng để cập nhật trọng số cho các mô hình.
  * **Tập Kiểm định (Validation set):** Dùng để theo dõi hiện tượng quá khớp (Overfitting) và tinh chỉnh siêu tham số.
  * **Tập Kiểm thử (Test set):** Dùng để đánh giá độc lập hiệu năng cuối cùng của cả 3 mô hình.

### 2.2. Chiến lược Quản lý Dữ liệu Thực nghiệm (Data Sampling Strategy)
Do tập dữ liệu raw có dung lượng và số lượng ảnh lớn, việc huấn luyện trực tiếp ngay từ đầu sẽ tiêu tốn nhiều thời gian và chi phí tính toán. Nhóm đã áp dụng chiến lược **Chia nhỏ dữ liệu (Data Partitioning)**:
* **Tập dữ liệu nhỏ (Sampled Data):** Trích xuất một tập dữ liệu đại diện có quy mô nhỏ hơn, đặt song song với thư mục dữ liệu thô (Raw Data).
* **Mục đích:** Giúp nhóm kiểm thử nhanh luồng mã nguồn (Data Loader, Pipeline, Architecture), phát hiện lỗi lập trình và đảm bảo mô hình hoạt động ổn định trước khi tiến hành huấn luyện quy mô lớn trên toàn bộ dữ liệu thô.

---

## 3. Kiến trúc 3 Mô hình Deep Learning

Nhóm thiết kế và so sánh **3 kiến trúc mô hình** với mức độ phức tạp tăng dần nhằm đánh giá sự đánh đổi giữa chi phí tính toán và độ chính xác:

| Tiêu chí | Mô hình 1: Simple CNN | Mô hình 2: Complex CNN | Mô hình 3: Transfer Learning (MobileNet) |
| :--- | :--- | :--- | :--- |
| **Đặc điểm kiến trúc** | Mạng cuộn xếp chồng tuần tự nông (2-3 lớp Conv2D + Pooling) | Mạng cuộn đa tầng nhiều khối (Multi-block CNN) kết hợp khối chức năng nâng cao | Kiến trúc MobileNet đã được huấn luyện sẵn trên tập ImageNet |
| **Kỹ thuật chống Overfitting** | Không sử dụng hoặc chỉ có lớp Flatten/Dense cơ bản | Tích hợp **Data Augmentation** (xoay, lật, zoom), **Batch Normalization** và **Dropout** | Đóng đóng các lớp trích xuất đặc trưng (Frozen Base) và chỉ huấn luyện Custom MLP Head |
| **Mục đích thử nghiệm** | Tạo mô hình cơ sở (Baseline) để đánh giá ngưỡng hiệu năng tối thiểu | Tối ưu hóa khả năng trích xuất đặc trưng sâu từ đầu mà không dùng tri thức bên ngoài | Tận dụng tri thức học sẵn để đạt độ chính xác cao nhất với thời gian huấn luyện ngắn |
| **Phân công đảm nhận** | Thư đảm nhận triển khai | Lân đảm nhận triển khai | Cả nhóm phối hợp tinh chỉnh (Fine-tuning) |

---

## 4. Khung Đánh giá & Tiêu chí So sánh

Sau khi huấn luyện trên tập Test độc lập, cả 3 mô hình sẽ được so sánh toàn diện dựa trên hai nhóm chỉ số:

### 4.1. Độ đo Chất lượng Chẩn đoán (Diagnostic Metrics)
* **Accuracy (Độ chính xác tổng thể):** Tỷ lệ dự đoán đúng trên toàn bộ 4 lớp.
* **Precision (Độ chuẩn xác):** Đo lường tỷ lệ các ca thực sự mắc bệnh trong số các ca mô hình chẩn đoán là bệnh.
* **Recall (Độ nhạy):** Tỷ lệ phát hiện thành công các bệnh nhân thực sự mắc bệnh (chỉ số tối quan trọng trong y tế để tránh bỏ sót bệnh nhân).
* **F1-Score:** Giá trị trung bình hài hòa giữa Precision and Recall.
* **Confusion Matrix (Ma trận nhầm lẫn):** Trực quan hóa chi tiết các trường hợp chẩn đoán đúng và các cặp nhầm lẫn giữa 4 lớp bệnh.

### 4.2. Chi phí Tính toán & Tài nguyên (Computational Cost)
* **Tổng số tham số (Total Parameters):** So sánh dung lượng bộ nhớ mà mô hình yêu cầu.
* **Thời gian suy luận (Inference Time):** Thời gian trung bình để mô hình xử lý và đưa ra kết quả cho 1 bức ảnh OCT.
* **Kích thước lưu trữ (.h5 / .keras):** Độ gọn nhẹ khi đóng gói mô hình.

---

## 5. Tích hợp Ứng dụng Web Diagnostic (Web Application)

Mô hình đạt kết quả tối ưu nhất về độ chính xác và tốc độ suy luận sẽ được trích xuất file trọng số và nhúng vào một **Ứng dụng Web Chẩn đoán Tự động (Diagnostic Web App)**:
* **Giao diện người dùng:** Cho phép bác sĩ hoặc người dùng tải lên (upload) ảnh chụp OCT bất kỳ.
* **Luồng xử lý:** Ảnh được tiền xử lý tự động và đưa qua mô hình Deep Learning để suy luận.
* **Kết quả đầu ra:** Hiển thị tức thì nhãn bệnh lý dự đoán (CNV, DME, Drusen, Normal), tỷ lệ xác suất phân bố cho từng lớp và các khuyến nghị y tế sơ bộ.

---

## 6. Tiến độ Thực hiện & Phân công Công việc

Nhóm đã hoàn thành việc thiết lập hạ tầng mã nguồn và đang tiến hành các bước thử nghiệm theo lộ trình:

1. **Khởi tạo dự án & Quản lý Mã nguồn:**
   * Khởi tạo repository công khai trên GitHub tại địa chỉ `G8_AI66A_DNN`.
   * Cấu trúc thư mục mã nguồn bao gồm các module tiền xử lý, định nghĩa kiến trúc và lưu trữ trọng số.
2. **Chiến lược Tiền xử lý Dữ liệu:**
   * Đã hoàn tất công đoạn chia tập dữ liệu thử nghiệm nhỏ (`Chia nhỏ data`) để kiểm thử đường ống nạp ảnh (Image Data Pipeline).
3. **Phân công Triển khai Mô hình:**
   * **Thư:** Đảm nhận triển khai mã nguồn và huấn luyện **Model 1 (Simple CNN)**.
   * **Lân:** Đảm nhận nghiên cứu, xây dựng và huấn luyện **Model 2 (Complex CNN)** với các khối Data Augmentation, Batch Normalization và Dropout.
   * **Cả nhóm:** Sau khi Model 1 và Model 2 chạy ổn định trên tập data nhỏ, nhóm sẽ mở rộng huấn luyện trên dữ liệu thô (Raw Data) và tiếp tục triển khai **Model 3 (MobileNet Transfer Learning)**.

---

## 7. Kế hoạch Hoàn thiện Dự án (Roadmap)

```
[Hoàn thiện Model 1 & 2 trên Data nhỏ]
                 │
                 ▼
[Huấn luyện Scale-up trên Full Raw Data]
                 │
                 ▼
[Triển khai Model 3 (MobileNet Transfer Learning)]
                 │
                 ▼
[Đánh giá So sánh 3 Model & Trích xuất Metrics / Confusion Matrix]
                 │
                 ▼
[Đóng gói Model tối ưu ➔ Xây dựng Web App Diagnostic]
                 │
                 ▼
[Hoàn thiện Báo cáo Word theo Mẫu Major_Assignment_Report]
```