# Các Script Xử Lý Dữ Liệu (Data Processing Scripts)

Thư mục này chứa các script Python được sử dụng để xử lý, lọc, tăng cường dữ liệu và đánh giá mô hình. Phần lớn các script này nhằm mục đích chuẩn bị dữ liệu chất lượng cao cho việc huấn luyện, tạo ra các tập con (subset) dữ liệu đặc thù để khám phá, cũng như làm sạch dữ liệu nhiễu.

Dưới đây là mô tả chi tiết chức năng của từng file:

## 1. Khám phá và Tạo tập dữ liệu con (Filtering & Subsetting)

* **`filter_silver_rare.py`**
  * **Chức năng:** Trích xuất (crop) các vùng dữ liệu thuộc các lớp hiếm (rare classes) như `annotation`, `table`, `formula` từ tập dữ liệu bạc (silver dataset). 
  * **Đặc điểm:** Giới hạn số lượng mẫu cho mỗi lớp (mặc định 2000 mẫu/lớp) để tránh mất cân bằng. Chuẩn hóa metadata và vẽ các ảnh preview (kèm bounding box) có chứa text để dễ dàng kiểm tra trực quan.

* **`filter_gold_rare.py`**
  * **Chức năng:** Duyệt qua file metadata (JSONL), lọc ra những ảnh chứa các nhãn hiếm (ví dụ: `table`, `graph`, `image`, `formula`) và copy chúng vào một tập dữ liệu con nhỏ gọn hơn.
  * **Đặc điểm:** Hỗ trợ vẽ preview các bounding box trực tiếp lên ảnh để dễ dàng đánh giá chất lượng gán nhãn của các lớp hiếm này.

## 2. Tăng cường dữ liệu (Data Augmentation)

* **`augment_gold_rare.py`**
  * **Chức năng:** Tăng cường dữ liệu (augmentation) chuyên biệt cho các nhãn hiếm (gold rare).
  * **Đặc điểm:** Script thực hiện các kỹ thuật như trích xuất vùng foreground, ước lượng màu nền (background/paper color), và tự động loại bỏ viền thừa (trim to foreground). Hỗ trợ link hoặc copy ảnh để tạo ra các biến thể mới cho tập huấn luyện, giúp mô hình học tốt hơn trên các mẫu dữ liệu hiếm.

## 3. Thử nghiệm và Tiền xử lý ảnh (Image Preprocessing & Scanning)

* **`train_image_preprocessing_pipeline.py`**
  * **Chức năng:** Đây là script **thử nghiệm scan ảnh dữ liệu**. Đóng vai trò là một pipeline tiền xử lý ảnh toàn diện (Image Preprocessing Pipeline) nhằm nâng cao chất lượng ảnh huấn luyện theo phong cách "máy scan" (scanner-like).
  * **Đặc điểm:**
    * Nhận diện và biến đổi phối cảnh trang giấy (Page warp/Deskew) khi độ tin cậy cao.
    * Tự động điều chỉnh bounding box (bbox) hoặc polygon khớp với các biến đổi hình học.
    * Tăng cường chất lượng ảnh: giảm bóng đổ, cân bằng sáng (CLAHE), khử nhiễu (denoise) và làm sắc nét (sharpen).
    * Giữ nguyên dữ liệu gốc, chỉ xuất ra ảnh và metadata đã qua xử lý.

## 4. Đánh giá OOF và Làm sạch dữ liệu (OOF Audit & Noise Removal)

* **`oof_label_audit.py`**
  * **Chức năng:** Script **đánh giá Out-Of-Fold (OOF) để loại bỏ dữ liệu nhiễu**. Sử dụng dự đoán từ các mô hình cross-validation (như M12, M13, M23) áp dụng lên tập valid để so sánh với nhãn gốc (Ground Truth).
  * **Đặc điểm:**
    * Tính toán các chỉ số IoU, Character Error Rate (CER).
    * Đánh giá độ khả nghi (Suspicious Score) của từng vùng (region) hoặc toàn bộ ảnh.
    * Tự động đánh dấu và loại bỏ các bounding box bị lỗi, text sai, hoặc các nhãn gán kém chất lượng. Kết quả xuất ra là một file metadata đã được làm sạch (Cleaned Metadata).

* **`official_formula_image_audit.py`**
  * **Chức năng:** Mở rộng từ quá trình đánh giá OOF, tập trung vào việc chấm điểm chất lượng ảnh tổng thể dựa trên các thước đo chính thức (Official-like metrics).
  * **Đặc điểm:** Tính toán F1-score cho Detection, Accuracy cho Classification, CER cho vùng và cho cả trang (Page CER). Dựa vào các chỉ số này để phân loại mức độ nhiễu/lỗi của ảnh thành các mức độ như `ok`, `low`, `review`, `drop`.

---
*Lưu ý: Hầu hết các script đều hỗ trợ chạy bằng command line (CLI) với các tham số tuỳ chỉnh qua `argparse`.*
