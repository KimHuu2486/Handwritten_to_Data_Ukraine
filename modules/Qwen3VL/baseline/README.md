# Qwen3-VL Baseline

Thư mục này lưu ba phiên bản baseline dùng Qwen3-VL 8B để nhận diện vùng tài liệu,
đọc chữ viết tay và tạo dữ liệu theo định dạng submission của RUKOPYS.

## Thứ tự đọc đề xuất

1. [`kaggle_zero_shot_baseline_v1.py`](./kaggle_zero_shot_baseline_v1.py)
   để hiểu pipeline cơ bản trên một tiến trình.
2. [`kaggle_zero_shot_baseline_v2.py`](./kaggle_zero_shot_baseline_v2.py)
   để xem cách pipeline được mở rộng sang nhiều GPU và bổ sung few-shot.
3. [`kaggle_zero_shot_baseline_v3.py`](./kaggle_zero_shot_baseline_v3.py)
   để xem phiên bản hoàn thiện hơn về bbox, checkpoint và xử lý lỗi.

Nếu chỉ cần phiên bản hiện tại để tham khảo, hãy đọc **V3**. Nếu muốn hiểu lý do
của từng thay đổi, hãy đọc lần lượt **V1 → V2 → V3**.

## So sánh nhanh

| Nội dung | V1 | V2 | V3 |
|---|---|---|---|
| Cách chạy | Một tiến trình | Một worker cho mỗi GPU | Một worker cho mỗi GPU |
| Prompt | Phát hiện các vùng văn bản | Tách riêng từng dòng | Tách từng dòng và giữ cả chữ bị gạch |
| Few-shot | Không | Có một ví dụ | Có ví dụ được thay đổi |
| Hệ tọa độ mong đợi | Pixel hoặc tự nhận biết | Pixel hoặc tự nhận biết | Chuẩn hóa `0–1000` |
| `MAX_PIXELS` | `1003520` | `786432` | `786432` |
| `max_new_tokens` | `2048` | `4096` | `4096` |
| Checkpoint | Một file chung | Một file cho mỗi GPU | Có thể khôi phục checkpoint từ Kaggle Dataset |
| Xử lý lỗi từng ảnh | Chưa có | Chưa có | Có, ảnh lỗi trả về danh sách rỗng |
| `TEST_MODE` | Không có | Bật | Tắt |

## Điểm thay đổi chính

### V1 → V2

- Chuyển từ một tiến trình sang multiprocessing theo GPU.
- Yêu cầu model trả về một region cho mỗi dòng.
- Bổ sung few-shot, parser JSON dự phòng và checkpoint riêng cho từng worker.
- Giảm kích thước ảnh đầu vào và tăng giới hạn token đầu ra.

### V2 → V3

- Yêu cầu bbox theo hệ tọa độ tương đối `0–1000`.
- Cải thiện cách phân biệt bbox pixel, bbox `0–1` và bbox `0–1000`.
- Bổ sung nhận diện chữ bị gạch, xử lý lỗi từng ảnh và khôi phục checkpoint.
- Chỉ xóa checkpoint sau khi tạo được file submission hợp lệ.

## Luồng đọc bên trong mỗi file

Đọc theo thứ tự: **cấu hình → prompt → parse JSON → chuẩn hóa bbox → inference
một ảnh → worker GPU → `main()`**. V1 không có worker riêng; phần inference được
gọi trực tiếp từ `main()`.

> [!NOTE]
> Tên file vẫn chứa `zero_shot`, nhưng V2 và V3 có chèn ví dụ vào prompt nên về
> bản chất là few-shot inference.
