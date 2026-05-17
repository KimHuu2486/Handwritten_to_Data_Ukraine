# 📝 Báo cáo ngữ cảnh dự án (Project Context)

## 1. 🎯 Tóm tắt dự án (Project Overview)
- **Mục tiêu:** Xây dựng pipeline End-to-End Document Understanding cho cuộc thi Kaggle Handwritten to Data (RUKOPYS), đầu ra là submission CSV gồm bbox + type + text cho tài liệu viết tay tiếng Ukraina.
- **Mục tiêu phụ:** Chuẩn bị hướng nghiên cứu VLM + CoT (Curriculum Silver -> Gold -> CoT + hậu xử lý) theo lộ trình ablation và cache-first.
- **Trạng thái hiện tại:** Đang phát triển. Đã có notebook Stage 1 (silver warm-up) và Stage 2 (gold fine-tune), cùng notebook inference/submission hai-pass (page JSON + crop OCR). Đang khóa B2 baseline, chuẩn bị ablation theo data/model plan giai đoạn tiếp theo.

## 2. 🛠️ Công nghệ đã sử dụng (Tech Stack)
- **Frontend/Client:** Không có frontend; workflow chủ yếu qua Kaggle Notebook + script Python.
- **Backend/Server:** Không có server riêng; pipeline batch offline.
- **Database:** Không dùng DB; dữ liệu ở JSONL/CSV.
- **Công cụ/Thư viện khác:** Python, PyTorch, Transformers, TRL (SFTTrainer), PEFT (LoRA/QLoRA), bitsandbytes, qwen-vl-utils, PIL, pandas, multiprocessing.
- **Mô hình chính:** Qwen3-VL-8B (detection+transcription end-to-end), Qwen2.5-1.5B-Instruct (spell-check text hậu xử lý).
- **Metric/chuẩn đánh giá:** IoU matching, class accuracy, CER/PageCER, text normalization theo rule chính thức.

## 3. ✅ Công việc đã hoàn thành (Completed Tasks)
- [x] Phân tích rule chấm điểm và format submit chính thức (bbox, type, text; JSON trong cột regions).
- [x] Xây baseline zero-shot cho Qwen3-VL và xuất submission.
- [x] Xây script chuẩn bị dữ liệu train/val VLM từ metadata (`prepare_vlm_dataset.py`).
- [x] Xây notebook Stage 1 silver warm-up (tạo page JSON + crop OCR samples, LoRA/QLoRA, checkpoint + prompt config).
- [x] Xây notebook Stage 2 gold fine-tune (load Stage 1 adapter, train tiếp trên train split, lưu validation records).
- [x] Xây notebook inference/submission hai-pass: page JSON -> crop OCR, checkpoint theo GPU, resume từ partial CSV.
- [x] Xây script fine-tune Qwen3-VL bằng LoRA/QLoRA, có custom callback log và cơ chế phục hồi OOM (`train_qwen_lora.py`).
- [x] Xây script inference fine-tuned đa GPU, parse JSON output, rescale bbox về ảnh gốc, sort theo thứ tự đọc (`kaggle_finetuned_inference.py`).
- [x] Xây hậu xử lý spell-check bằng LLM nhỏ để sửa lỗi OCR text (`spell_check_submission.py`).
- [x] Có tài liệu kế hoạch và nghiên cứu: kế hoạch tổng thể, báo cáo metric, tài liệu CoT/VLM trong thư mục `document/`.
- [x] Soạn kế hoạch next phase theo hướng cache-first + ablation ladder (data plan + model plan) trong `document/plan/`.

## 4. 🐛 Những lỗi đã khắc phục (Fixed Bugs)
- **Bug 1: Sai/không nhất quán định dạng bbox** -> **Cách giải quyết:** Chuẩn hóa quy đổi bbox về [x1, y1, x2, y2], hỗ trợ fallback khi model trả theo pixel hoặc normalized.
- **Bug 2: Lỗi OOM khi train/inference trên Kaggle T4** -> **Cách giải quyết:** Dùng 4-bit quantization, gradient checkpointing, giới hạn max pixels (page/crop), và dọn cache GPU định kỳ.
- **Bug 3: Crash do bfloat16 trên môi trường T4** -> **Cách giải quyết:** Ép dtype về float16/float32 cho model và tham số trainable LoRA.
- **Bug 4: Output model không phải JSON sạch** -> **Cách giải quyết:** Thêm lớp robust parsing (strip code fences, greedy JSON array extract, fallback object regex).
- **Bug 5: Lệch thứ tự đọc ảnh gây giảm PageCER** -> **Cách giải quyết:** Sort regions theo trục y/x trước khi ghi kết quả.
- **Bug 6: Khó resume inference khi chạy đa GPU** -> **Cách giải quyết:** Checkpoint theo GPU, merge partial CSV và cho phép resume từ file có sẵn.

## 5. 🚀 Công việc tiếp theo (Current / Pending Tasks)
- [ ] Khóa `frozen_validation_manifest.jsonl` và tạo/kiểm tra `baseline_predictions_cache/` cho B2.
- [ ] Chạy ablation crop OCR mode (none/smart/all_text) và ghi `ablation_results.csv`.
- [ ] Tạo `risk_labels_cache.parquet` + routing curve rẻ (rule/entropy/disagreement).
- [ ] Tạo `refine_candidates.jsonl`, test one-step zoom-only trên subset; chỉ mở reasoning/SFT nếu có gain.
- [ ] Chuẩn hóa reading-order + post-processing ablation và so sánh PageCER.
- [ ] Cập nhật checklist validation CSV/JSON parse theo submission contract.

## 6. 📂 Cấu trúc thư mục cốt lõi (Core Structure)
- `baseline/`: Baseline zero-shot và kết quả submit tham chiếu.
- `finetune/`: Notebook Stage 1/2, inference/submission 2-pass, train LoRA, spell-check, kết quả finetune.
- `VLM_OCR/`: Pipeline OCR theo hướng crop region + VLM và script chuyển đổi định dạng.
- `dataset/`: Metadata các split (`train`, `test`, `sliver`) và file mẫu submission.
- `document/`: Tài liệu kế hoạch CoT/VLM, ghi chú thiết kế và paper notes.
- `document/plan/`: Kế hoạch data/model giai đoạn tiếp theo (cache-first, ablation ladder).
- `official-evaluation-metric-text-normalization.ipynb`: Notebook chuẩn để bám sát metric/normalization chính thức.
- `plan.md`, `report.md`, `README.md`: Kế hoạch tổng quan, báo cáo phân tích và mô tả dataset/cuộc thi.
