# Hướng dẫn phân công viết `ImplementationDetail.pdf`

> Tài liệu này dùng để **phân công nội bộ** cho các thành viên viết phần supplementary `ImplementationDetail.pdf`.
> Đây **không phải** bản supplementary final. Mỗi người phụ trách cần viết nội dung của section mình, có thể thêm bớt nội dung sao cho hợp lí, kiểm tra số liệu với notebook/log/code tương ứng, rồi tổng hợp sang LaTeX.

## 1. Mục tiêu của file phân công

`ImplementationDetail.pdf` là supplementary độc lập, chỉ tập trung vào **chi tiết triển khai** của hệ thống BoustoDoc. File này không thay thế main paper, mà giúp reviewer hiểu rõ hơn:

- Dữ liệu train/validation/test được chia và sử dụng như thế nào.
- Detector DocLayout-YOLO được train, augment, validate và ensemble ra sao.
- TrOCR HPA branch được fine-tune theo curriculum như thế nào.
- Qwen3-VL LoRA branch được train, prompt và inference như thế nào.
- Final inference pipeline route từng region type ra sao.
- Output `submission.csv` được serialize và validate như thế nào.
- Môi trường chạy, package chính, GPU/runtime có gì cần biết.

Mục tiêu của từng thành viên là viết phần của mình theo hướng **ngắn, kiểm chứng được, không phóng đại claim, không đưa thêm kết quả mới sau deadline**.

## 2. Quy tắc chung khi viết

- Main paper phải self-contained. Supplementary chỉ bổ sung chi tiết implementation, không dùng để né page limit.
- Không đưa kết quả mới từ version method cải tiến sau deadline.
- Không đưa bản paper đã sửa lại vào supplementary.
- Không dùng external link để mở rộng nội dung submission.
- Nếu đưa code/config/log/prompt thì phải đưa trực tiếp trong supplementary package và phải ẩn danh.
- Không ghi tên cá nhân, username, email, private path, Kaggle/HuggingFace/GitHub private link.
- Mọi số liệu phải khớp với main paper hoặc log/notebook final. Nếu chưa chắc, ghi rõ `NEED VERIFY` để tổng hợp lại.
- Viết bằng tiếng Anh cho đoạn sẽ đưa vào supplementary. Ghi chú nội bộ có thể để tiếng Việt.

## 3. Phân công nhanh

| Section | Nội dung | Người phụ trách | Output |
|---|---|---|---|
| Section 1 | Scope of This Supplement | NQH | 1 đoạn tiếng Anh 5-7 câu |
| Section 2 | Shared Data Protocol | NQH | Đoạn mô tả split + 2 bảng data |
| Section 3 | Detector Training Details | LQM | Mô tả data/preprocess/train/ensemble + bảng config |
| Section 4 | HPA OCR Training Details | HVTB | Mô tả TrOCR curriculum + typed decoding + validation |
| Section 5 | Qwen3-VL LoRA Training Details | LHNH | Mô tả data/LoRA/prompt/inference + bảng config |
| Section 6 | Final Inference and Routing | LHNH | Mô tả end-to-end inference + routing + output |
| Section 7 | Hardware, Runtime, and Reproducibility | LQM + HVTB + LHNH | Mỗi người điền môi trường/hardware phần mình |
| Section 8 | Anonymization Checklist | NQH | Checklist ẩn danh trước khi nộp |

## 4. Cấu trúc dự kiến của `ImplementationDetail.pdf`

```latex
\section{Scope of This Supplement}
\section{Shared Data Protocol}
  \subsection{Gold/Silver Data Usage}
  \subsection{Page-Level Internal Split}
  \subsection{General Preprocessing Rules}
\section{Detector Training Details}
  \subsection{Detector Training Data Preparation}
  \subsection{Class Mapping and Bounding Box Conversion}
  \subsection{Class-Aware Augmentation}
  \subsection{Detector Training Hyperparameters}
  \subsection{Detector Inference and Ensemble}
\section{HPA OCR Training Details}
  \subsection{HPA Crop Data Preparation}
  \subsection{Tokenizer Adaptation}
  \subsection{Silver-to-Gold Curriculum}
  \subsection{Type-Specific Decoding}
  \subsection{HPA Validation Protocol}
\section{Qwen3-VL LoRA Training Details}
  \subsection{Data Preparation}
  \subsection{Instruction Data Construction}
  \subsection{LoRA Training Configuration}
  \subsection{Prompt Templates}
  \subsection{Qwen Inference Settings}
\section{Final Inference and Routing}
  \subsection{Region Routing Policy}
  \subsection{Post-Processing}
  \subsection{Output Serialization and Validation}
\section{Hardware, Runtime, and Reproducibility}
\section{Anonymization Checklist}
```

---

# Section 1 -- Scope of This Supplement -- NQH

## Mục tiêu

Giới thiệu ngắn rằng supplementary này chỉ bổ sung chi tiết triển khai cho BoustoDoc. Không lặp lại toàn bộ method, result hoặc ablation trong main paper.

## Output cần nộp

- Một đoạn tiếng Anh 5-7 câu.
- Không cần bảng.
- Không đưa kết quả mới.

## Nội dung bắt buộc

- Nói rõ file này chỉ cung cấp implementation details.
- Nhắc lại pipeline ở mức rất ngắn: detector -> router -> TrOCR/Qwen -> output.
- Nói rõ main result và ablation chính nằm trong main paper.
- Nói rõ mục tiêu là giúp setup dễ inspect/reproduce hơn.

## Template gợi ý

```latex
\section{Scope of This Supplement}

This supplementary file provides implementation details for the BoustoDoc pipeline. It expands the training, inference, routing, prompt, and validation settings described in the main paper. The main paper remains self-contained: the system design, main results, and key routing ablation are reported there. At a high level, BoustoDoc detects page regions, routes each region by predicted type, applies either TrOCR or Qwen3-VL when text recognition is required, and serializes the final output. This file is intended to make the experimental setup easier to inspect and reproduce.
```

## Tiêu chí hoàn thành

- [ ] Đoạn viết không dài quá 7 câu.
- [ ] Không lặp bảng/result từ main paper.
- [ ] Không có claim mới ngoài main paper.

---

# Section 2 -- Shared Data Protocol -- NQH

## Mục tiêu

Giải thích rõ dữ liệu được chia và dùng như thế nào, tránh reviewer nghi ngờ leakage hoặc dùng silver sai cách.

## Output cần nộp

- 2-3 đoạn tiếng Anh.
- Bảng internal split.
- Bảng source distribution.
- Ghi rõ silver chỉ là auxiliary noisy training data.

## Nội dung bắt buộc
- Cách chia thì liên hệ TKH để lấy nguồn để viết cho chính xác (file `split_summary_70_15_15.md` đã từng gửi trong drive data).
- Gold set được chia theo **page-level split**.
- Không tách các regions của cùng một page sang nhiều split khác nhau.
- Split 70/15/15: train / validation / internal test.
- Source-stratified shuffling.
- Validation dùng cho checkpoint selection và hyperparameter tuning.
- Internal test chỉ dùng cho final internal reporting.
- Silver chỉ là auxiliary noisy data, không dùng để chọn checkpoint hoặc final metric.

## Bảng internal split

```latex
\begin{table}[t]
\centering
\caption{Internal page-level split used for controlled evaluation.}
\begin{tabular}{lrrl}
\toprule
Split & Pages & Regions & Usage \\
\midrule
Gold train & 931 & 18,007 & Supervised training \\
Gold validation & 199 & 3,823 & Checkpoint selection and tuning \\
Gold internal test & 200 & 3,821 & Final internal evaluation \\
Silver auxiliary train & 8,207 & 161,065 & Noisy auxiliary training only \\
\bottomrule
\end{tabular}
\end{table}
```

## Bảng source distribution

```latex
\begin{table}[t]
\centering
\caption{Source distribution of the internal gold split.}
\begin{tabular}{lrrrr}
\toprule
Source & Train & Validation & Test & Total \\
\midrule
archive & 89 & 19 & 19 & 127 \\
dictation & 251 & 54 & 54 & 359 \\
school & 478 & 102 & 102 & 682 \\
university & 113 & 24 & 25 & 162 \\
\bottomrule
\end{tabular}
\end{table}
```

## Ghi chú cần nhấn mạnh

- Page-level split quan trọng hơn region-level split vì tránh việc crop từ cùng một ảnh xuất hiện ở cả train và test.
- Silver data có thể nhiễu, nên chỉ dùng cho warm-up, augmentation hoặc auxiliary training.
- Official test labels bị hidden, nên internal test là protocol kiểm soát để báo cáo trong paper.

## Tiêu chí hoàn thành

- [ ] Split numbers khớp `data/split_summary_70_15_15.md`.
- [ ] Có câu nói rõ silver không dùng cho checkpoint/final metric.
- [ ] Không có mô tả làm reviewer hiểu nhầm silver là ground truth.

---

# Section 3 -- Detector Training Details -- LQM

## Mục tiêu

Mô tả rõ DocLayout-YOLO được fine-tune ra sao, class mapping như thế nào, vì sao cần class-aware augmentation, và final detector ensemble hoạt động ra sao.

## Output cần nộp

- 4-6 đoạn tiếng Anh.
- Bảng class mapping.
- Bảng detector training hyperparameters.
- Bảng hoặc đoạn ngắn về inference/ensemble settings.
- Ghi chú rõ thông số nào là final, thông số nào còn cần verify.

## Nội dung bắt buộc

- Cách chuẩn bị data cho finetune.
- Base model: DocLayout-YOLO pretrained checkpoint.
- 7 classes: `handwritten`, `printed`, `formula`, `table`, `annotation`, `image`, `graph`.
- Gold annotations + rare-class auxiliary samples nếu đúng với final training.
- COCO bbox -> YOLO normalized format.
- Bbox validation và clamping.
- Class-aware augmentation cho rare classes.
- V4.1 và V4.2 checkpoints.
- Final detector dùng ensemble.

## Bảng class mapping

```latex
\begin{table}[t]
\centering
\caption{Detector class mapping.}
\begin{tabular}{ll}
\toprule
YOLO ID & Region type \\
\midrule
0 & handwritten \\
1 & printed \\
2 & formula \\
3 & table \\
4 & annotation \\
5 & image \\
6 & graph \\
\bottomrule
\end{tabular}
\end{table}
```

## Bảng hyperparameters cần điền/kiểm tra

```latex
\begin{table}[t]
\centering
\caption{Detector training hyperparameters.}
\begin{tabular}{ll}
\toprule
Parameter & Value \\
\midrule
Model & DocLayout-YOLO \\
Initialization & DocStructBench pretrained checkpoint \\
Number of classes & 7 \\
Image size & NEED VERIFY \\
Batch size & 8 \\
Optimizer & Adam \\
Initial learning rate & 0.001 \\
Weight decay & 0.0005 \\
Momentum & 0.9 \\
Warm-up epochs & NEED VERIFY \\
Early stopping patience & NEED VERIFY \\
V4.1 mosaic & 0.1 \\
V4.2 mosaic & 0.25 \\
\bottomrule
\end{tabular}
\end{table}
```

## Gợi ý viết

```latex
\section{Detector Training Details}

The detector is fine-tuned from a DocLayout-YOLO document-layout checkpoint. We use seven RUKOPYS region classes and convert all bounding boxes to normalized YOLO format. Bounding boxes are validated and clamped to the image boundaries before training. To reduce the effect of class imbalance, we apply class-aware augmentation with higher priority for rare or structurally important classes such as table, image, graph, and annotation. The final inference system uses an ensemble of two detector checkpoints, V4.1 and V4.2.
```

## Điểm cần LQM xác nhận

- [ ] Final training image size là `1024`.
- [ ] Warm-up epochs là `1.0` hay `2`.
- [ ] Early stopping patience là `8` hay `10`.
- [ ] Inference confidence / max detections / NMS IoU final có khớp main paper không.
- [ ] V4.1 và V4.2 khác nhau chính xác ở mosaic, epoch, dataset hay checkpoint selection.

## Tiêu chí hoàn thành

- [ ] Có class mapping.
- [ ] Có train hyperparameters.
- [ ] Có mô tả ensemble.
- [ ] Không mô tả quá sâu kiến trúc DocLayout-YOLO nếu không cần thiết.

---

# Section 4 -- HPA OCR Training Details -- HVTB

## Mục tiêu

Mô tả nhánh TrOCR xử lý `handwritten`, `printed`, và `annotation`: data preparation, tokenizer adaptation, curriculum training, typed decoding, và validation protocol.

## Output cần nộp

- 4-6 đoạn tiếng Anh.
- Bảng HPA curriculum.
- Bảng type-specific decoding.
- Đoạn nói rõ checkpoint selection theo typed validation CER.

## Nội dung bắt buộc

- Cách chuẩn bị HPA crops cho finetune.
- Base model: Cyrillic HTR / TrOCR-style checkpoint.
- HPA types: `handwritten`, `printed`, `annotation`.
- Tokenizer adaptation cho Ukrainian-specific characters.
- Three-stage curriculum:
  1. Silver warm-up.
  2. Gold fine-tuning.
  3. Gold recovery với learning rate thấp.
- Checkpoint selection theo typed validation CER.
- Type-specific decoding budgets.

## Bảng curriculum

```latex
\begin{table}[t]
\centering
\caption{HPA OCR curriculum.}
\begin{tabular}{llll}
\toprule
Stage & Data & Epochs & Learning rate \\
\midrule
Silver warm-up & Silver HPA crops & 1 & $3\times10^{-5}$ \\
Gold fine-tuning & Gold HPA crops & 6 & $1\times10^{-5}$ \\
Gold recovery & Gold HPA crops & 2 & $5\times10^{-6}$ \\
\bottomrule
\end{tabular}
\end{table}
```

## Bảng decoding

```latex
\begin{table}[t]
\centering
\caption{Type-specific decoding settings for HPA regions.}
\begin{tabular}{lll}
\toprule
Region type & Branch & Decoding budget \\
\midrule
handwritten & TrOCR HPA & beam search 3, max new tokens 192 \\
printed & TrOCR HPA & beam search 3, max new tokens 128 \\
annotation & TrOCR HPA & beam search 1, max new tokens 16 \\
\bottomrule
\end{tabular}
\end{table}
```

## Gợi ý viết

```latex
\section{HPA OCR Training Details}

The HPA branch handles handwritten, printed, and annotation regions. It is initialized from a Cyrillic HTR checkpoint and adapted to Ukrainian by extending the tokenizer with Ukrainian-specific Cyrillic characters. Training follows a three-stage curriculum: silver warm-up, gold fine-tuning, and low-learning-rate gold recovery. The best checkpoint is selected using typed validation CER. At inference time, each HPA type uses a different decoding budget to reduce over-generation on short annotations while preserving long handwritten lines.
```

## Điểm cần HVTB xác nhận

- [ ] Tên/base checkpoint chính xác.
- [ ] Danh sách ký tự Ukrainian thêm vào tokenizer.
- [ ] Validation split dùng cho HPA là split nào, tỉ lệ bao nhiêu.
- [ ] Metric final nên ghi `0.1005` từ report hay `0.0954` theo main paper; nếu khác, cần chỉ rõ nguồn.

## Tiêu chí hoàn thành

- [ ] Có đủ 3 phase curriculum.
- [ ] Có typed decoding chính xác.
- [ ] Có protocol chọn checkpoint.
- [ ] Có ghi rủi ro annotation/printed ít sample nếu nhắc tới kết quả.

---

# Section 5 -- Qwen3-VL LoRA Training Details -- LHNH

## Mục tiêu

Mô tả nhánh Qwen3-VL dùng cho `formula` và `table`: data preparation, instruction construction, LoRA/QLoRA config, prompt templates, guardrails, và inference settings.

## Output cần nộp

- 4-6 đoạn tiếng Anh.
- Bảng Qwen3-VL LoRA configuration.
- Bảng LoRA hyperparameters nếu có log chính xác.
- Prompt template cho formula và table, có thể rút gọn nhưng phải đủ guardrails chính.

## Nội dung bắt buộc

- Data preparation cho từng stage (stage 2B đổi tên thành stage 3).
- Base model: Qwen3-VL-8B-Instruct.
- Fine-tuning method: LoRA hoặc QLoRA, ghi đúng theo notebook final.
- Quantization: 4-bit NF4 nếu đúng với final pipeline.
- Final routed types: `formula`, `table`.
- Formula output: mathematical/chemical expression, dùng LaTeX khi phù hợp.
- Table output: pipe-separated rows.
- Decoding: greedy, `max_new_tokens = 192` nếu đúng với pipeline final.
- Guardrails: no explanation, no normalization, no solving, no hallucination, no JSON/Markdown wrappers.

## Bảng configuration

```latex
\begin{table}[t]
\centering
\caption{Qwen3-VL LoRA configuration.}
\begin{tabular}{ll}
\toprule
Parameter & Value \\
\midrule
Base model & Qwen3-VL-8B-Instruct \\
Adaptation & LoRA / QLoRA, NEED VERIFY \\
Quantization & 4-bit NF4 \\
Final routed types & formula, table \\
Decoding & Greedy decoding \\
Maximum new tokens & 192 \\
Formula target & Visible mathematical or chemical expression \\
Table target & Pipe-separated rows \\
\bottomrule
\end{tabular}
\end{table}
```

## Bảng LoRA hyperparameters cần điền nếu có log

```latex
\begin{table}[t]
\centering
\caption{LoRA hyperparameters for the Qwen3-VL branch.}
\begin{tabular}{ll}
\toprule
Parameter & Value \\
\midrule
LoRA rank & NEED VERIFY \\
LoRA alpha & NEED VERIFY \\
LoRA dropout & NEED VERIFY \\
Target modules & NEED VERIFY \\
Training precision & NEED VERIFY \\
Batch size & NEED VERIFY \\
Learning rate & NEED VERIFY \\
\bottomrule
\end{tabular}
\end{table}
```

## Gợi ý viết

```latex
\section{Qwen3-VL LoRA Training Details}

The Qwen branch is used for structured crop transcription. We train a LoRA adapter on top of Qwen3-VL-8B-Instruct and use the adapted model only for formula and table regions in the final pipeline. This branch is not used as a full-page parser. Formula prompts ask the model to copy visible mathematical or chemical notation, while table prompts require pipe-separated rows. The prompts explicitly discourage solving, normalization, explanation, and hallucinated completion.
```

## Điểm cần LHNH xác nhận

- [ ] Final adapter lấy từ stage nào: stage1, stage2, hay stage3.
- [ ] LoRA rank/alpha/dropout/target modules chính xác.
- [ ] Training precision và batch size chính xác.
- [ ] Learning rate final theo notebook/log.
- [ ] Prompt final có source-aware hint hay không.
- [ ] Guardrails final có làm giảm CER/invalid rate không; nếu có trade-off thì viết thận trọng.

## Tiêu chí hoàn thành

- [ ] Có mô tả data instruction.
- [ ] Có prompt behavior cho formula/table.
- [ ] Có decoding setting.
- [ ] Không nói Qwen xử lý full-page trong final pipeline.

---

# Section 6 -- Final Inference and Routing -- LHNH

## Mục tiêu

Mô tả pipeline inference cuối cùng một cách rõ ràng và có thể kiểm tra: input discovery, detector ensemble, post-processing, routing, OCR branches, output serialization, và validation.

## Output cần nộp

- 4-6 đoạn tiếng Anh.
- Bảng routing policy.
- Bảng final inference settings.
- Checklist output validation.

## Nội dung bắt buộc

- Input: test images hoặc `metadata.jsonl`.
- Detector ensemble chạy trước.
- Post-processing: NMS, clamping, deduplication, sorting.
- Router theo predicted region type.
- HPA types -> TrOCR.
- Formula/table -> Qwen3-VL.
- Image/graph -> empty text.
- Output sorted theo reading order trước khi serialize.
- `submission.csv` có columns `image`, `regions`.
- Mỗi region có đúng keys `bbox`, `type`, `text`.

## Bảng routing policy

```latex
\begin{table}[t]
\centering
\caption{Routing policy used in the final pipeline.}
\begin{tabular}{ll}
\toprule
Region type & Recognition branch \\
\midrule
handwritten & TrOCR HPA \\
printed & TrOCR HPA \\
annotation & TrOCR HPA \\
formula & Qwen3-VL + LoRA \\
table & Qwen3-VL + LoRA \\
image & Empty text \\
graph & Empty text \\
\bottomrule
\end{tabular}
\end{table}
```

## Bảng inference settings cần verify

```latex
\begin{table}[t]
\centering
\caption{Final inference settings.}
\begin{tabular}{ll}
\toprule
Parameter & Value \\
\midrule
Detector checkpoints & V4.1 + V4.2 ensemble \\
Image size & NEED VERIFY \\
Confidence threshold & NEED VERIFY \\
Max detections per image & NEED VERIFY \\
NMS type & Class-agnostic NMS \\
NMS IoU & NEED VERIFY \\
Deduplication IoU & NEED VERIFY \\
Qwen crop max new tokens & 192 \\
Output sorting & Top-to-bottom, then left-to-right \\
\bottomrule
\end{tabular}
\end{table}
```

## Output validation checklist

```latex
\paragraph{Output validation.}
Before writing the final CSV, the pipeline checks that the number of output rows matches the number of input images, each row contains only the \texttt{image} and \texttt{regions} columns, each \texttt{regions} value is a JSON list, and every region contains exactly \texttt{bbox}, \texttt{type}, and \texttt{text}. Bounding boxes must contain four numeric values, and region types must belong to the seven valid RUKOPYS labels.
```

## Gợi ý viết

```latex
\section{Final Inference and Routing}

At inference time, each page is processed by the detector ensemble. The predicted boxes are merged, filtered by class-agnostic NMS, clamped to image boundaries, deduplicated, and sorted in reading order. Each remaining region is routed according to its predicted type. HPA regions are sent to TrOCR, formula and table regions are sent to Qwen3-VL, and image or graph regions are preserved with empty text. The final output is serialized as a CSV file with one row per image.
```

## Điểm cần LHNH xác nhận

- [ ] Thông số final trong paper và `submission/inference.py` đang khác nhau; cần chọn source of truth trước khi nộp.
- [ ] `image size`: paper hiện ghi 1024, script default đang là 1280.
- [ ] `confidence threshold`: paper hiện ghi 0.001, script default đang là 0.26, một số prompt cũ ghi 0.1.
- [ ] `max_det`: paper hiện ghi 300, script default đang là 200.
- [ ] `NMS IoU`: paper hiện ghi 0.55, script default đang là 0.60.

## Tiêu chí hoàn thành

- [ ] Có routing table.
- [ ] Có output validation.
- [ ] Có note rõ conflict thông số đã được resolve hoặc còn `NEED VERIFY`.
- [ ] Không mô tả nhầm image/graph là OCR text.

---

# Section 7 -- Hardware, Runtime, and Reproducibility -- LQM + HVTB + LHNH

## Mục tiêu

Ghi rõ môi trường chạy để reviewer thấy pipeline thực tế và có thể tái lập ở mức hợp lý. Mỗi người điền phần hardware/runtime cho module mình phụ trách.

## Output cần nộp

- Một bảng environment chung.
- Một đoạn reproducibility checklist.
- Mỗi owner gửi hardware/runtime của branch mình nếu có.

## Nội dung nên có

- Framework chính: PyTorch, Transformers, PEFT, bitsandbytes, DocLayout-YOLO, Ultralytics/OpenCV/Pillow nếu dùng.
- Detector training hardware và runtime nếu biết.
- HPA training hardware và runtime nếu biết.
- Qwen training/inference hardware và memory nếu biết.
- Offline inference: không cần internet runtime.
- Model weights phải có local.
- Dependency list hoặc package chính.

## Bảng environment

```latex
\begin{table}[t]
\centering
\caption{Software and runtime environment.}
\begin{tabular}{ll}
\toprule
Item & Setting \\
\midrule
Deep learning framework & PyTorch \\
OCR/VLM library & Transformers \\
LoRA library & PEFT \\
Quantization & bitsandbytes 4-bit NF4 \\
Detector framework & DocLayout-YOLO / Ultralytics-compatible backend \\
Image processing & OpenCV / Pillow \\
Runtime mode & Offline inference \\
\bottomrule
\end{tabular}
\end{table}
```

## Reproducibility checklist

```latex
\paragraph{Reproducibility checklist.}
The final inference package contains the detector checkpoints, TrOCR model, Qwen3-VL base model, LoRA adapter, tokenizer/processor files, inference script, dependency list, and output validation logic. Runtime inference does not require external network access.
```

## Phân việc chi tiết

- LQM: điền detector training hardware, GPU, runtime, DocLayout-YOLO dependency.
- HVTB: điền HPA TrOCR training hardware, runtime, tokenizer/model files.
- LHNH: điền Qwen LoRA training/inference hardware, quantization, memory nếu có.

## Tiêu chí hoàn thành

- [ ] Không có private path.
- [ ] Không có link external private.
- [ ] Có nói rõ offline inference.
- [ ] Dependency khớp package thực tế.

---

# Section 8 -- Anonymization Checklist -- NQH

## Mục tiêu

Đảm bảo supplementary và package đi kèm không làm lộ danh tính nhóm/tác giả.

## Output cần nộp

- Checklist đã tick trước khi submit.
- Note các file cần sửa nếu phát hiện thông tin nhạy cảm.

## Checklist trước khi nộp

- [ ] Không có tên thành viên.
- [ ] Không có tên team nếu tên đó làm lộ danh tính.
- [ ] Không có email cá nhân.
- [ ] Không có Kaggle username.
- [ ] Không có GitHub username.
- [ ] Không có đường dẫn máy cá nhân như `/home/name/...` hoặc `C:\Users\name\...`.
- [ ] Không có link Google Drive, Kaggle private dataset, HuggingFace private repo.
- [ ] Không có comment trong code/log chứa tên người.
- [ ] Không có screenshot notebook hiển thị account hoặc path cá nhân.
- [ ] Không có "revised paper" hoặc nội dung sửa main paper sau deadline.
- [ ] Không có new result từ version method cải tiến sau deadline.

## Gợi ý viết

```latex
\section{Anonymization Checklist}

All implementation details in this supplementary file are anonymized. We remove personal names, usernames, private paths, and external links that would expand the submission content. Any code snippets are included only to clarify the implementation and do not identify the authors.
```

## Tiêu chí hoàn thành

- [ ] Đã rà supplementary `.tex`/`.pdf`.
- [ ] Đã rà code/config/log nếu đưa vào package.
- [ ] Đã rà hình/screenshot nếu có.
- [ ] Đã rà các đường dẫn trong README/package.

