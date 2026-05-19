# 📝 Báo cáo ngữ cảnh dự án (Project Context)

## 1. 🎯 Tóm tắt dự án (Project Overview)
- **Muc tieu:** Xay dung pipeline End-to-End Document Understanding cho Kaggle Handwritten to Data (RUKOPYS), dau ra la submission CSV gom bbox + type + text cho tai lieu viet tay tieng Ukraina.
- **Muc tieu phu:** Chuan bi huong nghien cuu VLM + CoT theo lo trinh cache-first, ablation ladder, routing/refine va hau xu ly.
- **Trang thai hien tai:** Dang o Phase B2. Frozen validation manifest v1 da pass gate; B2 baseline anchor `b2_stage2_gold__crop-none__prompt-v1__val-v1` da co cache/score va da lock vao `artifacts/ablations/ablation_results.csv`.
- **Huong dang debug tren VM:** Chay lai B2 `.py only` tren RTX A6000 48GB de xac nhan runtime; dang uu tien FP16 + batch inference, hien da tang local VM batch size len 24 de smoke/full-run.

## 2. 🛠️ Công nghệ đã sử dụng (Tech Stack)
- **Frontend/Client:** Khong co frontend; workflow qua Kaggle Notebook, script Python va VM SSH.
- **Backend/Server:** Khong co server rieng; pipeline batch offline.
- **Database:** Khong dung DB; du lieu o JSONL/CSV.
- **Thu vien/chay may:** Python, PyTorch, Transformers, TRL/SFTTrainer, PEFT LoRA/QLoRA, bitsandbytes khi can 4-bit fallback, qwen-vl-utils, Hugging Face Hub, PIL, pandas, multiprocessing.
- **Model chinh:** Qwen3-VL-8B cho detection+transcription end-to-end; Qwen2.5-1.5B-Instruct cho spell-check text hau xu ly.
- **Runtime B2 tren A6000:** Uu tien FP16 (`load_in_4bit=false`) va batch page generation; batch 24 dang duoc test do VRAM con du, 4-bit chi la fallback khi OOM/GPU nho.
- **Metric:** IoU matching, class accuracy, CER/PageCER, text normalization theo official-compatible metric.

## 3. ✅ Công việc đã hoàn thành (Completed Tasks)
- [x] Phan tich rule cham diem va format submission chinh thuc: bbox, type, text; JSON trong cot `regions`.
- [x] Xay baseline zero-shot Qwen3-VL va submission tham chieu.
- [x] Xay script chuan bi train/val VLM tu metadata (`prepare_vlm_dataset.py`).
- [x] Xay notebook Stage 1 silver warm-up: page JSON + crop OCR samples, LoRA/QLoRA, checkpoint va prompt config.
- [x] Xay notebook Stage 2 gold fine-tune: load Stage 1 adapter, train tiep tren train split, luu validation records.
- [x] Xay notebook inference/submission hai-pass: page JSON -> crop OCR, checkpoint theo GPU, resume partial CSV.
- [x] Xay script train Qwen3-VL LoRA/QLoRA voi custom callback log va co che giam OOM (`train_qwen_lora.py`).
- [x] Xay script inference fine-tuned da GPU, robust JSON parse, bbox rescale ve anh goc, sort reading order (`kaggle_finetuned_inference.py`).
- [x] Xay hau xu ly spell-check bang LLM nho (`spell_check_submission.py`).
- [x] Tao `phaseB2/` cho B2 baseline `.py only`: readiness checker, runtime config template, parser/normalizer, metric adapter va cache writer.
- [x] Tao frozen validation manifest v1: 159 images, 3129 regions, gate pass, checksum locked.
- [x] Chay va lock B2 baseline anchor vao `artifacts/ablations/ablation_results.csv`: total_score `0.7860404803781254`, detection_f1 `0.7808241668034807`, class_acc `0.9697224558452481`, region_cer `0.26224961652658973`, page_cer `0.20178876495336442`.
- [x] Cap nhat artifact policy cho Phase B2: dataset, Qwen3-VL base model va `qwen3vl_rukopys_lora_final` la external artifacts; khong commit trong repo.
- [x] Them runtime diagnostics cho B2 runner: log model load mode, per-image/batch progress, generate start/done, parse ok/fail va smoke flush.
- [x] Them A6000 FP16 profile va batch inference cho `run_b2_baseline_cache.py`; batch size 2/4/16 smoke da chay duoc, batch 16 dat khoang `1071s/16 pages` nhung phat hien bad case runaway; local VM dang thu batch 24.

## 4. 🐛 Những lỗi đã khắc phục (Fixed Bugs)
- **Bug 1: Sai/khong nhat quan bbox format** -> **Fix:** Chuan hoa ve `[x1, y1, x2, y2]`, ho tro normalized, pixel va 0-1000 grid.
- **Bug 2: OOM khi train/inference tren Kaggle T4** -> **Fix:** Dung 4-bit quantization, gradient checkpointing, gioi han max pixels va don GPU cache dinh ky.
- **Bug 3: Crash do bfloat16 tren T4** -> **Fix:** Ep dtype ve float16/float32 cho model va LoRA trainable params.
- **Bug 4: Output model khong phai JSON sach** -> **Fix:** Robust parsing: strip code fences, greedy JSON array extract va fallback object regex.
- **Bug 5: Lech reading order lam giam PageCER** -> **Fix:** Sort regions theo y/x truoc khi ghi output.
- **Bug 6: Kho resume inference da GPU** -> **Fix:** Checkpoint theo GPU, merge partial CSV va cho phep resume tu file co san.
- **Bug 7: GitHub tu choi push do LoRA `adapter_model.safetensors` >100MB** -> **Fix:** Go tracking adapter, them ignore/policy, luu adapter tren Hugging Face/cloud hoac `/mnt/models`.
- **Bug 8: Smoke test nhin nhu treo sau khi load model** -> **Fix:** Them log quanh process vision, processor encode, tensor move va `model.generate`; xac nhan bottleneck la autoregressive generation, khong phai path/model load.
- **Bug 9: 4-bit tren A6000 khong toi uu toc do** -> **Fix:** Doi B2 VM profile sang FP16 (`load_in_4bit=false`), giu 4-bit la fallback.
- **Bug 10: Full baseline run bi chan do output cache da ton tai** -> **Fix:** Dung `--resume` de chay tiep tu `validation_predictions.csv`, hoac `--overwrite`/output-dir rieng neu muon chay lai tu dau.

## 5. 🚀 Công việc tiếp theo (Current / Pending Tasks)
- [ ] Smoke batch 24 voi output-dir rieng; neu pass thi chay full B2 `.py only` bang batch 24, neu batch keo qua lau/OOM thi giam ve 16/8 va `--resume`.
- [ ] So sanh runtime/page giua batch size 4, 8, 16, 24; khong mac dinh batch lon nhat vi batch bi keo boi sample dai/runaway.
- [ ] Them post-process guardrail cho bad cases: loc bbox degenerate/out-of-image, duplicate/repetition text, region count suspicious va raw JSON incomplete flags.
- [ ] Dieu tra annotation inconsistency `image` vs `graph`: README dinh nghia graph la chart/plot, nhung GT hien co nhieu graph-like regions gan `type=image`; metric phai bam nhan GT thuc te.
- [ ] Thiet ke `prompt-v2-diagram-aware` ablation: gom graph/chart/axis/plot thanh `graph` hoac `image` theo policy dataset, khong OCR tung label ben trong do thi.
- [ ] Tao `risk_labels_cache.parquet` + routing curve re: rule/entropy/disagreement, bao gom flag diagram-heavy/runaway.
- [ ] Tao `refine_candidates.jsonl`, test one-step zoom/refine tren subset; chi mo reasoning/SFT neu co gain.
- [ ] Chuan hoa reading-order + post-processing ablation va so sanh PageCER.
- [ ] Cap nhat validation checklist cho CSV/JSON parse theo submission contract va raw-output quality contract.

## 6. 📂 Cấu trúc thư mục cốt lõi (Core Structure)
- `baseline/`: Baseline zero-shot va ket qua submit tham chieu.
- `finetune/`: Notebook Stage 1/2, inference/submission hai-pass, train LoRA, spell-check va ket qua fine-tune.
- `phaseB2/`: Runner `.py only` cho B2 baseline cache, readiness check, config template, metric/cache helpers, FP16/batch runtime diagnostics.
- `artifacts/manifests/`: Frozen validation manifest v1, config va gate report.
- `artifacts/baseline_predictions_cache/`: Cache predictions/raw outputs/scores cho B2 baseline va smoke runs.
- `artifacts/ablations/ablation_results.csv`: Bang lock ket qua ablation; da co dong B2 baseline anchor.
- `dataset/`: Metadata cac split (`train`, `test`, `silver`) va sample submission.
- `document/` va `document/plan/`: Ke hoach CoT/VLM, notes thiet ke va ablation ladder.
- `VLM_OCR/`: Pipeline OCR theo crop region + VLM va script chuyen doi dinh dang.
- `official-evaluation-metric-text-normalization.ipynb`: Notebook metric/normalization de bam official-compatible scoring.
- `README.md`, `plan.md`, `report.md`: Mo ta dataset/cuoc thi, ke hoach tong quan va bao cao phan tich.
