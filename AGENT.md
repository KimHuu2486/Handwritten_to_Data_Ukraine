# 📝 Báo cáo ngữ cảnh dự án (Project Context)

## 1. 🎯 Tóm tắt dự án (Project Overview)
- **Muc tieu:** Xay dung pipeline End-to-End Document Understanding cho Kaggle Handwritten to Data (RUKOPYS), dau ra la submission CSV gom bbox + type + text cho tai lieu viet tay tieng Ukraina.
- **Muc tieu phu:** Chuan bi huong nghien cuu VLM + CoT theo lo trinh cache-first, ablation ladder, routing/refine va hau xu ly.
- **Trang thai hien tai:** Dang o Phase B2, dong thoi da chot thiet ke `NewPipeline/pipeline.md` cho pipeline competition-first retrain tu baseline. Frozen validation manifest v1 da pass gate; B2 baseline anchor `b2_stage2_gold__crop-none__prompt-v1__val-v1` da co cache/score va da lock vao `artifacts/ablations/ablation_results.csv`.
- **Huong pipeline moi:** Stage A source-aware layout -> Stage A+ deterministic `page_context_light` -> Stage B type-aware OCR voi `ocr_context_light` -> Stage C weighted risk score -> Stage D multi-view/refine voi `risk_rerank_context_light` -> Stage E metric-aware assembly.
- **Huong dang debug tren VM:** Chay lai B2 `.py only` tren RTX A6000 48GB de xac nhan runtime; batch 24 da chay trong `tmux` nhung CUDA driver `free` giam dan sau moi batch, nen batch 24 chi xem la aggressive/benchmark, khong mac dinh production-safe.

## 2. 🛠️ Công nghệ đã sử dụng (Tech Stack)
- **Frontend/Client:** Khong co frontend; workflow qua Kaggle Notebook, script Python va VM SSH.
- **Backend/Server:** Khong co server rieng; pipeline batch offline.
- **Database:** Khong dung DB; du lieu o JSONL/CSV.
- **Thu vien/chay may:** Python, PyTorch, Transformers, TRL/SFTTrainer, PEFT LoRA/QLoRA, bitsandbytes khi can 4-bit fallback, qwen-vl-utils, Hugging Face Hub, PIL, pandas, multiprocessing.
- **Model chinh:** Qwen3-VL-8B cho detection+transcription end-to-end; Qwen2.5-1.5B-Instruct cho spell-check text hau xu ly.
- **Runtime B2 tren A6000:** Uu tien FP16 (`load_in_4bit=false`) va batch page generation; batch 16/12 la fallback on dinh hon, batch 24 chi dung khi chap nhan rui ro VRAM do CUDA workspace/free-memory drift; 4-bit chi la fallback khi OOM/GPU nho.
- **VM operation:** Dung `tmux` de giu job khi mat SSH/internet; log baseline ghi qua `tee` vao `logs/`.
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
- [x] Them A6000 FP16 profile va batch inference cho `run_b2_baseline_cache.py`; batch size 2/4/16 smoke da chay duoc, batch 16 dat khoang `1071s/16 pages`, batch 24 co throughput hon nhung bi sample dai/runaway keo batch len khoang `800-1140s/batch`.
- [x] Cai `tmux` tren VM va chuyen cach chay final baseline sang tmux session `baseline` de job tiep tuc khi mat ket noi.
- [x] Them CUDA memory diagnostics va cleanup sau batch (`gc.collect()` + `torch.cuda.empty_cache()`); log phan biet `allocated/reserved` cua PyTorch voi `free` cua CUDA driver.
- [x] Tao/cap nhat `NewPipeline/pipeline.md` thanh ban pipeline competition-first, tach layout/OCR va bo sung Stage E metric-aware postprocess.
- [x] Chot prompt hierarchy: Stage A prompt theo `source`, Stage B/D prompt theo `type` kem source hint ngan; khong dung source prompt day du cho crop OCR.
- [x] Bo sung Stage A rule chung: bbox grid 0-1000, granularity theo schema, type decision rules, chong merge line/split table sai, detect small meaningful regions.
- [x] Doi Stage A+ sang deterministic `page_context_light` build bang code tu output Stage A; khong train model sinh context phuc tap.
- [x] Tach `page_context_light` thanh `ocr_context_light` cho Stage B first-pass OCR va `risk_rerank_context_light` cho Stage C/D; khong dua `text_draft` vao prompt OCR mac dinh.
- [x] Chot Stage A ablation bat buoc: A `page_layout_only` la default competition v1; B `page_layout_with_text_draft` chi dung neu validation cho thay co loi.
- [x] Nang cap prompt `formula` va `table`: giu math/logic/matrix/determinant/set/chemistry, giu row/cell/empty-cell cua table, cam solve/simplify/infer missing cells.
- [x] Doi Stage C tu OR-rule high-risk sang weighted `risk_score` + threshold/top-K per page + hard override + budget cap.
- [x] Lam ro contextual OCR: Stage B dung `ocr_context_light` mac dinh; page thumbnail chi la visual fallback cho hard examples/high-risk regions.
- [x] Lam ro Stage D multi-view refinement: original crop, zoom crop, expanded context crop, optional page thumbnail + `risk_rerank_context_light`, kem guardrail chong hallucination tu context.

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
- **Bug 11: Batch 24 tren A6000 lam CUDA `free` giam dan du `allocated/reserved` da ve baseline** -> **Fix/Mitigation:** Xac dinh khong giong PyTorch tensor leak truc tiep; kha nang cao la CUDA workspace/cache/fragmentation khi generate dai. Da them cleanup/diagnostics; fallback thuc dung la batch 16/12 neu can rerun on dinh.
- **Bug 12: Runaway generation cham token cap va lam batch rat cham** -> **Fix/Mitigation:** Log da xac nhan nhieu batch cham `output_tokens_max=3072/4096`; can quality guard/postprocess cho repetition, raw JSON incomplete va region count suspicious truoc khi coi output la baseline sach.
- **Doc issue 13: Prompt theo source bi dat sai vao Stage B** -> **Fix:** Chuyen source-aware prompt ve Stage A; Stage B chi giu type-aware OCR prompt + source hint ngan.
- **Doc issue 14: Stage A schema mau thuan ve `text_draft`** -> **Fix:** Tach ro `page_layout_only` chi co bbox/type va `page_layout_with_text_draft` co them `text_draft`.
- **Doc issue 15: Markdown table prompt bi vo do ky tu pipe trong table instruction** -> **Fix:** Escape pipe trong vi du `A\|\|C`.
- **Doc issue 16: Page thumbnail bi hieu nhu context mac dinh cho Stage B/D** -> **Fix:** Them Stage A+ deterministic `page_context_light`; thumbnail chi con la visual fallback cho high-risk.
- **Doc issue 17: Model-sinh `page_context` khong co ground truth va co the hallucinate** -> **Fix:** Bo y tuong train context day du; build `page_context_light` bang rule geometry sau Stage A.
- **Doc issue 18: `text_draft` gay mau thuan voi default `page_layout_only`** -> **Fix:** Chot ablation A/B; `text_draft` chi bat trong ablation B va khong phai default.
- **Doc issue 19: `text_draft` co nguy co anchor sai Stage B OCR** -> **Fix:** Tach `ocr_context_light` khong chua draft; draft chi vao `risk_rerank_context_light` cho risk/rerank sau OCR.
- **Doc issue 20: OR-rule high-risk refine qua nhieu vung** -> **Fix:** Doi sang weighted `risk_score`, refine theo threshold/top-K/hard override va `hard_cap`.

## 5. 🚀 Công việc tiếp theo (Current / Pending Tasks)
- [ ] Sau khi B2 VM run ket thuc, verify `validation_predictions.csv`, `validation_raw_outputs.jsonl`, row count 159, score summary, runtime summary va checksum truoc khi dung cho Phase C.
- [ ] Neu can rerun final B2, uu tien batch 16/12 de on dinh; batch 24 chi dung cho benchmark/aggressive run va phai monitor CUDA `free` sau moi batch.
- [ ] So sanh runtime/page giua batch size 4, 8, 16, 24; khong mac dinh batch lon nhat vi batch bi keo boi sample dai/runaway.
- [ ] Them post-process guardrail cho bad cases: loc bbox degenerate/out-of-image, duplicate/repetition text, region count suspicious va raw JSON incomplete flags.
- [ ] Dieu tra annotation inconsistency `image` vs `graph`: README dinh nghia graph la chart/plot, nhung GT hien co nhieu graph-like regions gan `type=image`; metric phai bam nhan GT thuc te.
- [ ] Thiet ke `prompt-v2-diagram-aware` ablation: gom graph/chart/axis/plot thanh `graph` hoac `image` theo policy dataset, khong OCR tung label ben trong do thi.
- [ ] Tao `risk_labels_cache.parquet` + routing curve re: rule/entropy/disagreement, bao gom flag diagram-heavy/runaway.
- [ ] Tao `refine_candidates.jsonl`, test one-step zoom/refine tren subset; chi mo reasoning/SFT neu co gain.
- [ ] Chuan hoa reading-order + post-processing ablation va so sanh PageCER.
- [ ] Cap nhat validation checklist cho CSV/JSON parse theo submission contract va raw-output quality contract.
- [ ] Chuyen `NewPipeline/pipeline.md` thanh notebook/script retrain v2: Stage 1 silver + Stage 2 gold multitask voi `page_layout_only`, `page_layout_with_text_draft`, `crop_ocr`, `crop_with_page_context`.
- [ ] Cap nhat inference v2: source-aware layout, deterministic `page_context_light`, `ocr_context_light`, type-aware crop OCR, weighted risk score, multi-view refinement va Stage E assembly.
- [ ] Them validation/ablation cho prompt v2: layout-only vs layout+draft, source/type breakdown, formula/table oversampling, contextual OCR vs crop-only, `page_context_light` vs optional thumbnail.
- [ ] Thiet ke data builder cho `page_context_light`, `ocr_context_light`, `risk_rerank_context_light` va region ids; dam bao khong dua `text_draft` vao Stage B first-pass OCR.

## 6. 📂 Cấu trúc thư mục cốt lõi (Core Structure)
- `baseline/`: Baseline zero-shot va ket qua submit tham chieu.
- `finetune/`: Notebook Stage 1/2, inference/submission hai-pass, train LoRA, spell-check va ket qua fine-tune.
- `phaseB2/`: Runner `.py only` cho B2 baseline cache, readiness check, config template, metric/cache helpers, FP16/batch runtime diagnostics.
- `NewPipeline/pipeline.md`: Thiet ke pipeline competition-first moi voi Stage A/A+/B/C/D/E, prompt hierarchy, `page_context_light`, weighted risk score va refinement policy.
- `artifacts/manifests/`: Frozen validation manifest v1, config va gate report.
- `artifacts/baseline_predictions_cache/`: Cache predictions/raw outputs/scores cho B2 baseline va smoke runs.
- `artifacts/ablations/ablation_results.csv`: Bang lock ket qua ablation; da co dong B2 baseline anchor.
- `dataset/`: Metadata cac split (`train`, `test`, `silver`) va sample submission.
- `document/` va `document/plan/`: Ke hoach CoT/VLM, notes thiet ke va ablation ladder.
- `VLM_OCR/`: Pipeline OCR theo crop region + VLM va script chuyen doi dinh dang.
- `official-evaluation-metric-text-normalization.ipynb`: Notebook metric/normalization de bam official-compatible scoring.
- `README.md`, `plan.md`, `report.md`: Mo ta dataset/cuoc thi, ke hoach tong quan va bao cao phan tich.
