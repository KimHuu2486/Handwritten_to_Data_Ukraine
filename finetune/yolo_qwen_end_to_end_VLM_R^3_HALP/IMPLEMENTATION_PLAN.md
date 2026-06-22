# VLM-R3 HALP Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a paper-faithful VLM-R3 + HALP upgrade for the YOLO + Qwen3-VL OCR pipeline, with special focus on formula and table failures.

**Architecture:** Keep DocLayout-YOLO as the submit-time region proposal backbone while preserving Qwen3-VL's end-to-end full-page training curriculum. Teach Qwen3-VL to handle interleaved region refinement actions for hard formula/table crops. Add HALP-style pre-generation risk scoring so high-risk regions are routed through iterative crop/zoom/refine before the final transcription, and use `[illegible]` as the abstention marker.

**Tech Stack:** Kaggle notebooks, Python, PyTorch, Transformers, PEFT/LoRA, TRL SFTTrainer, qwen-vl-utils, DocLayout-YOLO, PIL.

**Status:** Implemented in this folder. The checklists below remain as the original execution plan and verification guide.

---

## File Structure

- Create: `finetune/yolo_qwen_end_to_end_VLM_R^3_HALP/rukopys_qwen3vl_stage1_silver_finetune_vlm_r3_halp.ipynb`
  - Stage 1 silver warm-up with the original page/crop tasks plus VLM-R3 action-imitation samples and HALP binary risk labels.
- Create: `finetune/yolo_qwen_end_to_end_VLM_R^3_HALP/rukopys_qwen3vl_stage2_gold_finetune_vlm_r3_halp.ipynb`
  - Stage 2 gold fine-tune continuing from Stage 1, with stronger formula/table sampling, VLIR-style refinement targets, and saved validation records for probe training.
- Create: `finetune/yolo_qwen_end_to_end_VLM_R^3_HALP/rukopys_yolo_qwen3vl_vlm_r3_halp_submit.ipynb`
  - Inference notebook with YOLO detections as the default boxes, optional Qwen full-page `page_json` ablation, HALP-style pre-generation scoring, multi-pass VLM-R3 crop/zoom, formula/table-specific prompts, and `[illegible]` fallback.
- Create: `finetune/yolo_qwen_end_to_end_VLM_R^3_HALP/IMPLEMENTATION_NOTES.md`
  - Practical notes on how this maps to HALP and VLM-R3, expected training order, and knobs to tune.
- Modify: none outside `finetune/yolo_qwen_end_to_end_VLM_R^3_HALP/`.

## Chunk 1: Notebook Copy And Identity

### Task 1: Create isolated notebook copies

**Files:**
- Create: `finetune/yolo_qwen_end_to_end_VLM_R^3_HALP/rukopys_qwen3vl_stage1_silver_finetune_vlm_r3_halp.ipynb`
- Create: `finetune/yolo_qwen_end_to_end_VLM_R^3_HALP/rukopys_qwen3vl_stage2_gold_finetune_vlm_r3_halp.ipynb`
- Create: `finetune/yolo_qwen_end_to_end_VLM_R^3_HALP/rukopys_yolo_qwen3vl_vlm_r3_halp_submit.ipynb`

- [ ] **Step 1: Copy original notebooks**

Copy from:

```text
finetune/yolo_qwen_end_to_end/rukopys_qwen3vl_stage1_silver_finetune_hybrid_prompt_v2.ipynb
finetune/yolo_qwen_end_to_end/rukopys_qwen3vl_stage2_gold_finetune_hybrid_prompt_v2.ipynb
finetune/yolo_qwen_end_to_end/rukopys_yolo_qwen3vl_hybrid_submit.ipynb
```

- [ ] **Step 2: Rename notebook titles and output directories**

Expected title suffix: `VLM-R3 + HALP`.

Expected output roots:

```python
OUTPUT_DIR = Path("/kaggle/working/qwen3vl_rukopys_stage1_silver_vlm_r3_halp")
OUTPUT_DIR = Path("/kaggle/working/qwen3vl_rukopys_stage2_gold_vlm_r3_halp")
OUTPUT_CSV = "submission_vlm_r3_halp.csv"
```

- [ ] **Step 3: Validate JSON**

Run:

```powershell
Get-ChildItem 'finetune/yolo_qwen_end_to_end_VLM_R^3_HALP' -Filter '*.ipynb' |
  ForEach-Object { Get-Content $_.FullName -Raw | ConvertFrom-Json | Out-Null; $_.Name }
```

Expected: all three notebook names print without JSON parse errors.

## Chunk 2: VLM-R3 Supervision

### Task 2: Add region action samples

**Files:**
- Modify: `finetune/yolo_qwen_end_to_end_VLM_R^3_HALP/rukopys_qwen3vl_stage1_silver_finetune_vlm_r3_halp.ipynb`
- Modify: `finetune/yolo_qwen_end_to_end_VLM_R^3_HALP/rukopys_qwen3vl_stage2_gold_finetune_vlm_r3_halp.ipynb`

- [ ] **Step 1: Add prompts for refinement actions**

Add `VLM_R3_ACTION_PROMPT` that requires a compact JSON action:

```json
{"action":"zoom","bbox_2d":[x1,y1,x2,y2],"reason":"formula/table unreadable"}
```

- [ ] **Step 2: Add formula/table-heavy sample builder**

For every `formula` or `table` region with valid text, create an additional `vlm_r3_refine` sample whose answer first emits a zoom action and then the final transcription.

- [ ] **Step 3: Weight hard region types**

Use repeat factors so formula/table samples appear more often than ordinary handwritten samples.

- [ ] **Step 3b: Preserve end-to-end learning**

Repeat `page_json` samples with `END_TO_END_PAGE_REPEAT` so the adapter keeps the full-page bbox+OCR behavior that tested better than OCR-only fine-tuning.

- [ ] **Step 4: Run notebook smoke build**

Expected: dataset print includes counts for `vlm_r3_refine`, and total samples increase.

## Chunk 3: HALP Training Hook

### Task 3: Add pre-generation risk labels

**Files:**
- Modify: `finetune/yolo_qwen_end_to_end_VLM_R^3_HALP/rukopys_qwen3vl_stage1_silver_finetune_vlm_r3_halp.ipynb`
- Modify: `finetune/yolo_qwen_end_to_end_VLM_R^3_HALP/rukopys_qwen3vl_stage2_gold_finetune_vlm_r3_halp.ipynb`

- [ ] **Step 1: Add risk label helper**

Compute `risk_label=1` for formula/table, very small crops, empty text, or text marked illegible.

- [ ] **Step 2: Preserve labels in Dataset rows**

Each sample should include:

```python
"risk_label": 0 or 1,
"risk_reason": "formula_table" | "empty_or_illegible" | "small_crop" | "normal"
```

- [ ] **Step 3: Save config metadata**

`rukopys_prompt_config.json` should include VLM-R3/HALP settings for inference.

## Chunk 4: VLM-R3 + HALP Inference

### Task 4: Add risk routing and iterative refinement

**Files:**
- Modify: `finetune/yolo_qwen_end_to_end_VLM_R^3_HALP/rukopys_yolo_qwen3vl_vlm_r3_halp_submit.ipynb`

- [ ] **Step 1: Add HALP heuristic scorer**

Before Qwen OCR, score each region using type, size, YOLO confidence, blur, and prior text quality.

- [ ] **Step 1b: Add optional Qwen page_json + YOLO merge ablation**

Keep YOLO as the default submit detector. Optionally run Qwen once on the full page, parse its compact JSON regions, convert 0-1000 bboxes into image pixels, merge overlapping regions into YOLO detections, and keep non-overlapping Qwen regions as recall boosters.

- [ ] **Step 2: Add VLM-R3 zoom scaling**

Use area-ratio zoom:

```python
if r < 0.125: scale = 2.0
elif r >= 0.5: scale = 1.0
else: scale = 2.0 - ((r - 0.125) / 0.375)
```

- [ ] **Step 3: Add multi-pass OCR**

Generate several crop variants for high-risk formula/table regions: normal crop, zoom crop, expanded context crop, and optional table-grid enhanced crop prompt.

- [ ] **Step 4: Add answer arbitration**

Choose the best candidate using deterministic rules: reject empty/JSON/explanation outputs, prefer formula/table outputs with symbols or separators, use `[illegible]` when no candidate passes.

- [ ] **Step 5: Run validation mode**

Set:

```python
RUN_SPLIT = "validation"
TEST_MODE = True
```

Expected: output CSV is produced and hard regions contain non-empty text or `[illegible]`.

## Chunk 5: Documentation

### Task 5: Write implementation notes

**Files:**
- Create: `finetune/yolo_qwen_end_to_end_VLM_R^3_HALP/IMPLEMENTATION_NOTES.md`

- [ ] **Step 1: Document the paper mapping**

Mention HALP features/routing and VLM-R3 action/crop/zoom/weaving.

- [ ] **Step 2: Document recommended run order**

Run Stage 1, publish adapter as Kaggle dataset, run Stage 2, publish final adapter, run submit.

- [ ] **Step 3: Document tuning knobs**

List thresholds for `HALP_RISK_THRESHOLD`, formula/table repeat factors, max refinement passes, and pixel budgets.

## Verification

- [ ] All new notebooks parse as valid JSON.
- [ ] No original notebook under `finetune/yolo_qwen_end_to_end/` is modified.
- [ ] New submit notebook has explicit formula/table refinement path.
- [ ] New training notebooks save VLM-R3/HALP config metadata.
