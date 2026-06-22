# Ukrainian Curriculum Clean Fine-Tuning

Three Kaggle notebooks for a clean Qwen3-VL OCR curriculum:

1. `00_stage0_ukrainian_glyph_pretrain.ipynb`
   - Excludes images listed in `manifest_labelerror_prederror_or_labelerror_predtrue.csv`.
   - Builds Stage 0 from synthetic Ukrainian/Cyrillic renderings derived from clean train labels plus short clean GT crops.
   - Saves `/kaggle/working/qwen3vl_ukrainian_stage0/qwen3vl_ukrainian_stage0_lora_final`.

2. `01_stage1_clean_region_finetune.ipynb`
   - Continue from Stage 0.
   - Uses only clean `train/metadata.jsonl` ground-truth `regions[].bbox`.
   - Does not use YOLO boxes.

3. `02_stage2_final_format_finetune.ipynb`
   - Continue from Stage 1.
   - Uses clean GT bboxes with lower LR and extra repeats for table/formula/annotation formatting.
   - Does not use YOLO boxes.

4. `03_submission_yolo_qwen3vl_stage2_no_pp.ipynb`
   - Submission notebook copied from the working YOLO + Qwen flow.
   - Loads `qwen3vl_ukrainian_stage2_final_lora_final` from the Stage 2 output.
   - Uses prompt config saved by the Stage 2 adapter.
   - Disables PP-StructureV3/PaddleOCR by default to reduce runtime and installation risk.

On Kaggle, upload/add the bad manifest CSV as an input dataset for every stage. For Stage 1, add the saved Stage 0 output as an input dataset and update `START_LORA_CANDIDATES` if Kaggle gives it a different path. Do the same for Stage 2 with the Stage 1 output.

For submission, add these Kaggle inputs:
- RUKOPYS dataset.
- Qwen3-VL 8B base model, unless internet is enabled.
- YOLO/DocLayout `.pt` weights.
- Stage 2 notebook output containing `qwen3vl_ukrainian_stage2_final_lora_final`.

Regenerate notebooks after editing the builder:

```bash
python finetune/ukrainian_curriculum_clean/_build_ukrainian_curriculum_notebooks.py
python finetune/ukrainian_curriculum_clean/_build_submission_notebook.py
```
