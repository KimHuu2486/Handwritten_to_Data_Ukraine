# VLM-R3 + HALP Notes

## What Changed

This folder is an isolated upgrade of `finetune/yolo_qwen_end_to_end/`. The original notebooks are untouched.

The new pipeline keeps the strong part of your current system:

```text
DocLayout-YOLO bbox -> Qwen3-VL crop OCR -> submission JSON
```

Then it adds three mechanisms. By default submit still uses YOLO as the detector, matching the stronger `yolo_qwen_end_to_end` direction you observed.

```text
YOLO bbox -> Qwen adapter trained with end-to-end page_json curriculum -> HALP risk score -> VLM-R3 crop/zoom/context refinement -> final OCR or [illegible]
```

## Paper Mapping

- HALP: the Stage 2 notebook can train `halp_probe.pt`, a lightweight MLP on pre-generation hidden states. The submit notebook loads it when present and falls back to heuristic risk routing when it is absent.
- VLM-R3: training adds `vlm_r3_refine` action-imitation samples for formula/table regions. Inference weaves multiple views into the same Qwen request: normal crop, zoomed crop, and wider context crop.
- End-to-end Qwen: Stage 1/2 now repeat `page_json` samples with `END_TO_END_PAGE_REPEAT`. Submit keeps YOLO as the default detector, then uses the end-to-end-trained adapter for crop OCR/refinement. This preserves the empirical behavior you observed: Qwen trained end-to-end can improve OCR even when YOLO supplies the boxes.
- Optional Qwen page_json ensemble: `USE_QWEN_PAGE_JSON` exists for ablation, but defaults to `False`.
- `[illegible]`: used as the abstention marker when all candidates are empty, invalid, or low quality.

## Recommended Run Order

1. Run `rukopys_qwen3vl_stage1_silver_finetune_vlm_r3_halp.ipynb`.
2. Save `/kaggle/working/qwen3vl_rukopys_stage1_silver_vlm_r3_halp/qwen3vl_silver_lora_final` as a Kaggle dataset.
3. Run `rukopys_qwen3vl_stage2_gold_finetune_vlm_r3_halp.ipynb`.
4. Let the last HALP probe cell run if you want the paper-faithful route. It saves:

```text
qwen3vl_rukopys_lora_final/halp_probe.pt
qwen3vl_rukopys_lora_final/halp_probe_config.json
```

5. Save `/kaggle/working/qwen3vl_rukopys_stage2_gold_vlm_r3_halp/qwen3vl_rukopys_lora_final` as a Kaggle dataset.
6. Run `rukopys_yolo_qwen3vl_vlm_r3_halp_submit.ipynb`.

The Stage 2 and submit notebooks intentionally search only the VLM-R3+HALP adapter paths. If those adapters are not mounted, they fail early instead of silently using an older `hybrid_prompt_v2` adapter.

## Important Knobs

- `VLM_R3_FORMULA_TABLE_REPEAT`: repeats formula/table refinement samples during SFT. Increase if formula/table remain weak.
- `END_TO_END_PAGE_REPEAT`: repeats full-page bbox+OCR samples. Increase if the Qwen end-to-end adapter is stronger than crop-only OCR.
- `USE_QWEN_PAGE_JSON`: optional full-page Qwen detection in submit. Default is `False` so submit follows YOLO -> Qwen OCR/refine.
- `QWEN_YOLO_MERGE_IOU`: overlap threshold for merging Qwen page regions into YOLO regions.
- `RUN_HALP_PROBE_TRAIN`: set `False` in Stage 2 if the probe cell is too slow.
- `HALP_PROBE_LABEL_MODE`: use `generation_cer` for serious HALP-style labels, or `risk_label` for a faster proxy.
- `HALP_RISK_THRESHOLD`: lower routes more regions through VLM-R3 refinement.
- `VLM_R3_MAX_REFINEMENT_PASSES`: number of candidate prompts considered for a high-risk region.
- `MAX_PIXELS_TABLE`: increase when a whole-table bbox is correct but Qwen returns blank.

## Formula/Table Focus

Formula and table regions are forced into the high-risk route in inference:

```python
VLM_R3_FORCE_TYPES = {"formula", "table"}
```

This is deliberate because your current largest failure mode is a correct bbox with empty or incorrect Qwen transcription. The new route gives Qwen more visual evidence and then chooses among candidates using deterministic checks for math symbols, table separators, non-empty output, and non-explanatory text.

## Fast Debug Mode

For a quick validation run in the submit notebook:

```python
RUN_SPLIT = "validation"
TEST_MODE = True
USE_HALP_PROBE = False
USE_QWEN_PAGE_JSON = False
```

Then re-enable `USE_HALP_PROBE = True` after confirming the VLM-R3 refinement path runs.
