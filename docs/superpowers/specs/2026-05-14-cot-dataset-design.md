# CoT Dataset Generation - Design (Production)

Date: 2026-05-14
Owner: DE + OCR team

## Summary
Generate a Chain-of-Thought (CoT) OCR dataset from the gold train set, focused on hard handwritten regions. The pipeline produces clean, versioned JSONL outputs with QC, retries, and full metadata for training and analysis.

## Goals
- Produce ~5,000 CoT samples from gold train data.
- Prioritize regions with legibility = partially_legible or illegible.
- Ensure outputs are parseable, consistent, and traceable to input regions.
- Provide logs and stats for coverage and QC pass rate.

## Non-goals
- No model training in this pipeline.
- No manual annotation UI or human correction loop.
- No dataset release packaging (handled later).

## Inputs
Each candidate region requires:
- image_path
- image_id
- region_bbox (x1, y1, x2, y2)
- region_type (handwritten, printed, formula, table, annotation)
- ground_truth
- legibility
- source

## Outputs
Primary files:
- cot_samples.jsonl
- cot_failed.jsonl
- cot_stats.json

### Sample schema (cot_samples.jsonl)
{
  "image_id": "uuid_xxx",
  "region_bbox": [x1, y1, x2, y2],
  "region_type": "handwritten",
  "ground_truth": "...",
  "cot_reasoning": "<ambiguous_chars>...</ambiguous_chars>...",
  "uncertainty_words": ["..."],
  "uncertainty_scores": [0.73],
  "model_name": "gpt-4o" ,
  "prompt_version": "cot_v1",
  "created_at": "2026-05-14T00:00:00Z"
}

### Failure schema (cot_failed.jsonl)
{
  "image_id": "uuid_xxx",
  "region_bbox": [x1, y1, x2, y2],
  "error_type": "parse_error|api_error|qc_fail",
  "error_detail": "...",
  "raw_response": "...",
  "model_name": "gpt-4o",
  "prompt_version": "cot_v1",
  "created_at": "2026-05-14T00:00:00Z"
}

## End-to-end pipeline
1) Load gold dataset and filter by region_type + legibility.
2) Crop region from full image using bbox; persist crop metadata.
3) Build prompt (versioned template) with ground_truth.
4) Call vision model API with crop + prompt.
5) Parse tags: ambiguous_chars, visual_analysis, context_clues, reasoning, conclusion.
6) QC checks (auto):
   - conclusion equals ground_truth
   - required tags present
   - no empty reasoning
   - no repeated n-gram loops
7) Derive uncertainty_words/scores (regex or rule-based from reasoning).
8) Write sample to cot_samples.jsonl; failures to cot_failed.jsonl.
9) Update stats: counts, pass rate, source/legibility distribution.

## Prompting and parsing
- Prompt template is versioned (prompt_version).
- The model must return all tags exactly once.
- Parsing fails if any tag is missing or malformed.

## QC rules
- conclusion == ground_truth (exact string match).
- reasoning length >= 20 chars.
- repeated n-gram loop detection (threshold = 5).
- if any rule fails, route to cot_failed.jsonl.

## Error handling and retries
- Retry on API errors with exponential backoff (max 3 retries).
- Timeout per call (configurable, default 60s).
- Preserve raw response for all failures.

## Storage layout
- data/cot/
  - cot_samples.jsonl
  - cot_failed.jsonl
  - cot_stats.json
  - prompts/
    - cot_v1.txt

## Logging and metrics
- Log per-sample latency, model name, token usage.
- cot_stats.json includes:
  - total_candidates
  - total_success
  - total_failed
  - pass_rate
  - distribution_by_source
  - distribution_by_legibility

## Acceptance criteria
- 5,000+ samples in cot_samples.jsonl.
- QC pass rate >= 85%.
- All samples parseable with schema above.
- Stats file generated and consistent with counts.
