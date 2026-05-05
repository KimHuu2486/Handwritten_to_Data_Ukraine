---
language:
- uk
license: cc-by-nc-sa-4.0
task_categories:
- object-detection
- image-to-text
tags:
- handwriting-recognition
- htr
- ocr
- bounding-box
- ukrainian
- document-analysis
- cyrillic
size_categories:
- 10K<n<100K
pretty_name: "RUKOPYS: Ukrainian Handwritten Text Recognition Dataset"
authors:
- Dmytro Voitekh
- Volodymyr Zmiivskyyi
- Oleksii Molchanovskyi
organizations:
- Ukrainian Catholic University
configs:
- config_name: full
  default: true
  data_files:
  - split: train
    path:
      - "train/metadata.jsonl"
      - "train/images/**"
  - split: silver
    path:
      - "silver/metadata.jsonl"
      - "silver/images/**"
- config_name: gt_only
  data_files:
  - split: train
    path:
      - "train/metadata.jsonl"
      - "train/images/**"
- config_name: test
  data_files:
  - split: test
    path:
      - "test/metadata.jsonl"
      - "test/images/**"
---

# RUKOPYS: Ukrainian Handwritten Text Recognition Dataset

**RUKOPYS** (Ukrainian: *рукопис* — manuscript) is the first large-scale open dataset for Ukrainian handwritten text recognition (HTR). It spans over a century of Ukrainian handwriting — from 1920s archival documents to present-day school homework — and is designed for end-to-end document understanding: region detection, type classification, and text transcription.

Ukrainian is among the largest Slavic languages (45M+ native speakers) yet had no dedicated open HTR dataset prior to RUKOPYS.

> **Competition:** RUKOPYS powers the [Handwritten to Data](https://www.kaggle.com/competitions/handwritten-to-data) challenge on Kaggle (April 16 — June 15, 2026). Submit your HTR model predictions and compete for $7,000 in prizes.

---

## What Makes RUKOPYS Different

Most HTR datasets are built from a single source — one archive, one corpus, one handwriting style. RUKOPYS is deliberately the opposite.

It combines four sources that differ across every dimension that makes handwriting recognition hard:

| Dimension | Range in RUKOPYS |
|-----------|-----------------|
| **Time period** | 1919–1935 (archival pen & ink) → 2020–2025 (modern ballpoint, pencil) |
| **Writers** | School children (grades 5–11), university students, adult citizens |
| **Document type** | Archival state documents, personal dictation sheets, exam papers, homework |
| **Capture method** | Flatbed scanner (archive, university) vs phone camera (dictation, school) |
| **Orthography** | Archaic pre-reform spelling (1920s) → contemporary Ukrainian |
| **Content** | Prose, formulas, chemistry, tables, teacher annotations |

This breadth is intentional. A model trained only on clean archival scans will fail on a phone photo of a student notebook — and vice versa. RUKOPYS is designed so that the models trained on it generalize across real-world variation, not just perform well on a narrow slice of it.

Although RUKOPYS is primarily a handwriting dataset, many pages naturally contain **printed text** alongside handwriting — preprinted headers, textbook excerpts, map captions, teacher stamps, table rulings and so on. These printed regions are annotated with the same schema (`type: printed`) and kept in the dataset deliberately: most downstream use cases (search, digitization, classroom assistive tools) need to process a full page end-to-end, not just the handwritten strokes. Training detectors that must distinguish `handwritten` vs `printed` on mixed-content pages is a first-class task supported by RUKOPYS.

---

## Splits

| Split | Images | GT Regions | `annotation_source` | Description |
|-------|--------|-----------|---------------------|-------------|
| **train** | 1,330 | 25,523 | `annotator` / `volunteer` | Human-annotated — full bboxes + verified transcription |
| **silver** | 8,207 | 161,065 | `auto` | Auto-annotated by Qwen3-VL 8B + Gemini — for self-training |
| **test** | 386 | — (hidden) | — | Images only — submit predictions to the [Kaggle competition](https://www.kaggle.com/competitions/handwritten-to-data) |
| **private benchmark** | 21 | — (hidden until June 15) | — | Held-out set withheld during the competition; published after the online stage closes as a reusable community benchmark |

Use `annotation_source` to distinguish professional-annotator ground truth from volunteer and auto-annotations when combining splits.

### Train Composition by Source & Annotation Source

| Source | Professional | Volunteer | Total |
|--------|--------------|-----------|-------|
| `dictation` | 221 | 138 | 359 |
| `archive` | 90 | 37 | 127 |
| `university` | 136 | 26 | 162 |
| `school` | 398 | 284 | 682 |
| **Total** | **845** | **485** | **1,330** |

Professional annotations were produced by the [Keymakr](https://keymakr.com/) team following a detailed brief. Volunteer annotations were contributed by community members. Both follow the same schema; volunteer data may have slightly higher transcription variance — the `annotation_source` field lets you filter or weight them differently during training.

---

## Data Sources

| Source | ID | Period | Images (train+test) | Description |
|--------|----|--------|---------------------|-------------|
| National Dictation | `dictation` | 2020–2025 | 509 | Phone photos of handwritten Ukrainian National Dictation. One canonical text per year, thousands of unique handwriting styles. |
| State Archive | `archive` | 1919–1935 | 178 | Scanned documents from 12 archival funds of the Central State Archive of Ukraine (ЦДАВО). Pen & ink, archaic orthography. |
| University (KNUTE) | `university` | 2024–2025 | 246 | Scanned student exam work from 5 faculties: text, math formulas, chemistry, tables. |
| School Homework | `school` | 2024–2025 | 782 | Phone photos of school homework (grades 5–11, 20+ subjects) from Opornyi Lyceum s. Zymne (Опорний ліцей с. Зимне) and Mriia volunteer contributors. |

---

## Dataset Structure

```
train/                         # Human-annotated (1,330 images)
  images/{uuid}.jpg
  metadata.jsonl               # bbox + type + language + legibility + text

silver/                        # Auto-annotated (8,210 images)
  images/{uuid}.jpg
  metadata.jsonl               # same schema as train

test/                          # Test images, no annotations (386 images)
  images/{uuid}.jpg
  metadata.jsonl               # file_name, image_width, image_height, source (regions: null)
```

`train` and `silver` share the same schema and can be combined freely with `concatenate_datasets`.

---

## Loading

### With `datasets` (recommended — loads images as PIL, regions as structured fields)

```python
from datasets import load_dataset, concatenate_datasets

ds = load_dataset("UkrainianCatholicUniversity/rukopys")

# Human-annotated train
gt_train = ds["train"]
example = gt_train[0]
print(example["image"])             # PIL Image
print(example["source"])            # "dictation"
print(example["annotation_source"]) # "annotator"
print(example["regions"])           # [{bbox, type, language, legibility, text}, ...]

# Combine GT + silver
full_train = concatenate_datasets([gt_train, ds["silver"]])

# GT-only config (no silver):
ds_gt = load_dataset("UkrainianCatholicUniversity/rukopys", "gt_only")
```

### With `pandas`

```python
import pandas as pd
df_train = pd.read_json("hf://datasets/UkrainianCatholicUniversity/rukopys/train/metadata.jsonl", lines=True)
```

### With `polars`

```python
import polars as pl
df_train = pl.read_ndjson("hf://datasets/UkrainianCatholicUniversity/rukopys/train/metadata.jsonl")
```

### Direct download with `huggingface_hub`

```python
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id="UkrainianCatholicUniversity/rukopys", repo_type="dataset")
# All files under `path` in the original folder structure (train/, silver/, test/)
```

---

## Annotation Schema

Each record in `train` and `silver` has a `regions` field — a list of annotated content regions:

```json
{
  "file_name": "images/abc123.jpg",
  "image_width": 3024,
  "image_height": 4032,
  "source": "dictation",
  "annotation_source": "annotator",
  "regions": [
    {
      "bbox": [134, 766, 3754, 1197],
      "type": "handwritten",
      "language": "uk",
      "legibility": "legible",
      "text": "Спочатку був брехунець. У нього кожного дня: „Клац!""
    }
  ]
}
```

`bbox` format: `[x1, y1, x2, y2]` — pixel coordinates, top-left origin.

### Region Types

| Type | Description | Transcription |
|------|-------------|---------------|
| `handwritten` | Handwritten text line | Exact text, 1 bbox = 1 line |
| `printed` | Printed/typed text line | Exact text, 1 bbox = 1 line |
| `formula` | Standalone math/chemistry expression | LaTeX |
| `table` | Full table | Pipe-separated values |
| `annotation` | Teacher marks, grades, numbering | Short text |
| `image` | Stamps, seals, drawings | Empty |
| `graph` | Charts, plots | Empty |

### Special Text Markers

| Marker | Meaning |
|--------|---------|
| `~~word~~` | Strikethrough text |
| `~~old~~{new}` | Strikethrough with correction |
| `[illegible]` | Unreadable word within a legible line |

### Region Attributes

| Attribute | Values |
|-----------|--------|
| `language` | `uk`, `other` |
| `legibility` | `legible`, `illegible` |
| `annotation_source` | `annotator`, `volunteer`, `auto` |

`annotation_source` values:

| Value | Meaning |
|-------|---------|
| `annotator` | Labeled by [Keymakr](https://keymakr.com/) — professional human annotation service |
| `volunteer` | Labeled by community volunteers; spot-checked for quality |
| `auto` | Auto-generated by the VLM pipeline (silver split only) |

---

## Anti-Leakage Design

| Source | Train | Test | Guarantee |
|--------|-------|------|-----------|
| **Dictation** | Year 2024 | Years 2020, 2022, 2025 | Different canonical texts |
| **Archive** | Archival file set A | Archival file set B | Non-overlapping archival document sets |
| **University** | Exam PDF group A | Exam PDF group B | Different students' exam files |
| **School** | Grades 5, 6, 7, 9, 11 | Grades 8, 10 | Different grade bands |

---

## Silver Split

The `silver` split contains 8,210 auto-annotated images generated by a multi-stage VLM pipeline:

```
Stage 1: Qwen3-VL 8B block detection
Stage 2: Gemini Flash block classification
Stage 3: Qwen3-VL 8B line segmentation within text blocks
Stage 4: Gemini Flash transcription
```

Known limitations: bbox sequence drift on dense text; axis-aligned boxes may clip skewed lines; ~440 archive files contain mixed Ukrainian/Russian text from the 1919–1935 period.

---

## Acknowledgements

Professional annotation was provided by [Keymakr](https://keymakr.com/), a human-in-the-loop data annotation company.

Additional annotations were contributed by volunteers. The full list of contributors will be published shortly. All volunteer annotations underwent spot-checking for quality assurance.

All images were reviewed prior to publication to remove personally identifiable information (PII).

---

## Roadmap

This is the first public release of RUKOPYS. The dataset will grow incrementally — both through additional sources and through expanded coverage of existing ones.

We welcome collaboration from:
- **Annotators** interested in contributing human-verified labels
- **Researchers** working on better automatic annotation approaches (layout analysis, HTR pre-annotation, active learning)

If you'd like to contribute, reach out via the [Kaggle competition forum](https://www.kaggle.com/competitions/handwritten-to-data/discussion) or open an issue on HuggingFace.

---

## Potential Uses

- Fine-tune HTR models on `train`, evaluate on `test` via the [Kaggle competition](https://www.kaggle.com/competitions/handwritten-to-data)
- Pseudo-labeling: GT text for each dictation year is publicly known — use it for text-line alignment
- Self-training / semi-supervised learning with the `silver` split
- Multi-source domain adaptation (modern handwriting → historical documents)

---

## Recommended Approaches

### VLM Fine-tuning
Adapt vision-language models (Gemma 4, Qwen3-VL, LLaMA, etc.) to Ukrainian handwritten text. This is the most direct path to strong results.

### Agentic Recognition Pipelines
Multi-step systems that classify document type and content, then route to specialized strategies — e.g., delegating formulas, tables, or printed text to dedicated models.

### Retrieval-Augmented Recognition (RAR)
Equip your system with external knowledge bases: domain-specific vocabularies, document templates, lexical patterns. Identify the document type, retrieve relevant context, and use it to guide recognition.

### Additional Data Sources
Our dataset alone may not be sufficient to train large models from scratch. We encourage creative use of external datasets, synthetic data generation, and pseudo-labeling to expand training data.

### Portability
Smaller and more efficient solutions are preferred. In production, these models must run on limited hardware. Efficiency per compute unit is a competitive advantage.

### Post-processing & Ensembles
Language model correction, dictionary-based fixing, and multi-model voting at the line or character level are proven ways to improve CER.

---

## Tips & Resources

### Self-training with the silver split
The `silver` split contains auto-annotated images — the same sources and format as `train`. Use it as a noisy-label pretraining stage before fine-tuning on human-verified `train` annotations. A simple curriculum: train on silver first, then fine-tune on gold.

### Pseudo-labeling with dictation ground truth
The National Dictation ground-truth texts are publicly available for each year. Because the canonical text is known, you can skip bbox-level annotation and align the full-page transcription directly to image lines — useful for text-line-level pretraining without human annotation.

### Synthetic data
Generating synthetic Ukrainian handwriting is practical and well-supported:

- [TextRecognitionDataGenerator (TRDG)](https://github.com/Belval/TextRecognitionDataGenerator) — render text with handwriting-style fonts, distortions, custom backgrounds. Supports any language and custom glyph sets.
- [FbSTG](https://github.com/mhlzcu/doc_gen) — tested specifically on historical Cyrillic documents; reduced CER by 24% and WER by 8% in a published evaluation.

For Ukrainian text content there're popular open text corpora like UberText, Kobza etc. For handwriting-style fonts with full Ukrainian character coverage (Ґ, Є, І, Ї) — check Google Fonts and Font Squirrel filtering for Ukrainian support.

### External datasets (permitted external data)
The following open datasets are compatible with competition rules — publicly available, non-commercial use:

| Dataset | Language | Description
| --- | --- | ---
| [HKR](https://github.com/abdoelsayed2016/HKR_Dataset) | Kazakh (Cyrillic) | ~63K sentences, ~200 writers, Nazarbayev University
| IAM Handwriting DB | English | 1,500+ pages, gold standard for Latin HTR benchmarks

### Pretrained checkpoints worth fine-tuning

#### End-to-end document understanding (layout + OCR in one model):

- [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) — strong vision-language model with native document understanding; fits on a single H100 with room for batching. The 72B variant is available for training if you have the compute.
- [Google Gemma 4](https://huggingface.co/collections/google/gemma-4) — open-weight multimodal family: 5B (gemma-4-E2B-it), 8B MoE (gemma-4-E4B-it), 27B (gemma-4-26B-A4B-it), 31B (gemma-4-31B-it). All are vision-capable and fine-tunable; the 5B–27B variants fit comfortably on a single H100.
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — production-grade OCR framework with pretrained detection + recognition pipelines; supports custom Cyrillic fine-tuning and has strong layout analysis tools built in.

#### Text-line recognition (after bbox detection):

- [microsoft/trocr-base-handwritten](https://huggingface.co/microsoft/trocr-base-handwritten) — encoder-decoder baseline for handwritten text lines; fine-tunes well on Cyrillic with modest data.
- [Kansallisarkisto/cyrillic-htr-model](https://huggingface.co/Kansallisarkisto) — trained on 30K+ lines of Cyrillic archival handwriting by the National Archives of Finland.

---

## License

**CC BY-NC-SA 4.0** — Attribution, Non-Commercial, Share-Alike.

- **National Dictation** images: provided under a data sharing agreement for academic research and publication
- **State Archive** (ЦДАВО): provided under a data sharing agreement for academic research and publication
- **KNUTE** and **Opornyi Lyceum s. Zymne (Опорний ліцей с. Зимне)**: provided under data sharing agreements for academic research and publication

---

## Changelog

### v1.3 — 2026-04-22

**Region-type corrections. No image or split changes.**

- **Silver — archive `handwritten` → `printed` reclassification:** 19,028 regions across 1,016 archive images corrected. The original auto-annotation pipeline marked every 1920s archival text region as `handwritten`, including typewritten letterheads, official stamps, and typeset forms. A per-block reclassification pass (Gemini 2.0 Flash, ~16.5K blocks) split handwritten vs printed. Counts for the archive split of silver: handwritten 106,811 → 87,786; printed 0 → 19,046. Bounding boxes and transcriptions are unchanged — only `type` differs.
- **Train — 2 bboxes `pii` → `annotation`:** a stray internal service label leaked into public metadata on 2 archive-volunteer images; remapped to the correct public type `annotation`. Bounding boxes and transcriptions unchanged.
- **Test set unchanged:** same 386 images, same regions, same `solution.csv`.

### v1.2 — 2026-04-22

**Annotation quality pass + image orientation normalization. Train: 25,768 → 25,523 regions (−245). Silver: 8,210 → 8,207 images, 163,081 → 161,065 regions.**

- **Annotation cleanup** (applied across all splits):
  - Text normalization: repeated-character loops (`aaaa…` → `aaa`), n-gram loops, control chars, whitespace.
  - **Type correction:** 755 regions reclassified from `handwritten` → `formula` where content is pure math notation (136 train + 619 silver).
  - Bounding box sanity: inverted coordinates auto-swapped, clamped to image bounds, degenerate boxes dropped.
  - Text cleared on structural region types (`image`, `graph`) per schema.
  - Dedup of overlapping same-text boxes (IoU ≥ 0.5).
- **Image orientation normalized:** rotation baked into pixels and EXIF orientation tags removed on 193 images, so loading is consistent across viewers and libraries.
- **Silver refinements:** 4 images removed due to annotation artifacts (hallucinated column of tiny bboxes with empty text); 7 images re-annotated via the full layout+transcription pipeline where earlier annotations had collapsed.
- **Test set unchanged:** same 386 images, same UUIDs. Kaggle evaluation set stable.

### v1.1 — 2026-04-21

**Train expanded: 770 → 1,330 images (+560), 16,381 → 25,768 regions (+9,387).**

- **School (professional, batch 2):** added 398 professionally-annotated school homework images (grades 5/6/7/9/11). Every image went through a manual QA review — bboxes verified, rotations corrected, EXIF orientation fixed on affected images, PII regions masked.
- **Volunteer batches:** added 167 additional volunteer annotations — 53 dictation, 11 archive, 103 school. Same QA review pipeline applied; rotations and image transforms from Label Studio propagated to bbox coordinates.
- **Field normalization:** `legibility` and `language` fields normalized to lowercase across all sources.
- **Anti-leakage preserved:** professional and volunteer school train contain grades 5/6/7/9/11 only; grades 8/10 remain exclusive to test. 24 newly-contributed volunteer school images with grades 8/10 were deliberately excluded rather than added to test, to keep the Kaggle evaluation set stable.
- **Test set unchanged:** 386 images, identical `solution.csv` — the mid-competition evaluation set is stable.

### v1.0 — 2026-04-13

Initial public release of RUKOPYS v1 alongside the Kaggle competition launch.

---

## Citation

```bibtex
@dataset{rukopys_2026,
  title        = {{RUKOPYS}: Ukrainian Handwritten Text Recognition Dataset},
  author       = {Dmytro Voitekh and Volodymyr Zmiivskyyi and Oleksii Molchanovskyi},
  organization = {Ukrainian Catholic University},
  year         = {2026},
  license      = {CC BY-NC-SA 4.0},
  url          = {https://huggingface.co/UkrainianCatholicUniversity/rukopys},
  note         = {First large-scale Ukrainian HTR dataset; from 1920s archival documents to 2025 school homework and exams}
}
```
