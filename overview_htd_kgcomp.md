Overview
Goal of the challenge
As Ukraine continues its digital transformation, working with handwritten documents remains a major bottleneck. There is still a lack of open and straightforward tools for processing Ukrainian handwritten materials. Most existing solutions are either closed or built from fragmented components that do not scale well. This slows down the launch and development of public digital services, such as ePermit.

In this challenge, participants will build AI solutions to recognize Ukrainian handwritten documents — applications, certificates, logs, signatures, stamps, and archival materials — aligned with real governmental processes and needs.

Rather than just training models on a dataset, this challenge aims to help create tools that can be adopted in practice by the Ministry of Economy, the Ministry of Digital Transformation, the State Archival Service, and many others, simplifying document processing and accelerating public service delivery.

The most successful solutions will demonstrate:

Robust handwriting recognition across different document types and writing styles;
Practical reproducibility and deployment readiness;
The ability to work with real-world, noisy, and diverse data.
Start

25 days ago
Close

a month to go
Technical Constraints
Open models only — the inference pipeline must use exclusively open-weight models. Proprietary APIs (OpenAI, Anthropic, Gemini, etc.) are not permitted at inference time and for submissions!
Single H100 — the complete inference pipeline must fit into a single NVIDIA H100 80GB GPU. Solutions requiring more compute will not pass verification.
Dataset — RUKOPYS
The challenge is built around RUKOPYS — the first large-scale open dataset for Ukrainian handwritten text recognition. It covers over a century of Ukrainian handwriting across four document types:

Source	Period	Description
National Dictation	2020–2025	Phone photos of handwritten Ukrainian National Dictation submissions. Thousands of unique handwriting styles, one known canonical text per year.
State Archive (ЦДАВО)	1919–1935	Scanned documents from 12 archival funds. Pen & ink, archaic orthography.
University (KNUTE)	2022–2025	Scanned student exam work: text, formulas, chemistry, tables.
School Homework – Opornyi Lyceum s. Zymne (Опорний ліцей с. Зимне)	2022–2025	Phone photos of school homework, grades 5–11, 20+ subjects.
The dataset is publicly available on HuggingFace: UkrainianCatholicUniversity/rukopys

Train: human-annotated images (bboxes + type + transcription)
Silver: auto-annotated images for self-training
Test: submit your predictions here
Evaluation
Submissions are scored using a composite metric that evaluates three aspects of handwritten document understanding: region detection, type classification, and text transcription.

Score = 0.15 × Detection_F1 + 0.05 × ClassAcc + 0.30 × (1 − CER) + 0.50 × (1 − PageCER)
Score Components
Detection F1 (15%) — Bounding box detection quality, type-agnostic. A predicted box is counted as a true positive if its IoU with a ground-truth box is ≥ 0.5. Precision, recall and F1 are computed globally across all images.

Classification Accuracy (5%) — Among all IoU-matched region pairs, the fraction where the predicted type matches the ground-truth type. Region types: handwritten, printed, formula, table, annotation, image, graph.

Character Error Rate — per-region (30%) — Levenshtein distance divided by ground-truth text length, averaged across all matched scorable regions. A region is scorable when the ground-truth has language=uk, legibility=legible, and type is not image or graph.

Page CER (50%) — Full-page text comparison. Ground-truth and predicted regions are independently sorted by reading order (top-to-bottom, left-to-right) and concatenated into a single string. Page CER is the Levenshtein distance between the two strings divided by the ground-truth string length, averaged across all test images. This component is agnostic to bbox granularity — it rewards correct page text even when individual box boundaries are imprecise.

Text Normalization
Before CER comparison, within the metric function, both ground-truth and predicted text are normalized identically:

Step	Rule
Strikethrough	~~old~~{new} → new; ~~text~~ → text
LaTeX symbols	\alpha → α, \cdot → ·, \rightarrow → →, etc.
Latin/Cyrillic lookalikes	Latin c, o, p, x → Cyrillic equivalents
Dashes	em-dash, en-dash → hyphen -
Quotes	«»"" → ", '' → '
Superscripts/subscripts	x² → x^2, H₂ → H_2
LaTeX braces	x_{3} → x_3
Whitespace	collapse, strip
Submission File
For each image in the test set, predict a list of regions. The submission is a CSV file with two columns:

image,regions
abc123.jpg,"[{""bbox"":[50,100,850,130],""type"":""handwritten"",""text"":""Доброго ранку""}]"
def456.jpg,[]
image — test image filename (e.g. abc123.jpg)
regions — JSON-encoded list of detected regions; use [] for images with no detections
Each region object requires three fields:

Field	Type	Description
bbox	[x1, y1, x2, y2]	Pixel coordinates, top-left origin
type	string	One of: handwritten, printed, formula, table, annotation, image, graph
text	string	Transcribed text; empty string for image and graph regions
All test images must be present in the submission. Missing images will raise a scoring error.

Full scoring code, text normalization rules, and a local debugger
Official Evaluation Metric & Text Normalization

Selection for the Final
Advancement to the final demo day is based on both the Kaggle leaderboard score and performance on a private benchmark — a separate held-out image set that is withheld during the competition to prevent manual annotation or test-set overfitting. The private benchmark will be published after the competition closes, making it a reusable evaluation set for the community. Performance on this benchmark is one of the key criteria for selecting teams for the July 4 final in Kyiv (hybrid format - online/offline).

Recommended Approaches
This competition is open to any method, but here are the directions we find most promising:

VLM Fine-tuning
Adapt vision-language models (Gemma 4, Qwen3-VL, LLaMA, etc.) to Ukrainian handwritten text. This is the most direct path to strong results.

Agentic Recognition Pipelines
Multi-step systems that classify document type and content, then route to specialized strategies — e.g., delegating formulas, tables, or printed text to dedicated models.

Retrieval-Augmented Recognition (RAR)
Equip your system with external knowledge bases: domain-specific vocabularies, document templates, lexical patterns. Identify the document type, retrieve relevant context, and use it to guide recognition.

Additional Data Sources
Our dataset alone may not be sufficient to train large models from scratch. We encourage creative use of external datasets, synthetic data generation, and pseudo-labeling to expand training data.

Portability
Smaller and more efficient solutions are preferred. In production, these models must run on limited hardware. Efficiency per compute unit is a competitive advantage.

Post-processing & Ensembles
Language model correction, dictionary-based fixing, and multi-model voting at the line or character level are proven ways to improve CER.

Timeline
April 16, 2026 — Challenge starts on Kaggle
June 15, 2026 — End of the online stage
July 4, 2026 — Hybrid final event in Kyiv (in-person or remote presentation)
Participation requires only a laptop and a Kaggle account. The number of participants is not limited. The top teams will advance to the final.

Prizes
There will be three winning teams.

The 1st place team will receive $4,000, the 2nd place team $2,000, and the 3rd place team $1,000.

In addition, the winning teams will have the opportunity to implement their solutions in real government services.

Tips & Resources
RUKOPYS is a starting point, not a ceiling. Here are proven ways to go further.

Self-training with the silver split
The silver split contains auto-annotated images — the same sources and format as train. Use it as a noisy-label pretraining stage before fine-tuning on human-verified train annotations. A simple curriculum: train on silver first, then fine-tune on gold.

Pseudo-labeling with dictation ground truth
The National Dictation ground-truth texts are publicly available for each year. Because the canonical text is known, you can skip bbox-level annotation and align the full-page transcription directly to image lines — useful for text-line-level pretraining without human annotation.

Synthetic data
Generating synthetic Ukrainian handwriting is practical and well-supported:

TextRecognitionDataGenerator (TRDG) — render text with handwriting-style fonts, distortions, custom backgrounds. Supports any language and custom glyph sets.
FbSTG — tested specifically on historical Cyrillic documents; reduced CER by 24% and WER by 8% in a published evaluation.
For Ukrainian text content there're popular open text corpora like UberText, Kobza etc. For handwriting-style fonts with full Ukrainian character coverage (Ґ, Є, І, Ї) — check Google Fonts and Font Squirrel filtering for Ukrainian support.

External datasets (permitted external data)
The following open datasets are compatible with competition rules — publicly available, non-commercial use:

Dataset	Language	Description
HKR (GitHub)	Kazakh (Cyrillic)	~63K sentences, ~200 writers, Nazarbayev University
IAM Handwriting DB	English	1,500+ pages, gold standard for Latin HTR benchmarks
Pretrained checkpoints worth fine-tuning
End-to-end document understanding (layout + OCR in one model):

Qwen/Qwen3-VL-8B-Instruct — strong vision-language model with native document understanding; fits on a single H100 with room for batching. The 72B variant is available for training if you have the compute.
Google Gemma 4 — open-weight multimodal family: 5B (gemma-4-E2B-it), 8B MoE (gemma-4-E4B-it), 27B (gemma-4-26B-A4B-it), 31B (gemma-4-31B-it). All are vision-capable and fine-tunable; the 5B–27B variants fit comfortably on a single H100.
PaddleOCR — production-grade OCR framework with pretrained detection + recognition pipelines; supports custom Cyrillic fine-tuning and has strong layout analysis tools built in.
Text-line recognition (after bbox detection):

microsoft/trocr-base-handwritten — encoder-decoder baseline for handwritten text lines; fine-tunes well on Cyrillic with modest data.
Kansallisarkisto/cyrillic-htr-model — trained on 30K+ lines of Cyrillic archival handwriting by the National Archives of Finland.
Note on Gemini: Gemini 2.5 Flash and Pro (via Google AI Studio) cannot be used in the inference pipeline, but are excellent for generating pseudo-annotations on additional unlabeled data before training your open model.

Available Сompute Resources
Training can use any hardware — only inference must fit on a single H100. Here is how to get GPU time at no cost.

AWS Credits from the organizers
Teams can apply for AWS credits through the competition organizers. Credits will be allocated during the competition period. To request — post in the competition Discord channel or contact the organizers directly.

Free notebook environments (no credit card required)
Platform	GPU	Limit	Notes
Kaggle Notebooks	P100 16GB	~30 h/week	Built into the competition — use it
Google Colab	T4 15GB	~15–30 h/week	Good for quick experiments
Lightning AI	T4	15 credits/month	100 GB persistent storage, collaborative
Amazon SageMaker Studio Lab	T4 16GB	4 h/day	Stable persistent environment
A team of 4 using Kaggle Notebooks alone gets 120+ GPU-hours/week.

Serious training — A100/H100 access
Modal — $30/month free credits (renewable), H100 available on-demand. No credit card. Extra credits via academics and startups programs.
EuroHPC AI Factory Playground — 5,000+ GPU-hours on H200/A100, approved in ~2 business days. Ukraine is eligible (Horizon Europe associated country). Requires an organization (company or NGO), not an individual.
Free inference APIs — useful for data preparation and annotation
Proprietary APIs cannot be used in the inference pipeline, but they are fully allowed for labeling additional data, filtering, pseudo-annotation, or any other data preparation step outside of inference.

Service	What's free
Google AI Studio	Gemini 2.5 Pro/Flash, 1M context, multimodal — excellent for OCR pre-annotation
Together AI	71+ open models free, $25 signup credits, fine-tuning supported
NVIDIA Build	1,000 free credits, 100+ open models including vision models
Groq	Free tier forever, no credit card, fast inference
Acknowledgements
The creation of the Handwritten to Data dataset and the organization of this challenge were made possible thanks to the support of our partners and collaborators. Building this dataset and preparing the challenge required significant effort across data collection, curation, annotation, and validation, and we are deeply grateful to everyone who contributed their time, expertise, and resources. We would like to sincerely thank the organizations and teams that supported this initiative.

Our General Partners: mono, MacPaw

Our Partners: Ministry of Digital Transformation of Ukraine, Kyivstar, AWS, SoftServe, Keymakr, State Archival Service of Ukraine, State Archives of Ukraine, Ukrainian Catholic University, Kyiv School of Economics, Kyiv National University of Trade and Economics (KNUTE), Opornyi Lyceum s. Zymne (Опорний ліцей с. Зимне), Corezoid, Ukrainian Public Broadcasting Service (NSTU), ukraineincolor.com

We are especially thankful to all the teams and individuals who worked on collecting, preparing, and validating the handwritten document data used in this challenge. Their contributions are essential to making this open dataset and the competition possible.

## Mô tả dataset RUKOPYS
Cấu trúc cây thư mục (Dựa trên Hình 1 và Tài liệu)
Dữ liệu RUKOPYS được chia thành ba tập chính. Dựa vào ảnh chụp màn hình, cấu trúc thư mục được tổ chức như sau:

silver/: Thư mục chứa tập dữ liệu được gán nhãn tự động (auto-annotated) dùng để tự huấn luyện (self-training).

images/: Thư mục con chứa các file hình ảnh của tài liệu.

common_metadata.json: File chứa các thông tin siêu dữ liệu chung cho toàn bộ tập silver.

metadata.jsonl: File JSON Lines chứa thông tin chi tiết (kích thước, nguồn, tọa độ bounding box, nội dung text,...) cho từng hình ảnh tương ứng.

train/: Thư mục chứa tập dữ liệu được gán nhãn bởi con người (human-annotated), bao gồm bboxes, type và transcription. Cấu trúc bên trong tương tự như thư mục silver.

test/: Thư mục chứa tập dữ liệu dùng để người tham gia dự đoán và nộp kết quả (submission).

Mẫu cấu trúc file metadata.jsonl (Dựa trên Hình 3 và Hình 4)
File metadata.jsonl lưu trữ dữ liệu dưới dạng JSON Lines, nghĩa là mỗi dòng trong file sẽ là một đối tượng JSON (JSON object) hoàn chỉnh đại diện cho một hình ảnh. Dựa trên hình ảnh bạn cung cấp, một đối tượng JSON này có cấu trúc như sau:

JSON
{
  "file_name": "images/3d5475d3-c020-5ae9-9262-1ff95e5992fd.jpg",
  "image_width": 1600,
  "image_height": 900,
  "source": "school",
  "annotation_source": "auto",
  "grade": 7,
  "subject": "Зарубіжна література",
  "group": "",
  "regions": [
    {
      "bbox": [
        256,
        22,
        677,
        78
      ],
      "type": "handwritten",
      "language": "uk",
      "legibility": "legible",
      "text": "Криптограма - шифр"
    }
  ]
}
Giải thích các trường dữ liệu chính:

Thông tin hình ảnh:

file_name: Đường dẫn tương đối đến file ảnh.

image_width / image_height: Kích thước chiều rộng và chiều cao của ảnh (pixel).

Thông tin nguồn gốc & bối cảnh:

source: Nguồn gốc của ảnh (ví dụ: "school" - bài tập về nhà của học sinh).

annotation_source: Nguồn gán nhãn (ví dụ: "auto" nghĩa là gán nhãn tự động, tương ứng với tập silver).

grade & subject: Lớp và Môn học (trong ví dụ là lớp 7, môn Văn học nước ngoài).

regions: Một mảng (array) chứa nhiều đối tượng. Mỗi đối tượng đại diện cho một vùng dữ liệu được nhận diện trên ảnh, bao gồm:

bbox: Mảng gồm 4 số nguyên [x1, y1, x2, y2] thể hiện tọa độ điểm ảnh trên cùng bên trái và dưới cùng bên phải.

type: Loại dữ liệu của vùng đó (trong ảnh là "handwritten" - chữ viết tay). Theo tài liệu, nó có thể là: handwritten, printed, formula, table, annotation, image, graph.

language: Ngôn ngữ (ví dụ: "uk" - tiếng Ukraina).

legibility: Mức độ dễ đọc (ví dụ: "legible" - có thể đọc được).

text: Nội dung văn bản được sao lại (transcribed text) từ hình ảnh. Đối với vùng là ảnh (image) hoặc biểu đồ (graph), chuỗi này sẽ rỗng.