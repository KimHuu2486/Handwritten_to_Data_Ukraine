import json
import re
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "finetune" / "yolo_qwen_end_to_end"
DST = ROOT / "finetune" / "yolo_qwen_end_to_end_VLM_R^3_HALP"


def read_nb(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_nb(path, nb):
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def src_text(cell):
    return "".join(cell.get("source", []))


def set_src(cell, text):
    cell["source"] = text.splitlines(keepends=True)


TRAIN_EXTRA_CONFIG = r'''

# VLM-R3 + HALP settings.
ILLEGIBLE_TOKEN = "[illegible]"
VLM_R3_ENABLE_REFINE_SAMPLES = True
END_TO_END_PAGE_REPEAT = 2
VLM_R3_FORMULA_TABLE_REPEAT = 3
HALP_SMALL_CROP_AREA_RATIO = 0.0025
HALP_EMPTY_TEXT_RISK = True

VLM_R3_ACTION_PROMPT = (
    "You are doing region-aware OCR. First decide whether this crop needs finer visual evidence. "
    "If it does, emit one compact JSON action exactly like "
    "{\"action\":\"zoom\",\"bbox_2d\":[x1,y1,x2,y2],\"reason\":\"short reason\"}. "
    "bbox_2d uses absolute pixel coordinates in the source image. "
    "After the action, provide the final transcription on the next line. "
    "If the visible content cannot be read even after refinement, return [illegible]. "
    "Do not invent hidden or memorized text."
)

VLM_R3_FINAL_PROMPTS = {
    "formula": (
        "Focus on math, physics, chemistry, indices, superscripts, subscripts, fractions, "
        "arrows, matrices, units, and signs. Preserve exact visible notation. "
        "Do not solve or simplify the expression."
    ),
    "table": (
        "Focus on table structure. Preserve row order and column order. "
        "Use one output line per visual row and | between cells. "
        "Use empty fields for visibly empty cells."
    ),
    "default": "Transcribe only the visible content exactly.",
}
'''


STAGE_BUILD_CELL = r'''def make_page_sample(record, root, split):
    regions = compact_page_regions(record)
    if not regions or len(regions) > MAX_PAGE_REGIONS:
        return None
    answer = json.dumps(regions, ensure_ascii=False, separators=(",", ":"))
    if len(answer) > MAX_PAGE_ANSWER_CHARS:
        return None
    return {
        "task": "page_json",
        "image_path": resolve_image_path(root, split, record["file_name"]),
        "bbox": [0, 0, 0, 0],
        "region_type": "page",
        "prompt": PAGE_PROMPT,
        "answer": answer,
        "risk_label": 0,
        "risk_reason": "page",
        "source": record.get("source", "unknown"),
    }


def box_area_ratio(box, w, h):
    if box is None:
        return 0.0
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1]) / max(1, w * h)


def compute_region_risk_label(record, region, box):
    rtype = normalize_type(region.get("type"))
    text = str(region.get("text") or "").strip()
    w = max(1, int(record.get("image_width") or 1))
    h = max(1, int(record.get("image_height") or 1))
    area_ratio = box_area_ratio(box, w, h)
    if rtype in {"formula", "table"}:
        return 1, "formula_table"
    if HALP_EMPTY_TEXT_RISK and ((not text) or ILLEGIBLE_TOKEN in text.lower()):
        return 1, "empty_or_illegible"
    if area_ratio and area_ratio < HALP_SMALL_CROP_AREA_RATIO:
        return 1, "small_crop"
    return 0, "normal"


def make_crop_sample(record, region, root, split):
    w = max(1, int(record.get("image_width") or 1))
    h = max(1, int(record.get("image_height") or 1))
    box = clamp_box(region.get("bbox"), w, h)
    if box is None or not region_is_crop_candidate(region):
        return None
    rtype = normalize_type(region.get("type"))
    prompt = CROP_PROMPTS.get(rtype, CROP_PROMPTS["default"])
    risk_label, risk_reason = compute_region_risk_label(record, region, box)
    return {
        "task": "crop_ocr",
        "image_path": resolve_image_path(root, split, record["file_name"]),
        "bbox": box,
        "region_type": rtype,
        "prompt": prompt,
        "answer": str(region.get("text") or ""),
        "risk_label": risk_label,
        "risk_reason": risk_reason,
        "source": record.get("source", "unknown"),
    }


def make_vlm_r3_refine_sample(record, region, root, split):
    if not VLM_R3_ENABLE_REFINE_SAMPLES:
        return None
    rtype = normalize_type(region.get("type"))
    if rtype not in {"formula", "table"}:
        return None
    text = str(region.get("text") or "").strip()
    if not text or text == ILLEGIBLE_TOKEN:
        return None
    w = max(1, int(record.get("image_width") or 1))
    h = max(1, int(record.get("image_height") or 1))
    box = clamp_box(region.get("bbox"), w, h)
    if box is None:
        return None
    action = {
        "action": "zoom",
        "bbox_2d": box,
        "reason": f"{rtype} requires fine visual evidence",
    }
    prompt = VLM_R3_ACTION_PROMPT + "\n" + VLM_R3_FINAL_PROMPTS.get(rtype, VLM_R3_FINAL_PROMPTS["default"])
    answer = json.dumps(action, ensure_ascii=False, separators=(",", ":")) + "\n" + text
    return {
        "task": "vlm_r3_refine",
        "image_path": resolve_image_path(root, split, record["file_name"]),
        "bbox": box,
        "region_type": rtype,
        "prompt": prompt,
        "answer": answer,
        "risk_label": 1,
        "risk_reason": "vlm_r3_formula_table",
        "source": record.get("source", "unknown"),
    }


def balanced_take(samples, limit):
    if limit is None or len(samples) <= limit:
        return samples
    buckets = defaultdict(list)
    for s in samples:
        key = (s.get("source", "unknown"), s.get("region_type", s.get("task")), s.get("task"))
        buckets[key].append(s)
    rng = random.Random(SEED)
    for items in buckets.values():
        rng.shuffle(items)
    selected = []
    keys = list(buckets.keys())
    while len(selected) < limit and keys:
        next_keys = []
        for key in keys:
            if buckets[key] and len(selected) < limit:
                selected.append(buckets[key].pop())
            if buckets[key]:
                next_keys.append(key)
        keys = next_keys
    rng.shuffle(selected)
    return selected


def build_samples(records, root, split, page_limit=None, crop_limit=None):
    rng = random.Random(SEED)
    rows = list(records)
    rng.shuffle(rows)
    page_samples, crop_samples, refine_samples = [], [], []
    for record in rows:
        page = make_page_sample(record, root, split)
        if page is not None:
            page_samples.append(page)
        for region in record.get("regions") or []:
            crop = make_crop_sample(record, region, root, split)
            if crop is not None:
                crop_samples.append(crop)
            refine = make_vlm_r3_refine_sample(record, region, root, split)
            if refine is not None:
                for _ in range(max(1, VLM_R3_FORMULA_TABLE_REPEAT)):
                    refine_samples.append(dict(refine))
    page_samples = balanced_take(page_samples, page_limit)
    crop_samples = balanced_take(crop_samples, crop_limit)
    repeated_page_samples = []
    for page in page_samples:
        for _ in range(max(1, END_TO_END_PAGE_REPEAT)):
            repeated_page_samples.append(dict(page))
    mixed = repeated_page_samples + crop_samples + refine_samples
    rng.shuffle(mixed)
    risk_count = sum(int(s.get("risk_label", 0)) for s in mixed)
    print(
        f"{split}: page={len(page_samples)} page_weighted={len(repeated_page_samples)} crop={len(crop_samples)} "
        f"vlm_r3_refine={len(refine_samples)} risk={risk_count} total={len(mixed)}"
    )
    return mixed

'''


STAGE1_POST_BUILD = r'''
root = get_dataset_root()
model_id = find_model_id()
silver_split = get_silver_split(root)

print("Dataset root:", root)
print("Silver split:", silver_split)
print("Base model:", model_id)

silver_records = read_jsonl(root / silver_split / "metadata.jsonl")
stage1_samples = build_samples(
    silver_records,
    root,
    silver_split,
    page_limit=SILVER_PAGE_LIMIT,
    crop_limit=SILVER_CROP_LIMIT,
)
if not stage1_samples:
    raise RuntimeError(f"No silver training samples were built. Check {root / silver_split / 'metadata.jsonl'}.")
stage1_ds = Dataset.from_list(stage1_samples)
'''


STAGE2_POST_BUILD = r'''
root = get_dataset_root()
model_id = find_model_id()
stage1_lora_dir = find_stage1_lora_dir()

print("Dataset root:", root)
print("Stage 1 LoRA:", stage1_lora_dir)
print("Base model:", model_id)

gold_records = read_jsonl(root / "train" / "metadata.jsonl")
gold_train_records, gold_val_records = stratified_split(gold_records, VAL_RATIO)
print(f"Gold train pages={len(gold_train_records)} val pages={len(gold_val_records)}")

stage2_samples = build_samples(
    gold_train_records,
    root,
    "train",
    page_limit=GOLD_PAGE_LIMIT,
    crop_limit=GOLD_CROP_LIMIT,
)
if not stage2_samples:
    raise RuntimeError(f"No gold training samples were built. Check {root / 'train' / 'metadata.jsonl'}.")
stage2_ds = Dataset.from_list(stage2_samples)

val_jsonl = OUTPUT_DIR / "gold_validation_records.jsonl"
with open(val_jsonl, "w", encoding="utf-8") as f:
    for row in gold_val_records:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print("Saved validation records to", val_jsonl)
'''


TRAIN_COLLATOR_CELL = r'''def crop_with_padding(image_path, bbox, pad_ratio=CROP_PAD_RATIO):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        x1, y1, x2, y2 = bbox
        pad = int(round(max(x2 - x1, y2 - y1) * pad_ratio))
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        return img.crop((x1, y1, x2, y2))


def build_messages(sample):
    if sample["task"] == "page_json":
        image_obj = sample["image_path"]
        max_pixels = MAX_PIXELS_PAGE
    else:
        image_obj = crop_with_padding(sample["image_path"], sample["bbox"])
        max_pixels = MAX_PIXELS_CROP
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_obj, "max_pixels": max_pixels},
                {"type": "text", "text": sample["prompt"]},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": sample["answer"]}]},
    ]


def build_user_messages(sample):
    return build_messages(sample)[:1]


def encode_marker(tokenizer):
    try:
        return tokenizer.encode("<|im_start|>assistant\n", allowed_special="all", add_special_tokens=False)
    except TypeError:
        return tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)


ASSISTANT_MARKER = encode_marker(processor.tokenizer)


def data_collator(examples):
    messages_list = [build_messages(ex) for ex in examples]
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        for msg in messages_list
    ]
    image_inputs, video_inputs = process_vision_info(messages_list)
    batch = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt",
    )

    labels = batch["input_ids"].clone()
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is not None:
        labels[labels == pad_id] = -100

    eos_id = processor.tokenizer.eos_token_id
    for i in range(labels.shape[0]):
        ids = batch["input_ids"][i].tolist()
        start = -1
        for j in range(0, len(ids) - len(ASSISTANT_MARKER) + 1):
            if ids[j:j + len(ASSISTANT_MARKER)] == ASSISTANT_MARKER:
                start = j + len(ASSISTANT_MARKER)
                break
        actual_len = int(batch["attention_mask"][i].sum().item())
        truncated = actual_len >= MAX_SEQ_LENGTH and (eos_id is None or ids[actual_len - 1] != eos_id)
        if start >= 0 and not truncated:
            labels[i, :start] = -100
        else:
            labels[i, :] = -100

    batch["labels"] = labels
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
            batch[key] = value.to(torch.float16)
    return batch
'''


HALP_PROBE_CELL = r'''# Optional HALP probe training.
# This is the paper-faithful part: collect pre-generation hidden states, assign hallucination-risk labels,
# then train a lightweight MLP probe. Set RUN_HALP_PROBE_TRAIN=False if you only want the LoRA adapter.
import torch.nn as nn
import torch.nn.functional as F

RUN_HALP_PROBE_TRAIN = True
HALP_PROBE_LABEL_MODE = "generation_cer"  # choices: "generation_cer", "risk_label"
HALP_PROBE_MAX_SAMPLES = 256
HALP_PROBE_LAYER = -1
HALP_PROBE_EPOCHS = 6
HALP_PROBE_LR = 1e-3
HALP_PROBE_CER_THRESHOLD = 0.35
HALP_PROBE_MAX_NEW_TOKENS = 192


def active_device(model):
    try:
        return next(p for p in model.parameters() if p is not None).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def apply_generation_template(processor, messages):
    try:
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except TypeError:
        return processor.apply_chat_template(messages, tokenize=False)


def encode_generation_inputs(processor, messages, device):
    text = apply_generation_template(processor, messages)
    image_inputs, video_inputs = process_vision_info(messages)
    batch = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return batch.to(device)


def halp_feature_from_messages(model, processor, messages):
    device = active_device(model)
    inputs = encode_generation_inputs(processor, messages, device)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    hidden = outputs.hidden_states[HALP_PROBE_LAYER]
    mask = inputs["attention_mask"]
    last_idx = mask.sum(dim=1).clamp(min=1) - 1
    feat = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_idx].float().detach().cpu()
    del inputs, outputs, hidden
    torch.cuda.empty_cache()
    return feat.squeeze(0)


def generate_probe_prediction(model, processor, messages):
    device = active_device(model)
    inputs = encode_generation_inputs(processor, messages, device)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
        out = model.generate(
            **inputs,
            max_new_tokens=HALP_PROBE_MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
        )
    trimmed = out[0, inputs.input_ids.shape[1]:]
    text = processor.decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    del inputs, out, trimmed
    torch.cuda.empty_cache()
    return text


def normalize_for_cer(text):
    text = str(text or "").strip()
    text = re.sub(r"^```[a-zA-Z]*", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def edit_distance(a, b):
    a, b = normalize_for_cer(a), normalize_for_cer(b)
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalized_cer(pred, gold):
    gold = normalize_for_cer(gold)
    if not gold:
        return 1.0 if normalize_for_cer(pred) else 0.0
    return edit_distance(pred, gold) / max(1, len(gold))


class HALPProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden = min(1024, max(128, input_dim // 2))
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_halp_probe_from_samples(samples):
    rng = random.Random(SEED)
    probe_samples = [s for s in samples if s.get("task") in {"crop_ocr", "vlm_r3_refine"}]
    rng.shuffle(probe_samples)
    probe_samples = probe_samples[:HALP_PROBE_MAX_SAMPLES]
    if not probe_samples:
        print("HALP probe skipped: no probe samples.")
        return None

    features, labels, diagnostics = [], [], []
    probe_model = trainer.model.eval()
    for idx, sample in enumerate(probe_samples, 1):
        messages = build_user_messages(sample)
        if HALP_PROBE_LABEL_MODE == "generation_cer":
            pred = generate_probe_prediction(probe_model, processor, messages)
            cer = normalized_cer(pred, sample["answer"])
            label = int(cer >= HALP_PROBE_CER_THRESHOLD or not normalize_for_cer(pred))
            diagnostics.append((sample.get("region_type"), round(cer, 3), label))
        else:
            label = int(sample.get("risk_label", 0))
        feat = halp_feature_from_messages(probe_model, processor, messages)
        features.append(feat)
        labels.append(label)
        if idx % 25 == 0:
            print(f"HALP probe feature {idx}/{len(probe_samples)}", flush=True)

    x = torch.stack(features).float()
    y = torch.tensor(labels, dtype=torch.float32)
    positives = int(y.sum().item())
    print(f"HALP probe dataset: n={len(y)} positives={positives} negatives={len(y) - positives}")
    if positives == 0 or positives == len(y):
        print("HALP probe warning: labels are one-sided; saved probe may not be useful.")

    probe = HALPProbe(x.shape[1])
    opt = torch.optim.AdamW(probe.parameters(), lr=HALP_PROBE_LR, weight_decay=0.01)
    for epoch in range(HALP_PROBE_EPOCHS):
        probe.train()
        logits = probe(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            probs = torch.sigmoid(probe(x))
            pred = (probs >= 0.5).float()
            acc = (pred == y).float().mean().item()
        print(f"HALP probe epoch {epoch + 1}/{HALP_PROBE_EPOCHS}: loss={loss.item():.4f} acc={acc:.3f}")

    probe_path = final_dir / "halp_probe.pt"
    torch.save(
        {
            "state_dict": probe.state_dict(),
            "input_dim": x.shape[1],
            "layer": HALP_PROBE_LAYER,
            "threshold": 0.5,
            "label_mode": HALP_PROBE_LABEL_MODE,
            "cer_threshold": HALP_PROBE_CER_THRESHOLD,
        },
        probe_path,
    )
    (final_dir / "halp_probe_config.json").write_text(
        json.dumps(
            {
                "input_dim": x.shape[1],
                "layer": HALP_PROBE_LAYER,
                "threshold": 0.5,
                "label_mode": HALP_PROBE_LABEL_MODE,
                "cer_threshold": HALP_PROBE_CER_THRESHOLD,
                "max_samples": len(y),
                "positive_count": positives,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Saved HALP probe to", probe_path)
    if diagnostics:
        print("HALP diagnostics sample:", diagnostics[:10])
    return probe


if RUN_HALP_PROBE_TRAIN:
    train_halp_probe_from_samples(stage2_samples)
else:
    print("HALP probe training disabled.")
'''


STAGE1_LORA_CANDIDATES_STRICT = '''STAGE1_LORA_CANDIDATES = [
    "/kaggle/working/qwen3vl_rukopys_stage1_silver_vlm_r3_halp/qwen3vl_silver_lora_final",
    "/kaggle/input/qwen3vl-rukopys-stage1-vlm-r3-halp/qwen3vl_silver_lora_final",
]
'''


SUBMIT_LORA_CANDIDATES_STRICT = '''LORA_CANDIDATES = [
    "/kaggle/working/qwen3vl_rukopys_stage2_gold_vlm_r3_halp/qwen3vl_rukopys_lora_final",
    "/kaggle/input/qwen3vl-rukopys-stage2-vlm-r3-halp/qwen3vl_rukopys_lora_final",
]
'''


SUBMIT_IMPORT_PATCH = "from PIL import Image, ImageFilter, ImageStat\n"


SUBMIT_EXTRA_CONFIG = r'''

# VLM-R3 + HALP routing settings.
ILLEGIBLE_TOKEN = "[illegible]"
USE_QWEN_PAGE_JSON = False  # keep YOLO as the submit detector; enable only for ablation/ensemble tests
QWEN_PAGE_JSON_MAX_NEW_TOKENS = 2048
QWEN_YOLO_MERGE_IOU = 0.45
USE_HALP_PROBE = True
HALP_RISK_THRESHOLD = 0.50
HALP_FORMULA_TABLE_BASE_RISK = 0.45
HALP_LOW_CONF_THRESHOLD = 0.35
HALP_LOW_CONF_BONUS = 0.18
HALP_SMALL_AREA_THRESHOLD = 0.0025
HALP_SMALL_AREA_BONUS = 0.18
HALP_BLUR_THRESHOLD = 12.0
HALP_BLUR_BONUS = 0.12
HALP_EMPTY_TEXT_BONUS = 0.25

VLM_R3_MAX_REFINEMENT_PASSES = 3
VLM_R3_FORCE_TYPES = {"formula", "table"}
VLM_R3_CONTEXT_SCALE = 1.35
VLM_R3_MIN_AREA = 0.125
VLM_R3_MAX_AREA = 0.5
MAX_PIXELS_REFINE = 524_288
MAX_PIXELS_TABLE = 786_432
MAX_NEW_TOKENS_REFINE = 256
'''


SUBMIT_PAGE_PROMPT_INSERT = r'''PAGE_PROMPT = (
    "Extract every visible document region. Return only a compact JSON array. "
    "Each item must have keys bbox,type,text. bbox is [x1,y1,x2,y2] on a 0-1000 grid. "
    "type is one of handwritten,printed,formula,table,annotation,image,graph. "
    "Use empty text for image and graph. Preserve reading order. "
    "For every text field, transcribe only visible characters exactly; do not translate, summarize, "
    "normalize spelling, correct grammar, expand abbreviations, or infer hidden/missing text. "
    "For unreadable visible words use [illegible]. "
    "No Markdown, no explanation."
)

'''


SUBMIT_QWEN_CELL = r'''from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
import torch.nn as nn


class HALPProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden = min(1024, max(128, input_dim // 2))
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


HALP_PROBE = None
HALP_PROBE_META = {}


def configure_processor_for_generation(processor):
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"
    return processor


def load_halp_probe(device):
    if not USE_HALP_PROBE:
        return None, {}
    probe_path = Path(lora_dir) / "halp_probe.pt"
    if not probe_path.exists():
        print("HALP probe not found; using heuristic risk routing.", flush=True)
        return None, {}
    try:
        payload = torch.load(probe_path, map_location="cpu")
        probe = HALPProbe(int(payload["input_dim"]))
        probe.load_state_dict(payload["state_dict"])
        probe.eval().to(device)
        meta = {
            "layer": int(payload.get("layer", -1)),
            "threshold": float(payload.get("threshold", 0.5)),
        }
        print("Loaded HALP probe:", probe_path, meta, flush=True)
        return probe, meta
    except Exception as e:
        print("Could not load HALP probe; using heuristic risk routing:", e, flush=True)
        return None, {}


def load_qwen_model(device):
    global HALP_PROBE, HALP_PROBE_META
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    base = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map={"": device},
        quantization_config=quantization_config,
        dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, str(lora_dir))
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    processor = configure_processor_for_generation(processor)
    if processor.tokenizer.pad_token_id is not None:
        model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    HALP_PROBE, HALP_PROBE_META = load_halp_probe(device)
    return model, processor


def apply_chat_template(processor, messages):
    candidates = [
        {"tokenize": False, "add_generation_prompt": True, "template_kwargs": {"enable_thinking": False}},
        {"tokenize": False, "add_generation_prompt": True, "processor_kwargs": {"enable_thinking": False}},
        {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False},
        {"tokenize": False, "add_generation_prompt": True},
    ]
    for kwargs in candidates:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*Kwargs passed to `processor\.__call__` have to be in `processor_kwargs` dict.*",
                )
                return processor.apply_chat_template(messages, **kwargs)
        except TypeError:
            continue
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def encode_generation_batch(processor, messages_batch, device):
    processor.tokenizer.padding_side = "left"
    texts = [apply_chat_template(processor, m) for m in messages_batch]
    image_inputs, video_inputs = process_vision_info(messages_batch)
    try:
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            text_kwargs={"padding": True, "return_tensors": "pt"},
            images_kwargs={"return_tensors": "pt"},
            videos_kwargs={"return_tensors": "pt"},
        )
    except TypeError:
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    return inputs.to(device)


def generate_batch(model, processor, messages_batch, device, max_new_tokens):
    inputs = encode_generation_batch(processor, messages_batch, device)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    del inputs, out, trimmed
    torch.cuda.empty_cache()
    return decoded


def halp_probe_predict(model, processor, messages, device):
    if HALP_PROBE is None:
        return None
    try:
        inputs = encode_generation_batch(processor, [messages], device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        layer = int(HALP_PROBE_META.get("layer", -1))
        hidden = outputs.hidden_states[layer]
        mask = inputs["attention_mask"]
        last_idx = mask.sum(dim=1).clamp(min=1) - 1
        feat = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_idx].float()
        with torch.no_grad():
            score = torch.sigmoid(HALP_PROBE(feat.to(next(HALP_PROBE.parameters()).device))).item()
        del inputs, outputs, hidden, feat
        torch.cuda.empty_cache()
        return float(score)
    except Exception as e:
        print("HALP probe forward failed; falling back to heuristic:", e, flush=True)
        torch.cuda.empty_cache()
        return None
'''


SUBMIT_INFERENCE_CELL = r'''def resize_to_pixel_budget(img, max_pixels):
    w, h = img.size
    total = max(1, w * h)
    if total <= max_pixels:
        return img
    scale = (max_pixels / total) ** 0.5
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def expand_box(bbox, img_w, img_h, scale=1.0, pad_ratio=0.0):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side_pad = max(bw, bh) * pad_ratio
    bw = bw * scale + 2 * side_pad
    bh = bh * scale + 2 * side_pad
    return clamp_xyxy([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], img_w, img_h)


def vlm_r3_zoom_scale(bbox, img_w, img_h):
    area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    ratio = area / max(1, img_w * img_h)
    if ratio < VLM_R3_MIN_AREA:
        return 2.0
    if ratio >= VLM_R3_MAX_AREA:
        return 1.0
    return 2.0 - ((ratio - VLM_R3_MIN_AREA) / (VLM_R3_MAX_AREA - VLM_R3_MIN_AREA))


def crop_image_variant(image_path, bbox, scale=1.0, pad_ratio=CROP_PAD_RATIO, max_pixels=MAX_PIXELS_CROP):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        box = expand_box(bbox, w, h, scale=scale, pad_ratio=pad_ratio)
        if box is None:
            box = bbox
        crop = img.crop(tuple(box))
        if scale > 1.0:
            cw, ch = crop.size
            crop = crop.resize((max(1, int(cw * scale)), max(1, int(ch * scale))), Image.Resampling.LANCZOS)
        return resize_to_pixel_budget(crop, max_pixels)


def crop_image(image_path, bbox):
    return crop_image_variant(image_path, bbox, scale=1.0, pad_ratio=CROP_PAD_RATIO, max_pixels=MAX_PIXELS_CROP)


def box_area_ratio(bbox, img_w, img_h):
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1]) / max(1, img_w * img_h)


def blur_score_for_region(image_path, bbox):
    try:
        with Image.open(image_path) as img:
            img = img.convert("L")
            w, h = img.size
            box = expand_box(bbox, w, h, scale=1.0, pad_ratio=CROP_PAD_RATIO)
            crop = img.crop(tuple(box or bbox)).resize((128, 128), Image.Resampling.BILINEAR)
            edges = crop.filter(ImageFilter.FIND_EDGES)
            stat = ImageStat.Stat(edges)
            return float(stat.var[0])
    except Exception:
        return 999.0


def page_messages(image_path):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path, "max_pixels": MAX_PIXELS_PAGE},
                {"type": "text", "text": PAGE_PROMPT},
            ],
        }
    ]


def extract_json_array(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, list):
                return obj
        except Exception:
            return []
    return []


def qwen_bbox_to_pixels(bbox, img_w, img_h):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        vals = [float(v) for v in bbox]
    except Exception:
        return None
    # The page_json training prompt uses a 0-1000 grid. If the model already emits pixels,
    # values usually exceed 1000 on full-resolution pages, so leave them as-is.
    if max(vals) <= 1000 and (img_w > 1000 or img_h > 1000):
        vals = [vals[0] * img_w / 1000.0, vals[1] * img_h / 1000.0, vals[2] * img_w / 1000.0, vals[3] * img_h / 1000.0]
    return clamp_xyxy(vals, img_w, img_h)


def parse_qwen_page_regions(text, img_w, img_h):
    regions = []
    for item in extract_json_array(text):
        if not isinstance(item, dict):
            continue
        box = qwen_bbox_to_pixels(item.get("bbox"), img_w, img_h)
        if box is None:
            continue
        rtype = normalize_type(item.get("type"))
        text_value = "" if rtype in {"image", "graph"} else clean_crop_text(item.get("text", ""))
        regions.append({"bbox": box, "type": rtype, "text": text_value, "_source": "qwen_page", "_score": 0.55})
    return sort_regions(regions)


def detect_regions_qwen_page(qwen_model, processor, image_path, device):
    if not USE_QWEN_PAGE_JSON:
        return []
    try:
        with Image.open(image_path) as img:
            img_w, img_h = img.size
        out = generate_batch(qwen_model, processor, [page_messages(image_path)], device, QWEN_PAGE_JSON_MAX_NEW_TOKENS)[0]
        regions = parse_qwen_page_regions(out, img_w, img_h)
        print(f"Qwen page_json regions={len(regions)}", flush=True)
        return regions
    except Exception as e:
        print("Qwen page_json failed; using YOLO only:", e, flush=True)
        torch.cuda.empty_cache()
        return []


def merge_qwen_yolo_regions(qwen_regions, yolo_regions):
    merged = [dict(r, _source=r.get("_source", "yolo")) for r in yolo_regions]
    for q in qwen_regions:
        best_idx, best_iou = None, 0.0
        for idx, old in enumerate(merged):
            overlap = iou(q["bbox"], old["bbox"])
            if overlap > best_iou:
                best_idx, best_iou = idx, overlap
        if best_idx is not None and best_iou >= QWEN_YOLO_MERGE_IOU:
            target = merged[best_idx]
            qtype = normalize_type(q.get("type"))
            if qtype in {"formula", "table"} or normalize_type(target.get("type")) == "handwritten":
                target["type"] = qtype
            if str(q.get("text") or "").strip() and not str(target.get("text") or "").strip():
                target["text"] = str(q.get("text") or "")
            target["_source"] = target.get("_source", "yolo") + "+qwen_page"
            target["_qwen_iou"] = best_iou
        else:
            merged.append(dict(q))
    return dedupe_yolo_regions(merged)


def should_crop_ocr(region):
    if CROP_OCR_MODE == "none":
        return False
    if region.get("type") not in TEXT_TYPES:
        return False
    if CROP_OCR_MODE == "all_text":
        return True
    if CROP_OCR_MODE == "halp_vlm_r3":
        if normalize_type(region.get("type")) in VLM_R3_FORCE_TYPES:
            return True
        text = str(region.get("text") or "").strip()
        if "qwen_page" in str(region.get("_source", "")) and text and len(text) >= 4 and ILLEGIBLE_TOKEN not in text.lower():
            return False
        return True
    text = str(region.get("text") or "")
    return (not text.strip()) or len(text) < 4 or len(text) > 160


def crop_prompt_for_region(region, refinement=False):
    rtype = normalize_type(region.get("type"))
    prompt = CROP_PROMPTS.get(rtype, CROP_PROMPTS["default"])
    if not refinement:
        return prompt
    if rtype == "formula":
        return (
            "You are reading a difficult formula region. Use every provided image view as visual evidence. "
            "Return only the exact visible formula or [illegible]. Do not solve, simplify, or explain. "
            + prompt
        )
    if rtype == "table":
        return (
            "You are reading a difficult table region. Use every provided image view as visual evidence. "
            "Return only pipe-separated table text, one visual row per line, or [illegible]. "
            + prompt
        )
    return "Use the provided crop and zoomed views. Return only the visible transcription or [illegible]. " + prompt


def crop_messages(image_path, region):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": crop_image(image_path, region["bbox"])},
                {"type": "text", "text": crop_prompt_for_region(region, refinement=False)},
            ],
        }
    ]


def vlm_r3_refine_messages(image_path, region, img_w, img_h):
    rtype = normalize_type(region.get("type"))
    bbox = region["bbox"]
    zoom_scale = vlm_r3_zoom_scale(bbox, img_w, img_h)
    max_pixels = MAX_PIXELS_TABLE if rtype == "table" else MAX_PIXELS_REFINE
    normal = crop_image_variant(image_path, bbox, scale=1.0, pad_ratio=CROP_PAD_RATIO, max_pixels=max_pixels)
    zoom = crop_image_variant(image_path, bbox, scale=zoom_scale, pad_ratio=CROP_PAD_RATIO, max_pixels=max_pixels)
    context = crop_image_variant(image_path, bbox, scale=VLM_R3_CONTEXT_SCALE, pad_ratio=CROP_PAD_RATIO, max_pixels=max_pixels)
    action = {"action": "zoom", "bbox_2d": bbox, "reason": f"{rtype} high HALP risk"}
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": normal},
                {"type": "text", "text": "Initial detected region."},
                {"type": "image", "image": zoom},
                {"type": "text", "text": "VLM-R3 zoomed region from action " + json.dumps(action, ensure_ascii=False) + "."},
                {"type": "image", "image": context},
                {"type": "text", "text": "Wider context view. " + crop_prompt_for_region(region, refinement=True)},
            ],
        }
    ]


def halp_heuristic_score(region, image_path, img_w, img_h):
    rtype = normalize_type(region.get("type"))
    score = 0.0
    signals = []
    if rtype in VLM_R3_FORCE_TYPES:
        score += HALP_FORMULA_TABLE_BASE_RISK
        signals.append("formula_table")
    yolo_score = float(region.get("_score", 1.0))
    if yolo_score < HALP_LOW_CONF_THRESHOLD:
        score += HALP_LOW_CONF_BONUS
        signals.append("low_yolo_conf")
    if box_area_ratio(region["bbox"], img_w, img_h) < HALP_SMALL_AREA_THRESHOLD:
        score += HALP_SMALL_AREA_BONUS
        signals.append("small_area")
    blur = blur_score_for_region(image_path, region["bbox"])
    if blur < HALP_BLUR_THRESHOLD:
        score += HALP_BLUR_BONUS
        signals.append("blur")
    text = str(region.get("text") or "").strip()
    if (not text) or ILLEGIBLE_TOKEN in text.lower():
        score += HALP_EMPTY_TEXT_BONUS
        signals.append("empty_or_illegible")
    return min(1.0, score), signals


def halp_region_score(qwen_model, processor, image_path, region, device, img_w, img_h):
    heuristic, signals = halp_heuristic_score(region, image_path, img_w, img_h)
    probe_score = None
    if USE_HALP_PROBE and HALP_PROBE is not None:
        probe_score = halp_probe_predict(qwen_model, processor, crop_messages(image_path, region), device)
    score = max(heuristic, probe_score if probe_score is not None else 0.0)
    region["_halp_risk"] = float(score)
    region["_halp_signals"] = ",".join(signals)
    if probe_score is not None:
        region["_halp_probe"] = float(probe_score)
    return score


def output_is_bad(text):
    text = str(text or "").strip()
    if not text:
        return True
    lowered = text.lower()
    if lowered in {"none", "null", "n/a"}:
        return True
    if lowered.startswith("{") or lowered.startswith("[") and not lowered.startswith(ILLEGIBLE_TOKEN.lower()):
        return True
    if "i cannot" in lowered or "sorry" in lowered:
        return True
    return False


def candidate_quality(text, rtype):
    text = clean_crop_text(text)
    if output_is_bad(text):
        return -1000
    if text == ILLEGIBLE_TOKEN:
        return -10
    score = min(len(text), 120) / 120.0
    if rtype == "formula":
        math_chars = sum(1 for ch in text if ch.isdigit() or ch in "=+-*/^_{}\\()[]<>√∑∫≈≤≥→←×÷")
        score += min(1.5, math_chars / 12.0)
    elif rtype == "table":
        separators = text.count("|") + text.count("\n")
        score += min(1.5, separators / 6.0)
    if ILLEGIBLE_TOKEN in text:
        score -= 0.5
    return score


def choose_best_candidate(candidates, rtype):
    cleaned = [clean_crop_text(c) for c in candidates]
    ranked = sorted(((candidate_quality(c, rtype), c) for c in cleaned), key=lambda x: x[0], reverse=True)
    if not ranked or ranked[0][0] < 0:
        return ILLEGIBLE_TOKEN
    return ranked[0][1] or ILLEGIBLE_TOKEN


def refine_region_vlm_r3(qwen_model, processor, image_path, region, device, img_w, img_h):
    rtype = normalize_type(region.get("type"))
    message_variants = [crop_messages(image_path, region), vlm_r3_refine_messages(image_path, region, img_w, img_h)]
    if rtype == "table":
        table_region = dict(region)
        table_region["type"] = "table"
        message_variants.append(vlm_r3_refine_messages(image_path, table_region, img_w, img_h))
    try:
        outs = generate_batch(qwen_model, processor, message_variants, device, MAX_NEW_TOKENS_REFINE)
    except Exception as e:
        print("VLM-R3 refinement failed:", e, flush=True)
        torch.cuda.empty_cache()
        return ILLEGIBLE_TOKEN
    return choose_best_candidate(outs[:VLM_R3_MAX_REFINEMENT_PASSES], rtype)


def ocr_regions(qwen_model, processor, image_path, regions, device):
    with Image.open(image_path) as img:
        img_w, img_h = img.size
    crop_indices = [i for i, r in enumerate(regions) if should_crop_ocr(r)]
    high_risk_indices, normal_indices = [], []
    for i in crop_indices:
        risk = halp_region_score(qwen_model, processor, image_path, regions[i], device, img_w, img_h)
        forced = normalize_type(regions[i].get("type")) in VLM_R3_FORCE_TYPES
        if risk >= HALP_RISK_THRESHOLD or forced:
            high_risk_indices.append(i)
        else:
            normal_indices.append(i)

    for start in range(0, len(normal_indices), CROP_BATCH_SIZE):
        batch_indices = normal_indices[start:start + CROP_BATCH_SIZE]
        msgs = [crop_messages(image_path, regions[i]) for i in batch_indices]
        try:
            outs = generate_batch(qwen_model, processor, msgs, device, MAX_NEW_TOKENS_CROP)
        except Exception as e:
            print("Crop OCR batch failed:", e, flush=True)
            torch.cuda.empty_cache()
            gc.collect()
            continue
        for idx, out in zip(batch_indices, outs):
            text = clean_crop_text(out)
            if text:
                regions[idx]["text"] = text

    for idx in high_risk_indices:
        text = refine_region_vlm_r3(qwen_model, processor, image_path, regions[idx], device, img_w, img_h)
        regions[idx]["text"] = text if text else ILLEGIBLE_TOKEN

    for region in regions:
        if normalize_type(region.get("type")) in TEXT_TYPES and not str(region.get("text") or "").strip():
            region["text"] = ILLEGIBLE_TOKEN
    return sort_regions([strip_internal_fields(r) for r in regions])


def infer_one_image(qwen_model, processor, yolo_model, image_path, device, yolo_device):
    yolo_regions = detect_regions_yolo(yolo_model, image_path, yolo_device)
    qwen_regions = detect_regions_qwen_page(qwen_model, processor, image_path, device)
    regions = merge_qwen_yolo_regions(qwen_regions, yolo_regions)
    if not regions:
        return []
    regions = ocr_regions(qwen_model, processor, image_path, regions, device)
    return sort_regions(regions)
'''


def patch_train_nb(nb, stage):
    title = "Stage 1: Silver Warm-up" if stage == 1 else "Stage 2: Gold Fine-tune"
    set_src(nb["cells"][0], f"# RUKOPYS Qwen3-VL {title} (VLM-R3 + HALP)\n")

    cell2 = src_text(nb["cells"][2])
    cell2 = cell2.replace('PROMPT_VERSION = "hybrid_prompt_v2"', 'PROMPT_VERSION = "vlm_r3_halp_v1"')
    cell2 = cell2.replace("same hybrid_prompt_v2 instructions", "same vlm_r3_halp_v1 instructions")
    if stage == 1:
        cell2 = cell2.replace(
            'OUTPUT_DIR = Path("/kaggle/working/qwen3vl_rukopys_stage1_silver_vlm_r3_halp")',
            'OUTPUT_DIR = Path("/kaggle/working/qwen3vl_rukopys_stage1_silver_vlm_r3_halp")',
        )
        cell2 = cell2.replace(
            'OUTPUT_DIR = Path("/kaggle/working/qwen3vl_rukopys_stage1_silver_hybrid_prompt_v2")',
            'OUTPUT_DIR = Path("/kaggle/working/qwen3vl_rukopys_stage1_silver_vlm_r3_halp")',
        )
    else:
        cell2 = cell2.replace(
            'OUTPUT_DIR = Path("/kaggle/working/qwen3vl_rukopys_stage2_gold_hybrid_prompt_v2")',
            'OUTPUT_DIR = Path("/kaggle/working/qwen3vl_rukopys_stage2_gold_vlm_r3_halp")',
        )
        cell2 = re.sub(
            r"STAGE1_LORA_CANDIDATES\s*=\s*\[\n.*?\]\n",
            STAGE1_LORA_CANDIDATES_STRICT,
            cell2,
            count=1,
            flags=re.S,
        )
    cell2 = re.sub(
        r"(CROP_PAD_RATIO\s*=\s*[0-9.]+\n)",
        r"\1" + TRAIN_EXTRA_CONFIG,
        cell2,
        count=1,
    )
    cell2 = cell2.replace(
        '"Do not solve, simplify, normalize, explain, or convert old notation into a different style."',
        '"Do not solve, simplify, normalize, explain, or convert old notation into a different style. If unreadable, return [illegible]."',
    )
    cell2 = cell2.replace(
        '"Do not infer missing cells, rebalance columns, summarize, or explain."',
        '"Do not infer missing cells, rebalance columns, summarize, or explain. If unreadable, return [illegible]."',
    )
    cell2 = cell2.replace(
        '"or explain."',
        '"or explain. If unreadable, return [illegible]."',
    )
    set_src(nb["cells"][2], cell2)

    if stage == 2:
        cell3 = src_text(nb["cells"][3])
        cell3 = re.sub(
            r"def find_stage1_lora_dir\(\):\n.*?\n\ndef get_dataset_root\(\):",
            '''def find_stage1_lora_dir():
    for item in STAGE1_LORA_CANDIDATES:
        p = Path(item)
        if (p / "adapter_config.json").exists():
            return p
    searched = chr(10).join(f"  - {p}" for p in STAGE1_LORA_CANDIDATES)
    raise FileNotFoundError(
        "No VLM-R3+HALP Stage 1 LoRA adapter_config.json found. Checked:" + chr(10) + searched
    )


def get_dataset_root():''',
            cell3,
            count=1,
            flags=re.S,
        )
        set_src(nb["cells"][3], cell3)

    if stage == 1:
        set_src(nb["cells"][4], STAGE_BUILD_CELL + STAGE1_POST_BUILD)
    else:
        set_src(nb["cells"][4], STAGE_BUILD_CELL + STAGE2_POST_BUILD)
    set_src(nb["cells"][6], TRAIN_COLLATOR_CELL)

    cell7 = src_text(nb["cells"][7])
    cell7 = cell7.replace(
        '"base_model": str(model_id),',
        '"vlm_r3_action_prompt": VLM_R3_ACTION_PROMPT,\n'
        '        "vlm_r3_final_prompts": VLM_R3_FINAL_PROMPTS,\n'
        '        "end_to_end_page_repeat": END_TO_END_PAGE_REPEAT,\n'
        '        "halp_small_crop_area_ratio": HALP_SMALL_CROP_AREA_RATIO,\n'
        '        "vlm_r3_formula_table_repeat": VLM_R3_FORMULA_TABLE_REPEAT,\n'
        '        "illegible_token": ILLEGIBLE_TOKEN,\n'
        '        "base_model": str(model_id),',
    )
    set_src(nb["cells"][7], cell7)

    if stage == 2:
        nb["cells"].append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": HALP_PROBE_CELL.splitlines(keepends=True),
            }
        )
    return nb


def patch_submit_nb(nb):
    set_src(nb["cells"][0], "# RUKOPYS YOLO + Qwen3-VL Submit (VLM-R3 + HALP)\n")
    cell2 = src_text(nb["cells"][2])
    cell2 = cell2.replace("from PIL import Image\n", SUBMIT_IMPORT_PATCH)
    cell2 = cell2.replace("OUTPUT_CSV = \"submission.csv\"", "OUTPUT_CSV = \"submission_vlm_r3_halp.csv\"")
    cell2 = cell2.replace("CROP_OCR_MODE = \"all_text\"  # choices: \"none\", \"smart\", \"all_text\"", "CROP_OCR_MODE = \"halp_vlm_r3\"  # choices: \"none\", \"smart\", \"all_text\", \"halp_vlm_r3\"")
    cell2 = cell2.replace(
        "HYBRID_PARTIAL_PREFIX = \"hybrid_prompt_v2_partial_results_gpu\"",
        "HYBRID_PARTIAL_PREFIX = \"vlm_r3_halp_partial_results_gpu\"",
    )
    cell2 = cell2.replace(
        "MAX_NEW_TOKENS_CROP = 192\nCROP_PAD_RATIO = 0.04\n",
        "MAX_NEW_TOKENS_CROP = 192\nCROP_PAD_RATIO = 0.04\n" + SUBMIT_EXTRA_CONFIG,
    )
    cell2 = re.sub(
        r"LORA_CANDIDATES\s*=\s*\[\n.*?\]\n",
        SUBMIT_LORA_CANDIDATES_STRICT,
        cell2,
        count=1,
        flags=re.S,
    )
    cell2 = cell2.replace(
        'VALIDATION_RECORDS_CANDIDATES = [\n',
        'VALIDATION_RECORDS_CANDIDATES = [\n    "/kaggle/working/qwen3vl_rukopys_stage2_gold_vlm_r3_halp/gold_validation_records.jsonl",\n    "/kaggle/input/qwen3vl-rukopys-stage2-vlm-r3-halp/gold_validation_records.jsonl",\n',
    )
    cell2 = cell2.replace(
        '"Do not solve, simplify, normalize, explain, or convert old notation into a different style."',
        '"Do not solve, simplify, normalize, explain, or convert old notation into a different style. If unreadable, return [illegible]."',
    )
    cell2 = cell2.replace(
        '"or explain."',
        '"or explain. If unreadable, return [illegible]."',
    )
    cell2 = cell2.replace("CROP_PROMPTS = {\n", SUBMIT_PAGE_PROMPT_INSERT + "CROP_PROMPTS = {\n")
    set_src(nb["cells"][2], cell2)

    cell3 = src_text(nb["cells"][3])
    cell3 = re.sub(
        r"def find_lora_dir\(\):\n.*?\n\ndef find_yolo_weights\(\):",
        '''def find_lora_dir():
    for item in LORA_CANDIDATES:
        p = Path(item)
        if (p / "adapter_config.json").exists():
            return p
    searched = chr(10).join(f"  - {p}" for p in LORA_CANDIDATES)
    raise FileNotFoundError(
        "No VLM-R3+HALP Stage 2 LoRA adapter_config.json found. Checked:" + chr(10) + searched
    )


def find_yolo_weights():''',
        cell3,
        count=1,
        flags=re.S,
    )
    cell3 = cell3.replace('OUTPUT_CSV = "hybrid_validation_pred.csv"', 'OUTPUT_CSV = "vlm_r3_halp_validation_pred.csv"')
    cell3 = cell3.replace('OUTPUT_CSV = "submission.csv"', 'OUTPUT_CSV = "submission_vlm_r3_halp.csv"')
    set_src(nb["cells"][3], cell3)

    cell5 = src_text(nb["cells"][5])
    cell5 = cell5.replace(
        'def strip_internal_fields(region):\n    return {"bbox": region["bbox"], "type": normalize_type(region.get("type")), "text": str(region.get("text") or "")}\n',
        'def strip_internal_fields(region):\n    return {"bbox": region["bbox"], "type": normalize_type(region.get("type")), "text": str(region.get("text") or "")}\n',
    )
    set_src(nb["cells"][5], cell5)

    cell6 = src_text(nb["cells"][6])
    cell6 = cell6.replace(
        "    return sort_regions([strip_internal_fields(r) for r in kept])",
        "    return sort_regions(kept)",
    )
    set_src(nb["cells"][6], cell6)

    set_src(nb["cells"][7], SUBMIT_QWEN_CELL)
    set_src(nb["cells"][8], SUBMIT_INFERENCE_CELL)
    return nb


def main():
    DST.mkdir(parents=True, exist_ok=True)
    stage1 = patch_train_nb(
        read_nb(SRC / "rukopys_qwen3vl_stage1_silver_finetune_hybrid_prompt_v2.ipynb"),
        stage=1,
    )
    stage2 = patch_train_nb(
        read_nb(SRC / "rukopys_qwen3vl_stage2_gold_finetune_hybrid_prompt_v2.ipynb"),
        stage=2,
    )
    submit = patch_submit_nb(read_nb(SRC / "rukopys_yolo_qwen3vl_hybrid_submit.ipynb"))

    write_nb(DST / "rukopys_qwen3vl_stage1_silver_finetune_vlm_r3_halp.ipynb", stage1)
    write_nb(DST / "rukopys_qwen3vl_stage2_gold_finetune_vlm_r3_halp.ipynb", stage2)
    write_nb(DST / "rukopys_yolo_qwen3vl_vlm_r3_halp_submit.ipynb", submit)


if __name__ == "__main__":
    main()
