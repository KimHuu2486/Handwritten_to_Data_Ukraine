import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "simple" / "rukopys_yolo_qwen3vl_ppstructurev3_submit.ipynb"
DST = ROOT / "finetune" / "ukrainian_curriculum_clean" / "03_submission_yolo_qwen3vl_stage2_no_pp.ipynb"


def read_nb(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_nb(path, nb):
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def cell_text(cell):
    return "".join(cell.get("source", []))


def set_cell_text(cell, text):
    cell["source"] = text.splitlines(keepends=True)


def replace_once(text, old, new):
    if old not in text:
        raise ValueError(f"Could not find expected text:\n{old[:500]}")
    return text.replace(old, new, 1)


def main():
    nb = read_nb(SRC)

    # Title/description.
    set_cell_text(
        nb["cells"][0],
        """# RUKOPYS Submission: YOLO + Qwen3-VL Ukrainian Curriculum Stage 2

Kaggle inference notebook: YOLO detects `bbox`/`type`; Qwen3-VL 8B Instruct performs OCR with the Ukrainian curriculum Stage 2 LoRA adapter. PP-StructureV3/PaddleOCR is disabled by default to reduce runtime and installation risk.
""",
    )

    config = cell_text(nb["cells"][1])
    config = config.replace("INSTALL_PADDLEOCR = True", "INSTALL_PADDLEOCR = False")
    config = config.replace("PADDLE_INSTALL_MODE = 'cpu'", "PADDLE_INSTALL_MODE = 'skip'")
    old_lora = """# Strongest Qwen OCR adapter: Stage 2 gold LoRA from the finetune pipeline.
USE_QWEN_LORA = True
LORA_CANDIDATES = [
    '/kaggle/input/datasets/lhongthyan/htd-fine-tune-v2-dataset/qwen3vl_rukopys_stage2_gold/qwen3vl_rukopys_lora_final',
    '/kaggle/input/qwen3vl-rukopys-curriculum/qwen3vl_rukopys_lora_final',
    '/kaggle/input/qwen3vl-rukopys-lora/qwen3vl_rukopys_lora_final',
]
LORA_FALLBACK_SEARCH = True
LOAD_LORA_CROP_PROMPTS = False  # keep the stronger PP/simple prompts unless you explicitly want adapter-saved prompts
"""
    new_lora = """# Ukrainian curriculum final OCR adapter from 02_stage2_final_format_finetune.ipynb.
USE_QWEN_LORA = True
LORA_CANDIDATES = [
    # Common Kaggle input layouts. Keep the first existing path.
    '/kaggle/input/qwen3vl-ukrainian-stage2-final-format/qwen3vl_ukrainian_stage2_final_lora_final',
    '/kaggle/input/qwen3vl-ukrainian-stage2-final-format/qwen3vl_ukrainian_stage2_final_format/qwen3vl_ukrainian_stage2_final_lora_final',
    '/kaggle/input/qwen3vl-ukrainian-stage2-final/qwen3vl_ukrainian_stage2_final_lora_final',
    '/kaggle/input/ukrainian-curriculum-stage2/qwen3vl_ukrainian_stage2_final_lora_final',
    '/kaggle/input/ukrainian-curriculum-stage2/qwen3vl_ukrainian_stage2_final_format/qwen3vl_ukrainian_stage2_final_lora_final',
    '/kaggle/input/qwen3vl-ukrainian-curriculum-stage2/qwen3vl_ukrainian_stage2_final_lora_final',
    '/kaggle/working/qwen3vl_ukrainian_stage2_final_format/qwen3vl_ukrainian_stage2_final_lora_final',
]
LORA_FALLBACK_SEARCH = True
LOAD_LORA_CROP_PROMPTS = True  # match prompts saved by the Stage 2 adapter
"""
    config = replace_once(config, old_lora, new_lora)
    config = config.replace(
        "PARTIAL_PREFIX = 'yolo_qwen_lora_pp_partial_gpu'",
        "PARTIAL_PREFIX = 'yolo_qwen_ukr_stage2_no_pp_partial_gpu'",
    )
    config = config.replace(
        "# PP-StructureV3 table helper. CPU is safer on Kaggle because Qwen 8B already occupies VRAM.\nPP_ENABLE = True",
        "# PP-StructureV3 table helper. Disabled by default; enable only if table hints improve validation.\nPP_ENABLE = False",
    )
    set_cell_text(nb["cells"][1], config)

    helpers = cell_text(nb["cells"][3])
    old_score = """                    score = sum(token in str(root_path).lower() for token in ('stage2', 'gold', 'rukopys', 'qwen3vl'))
                    candidates.append((score, root_path))
            if candidates:
                return sorted(candidates, key=lambda x: (-x[0], str(x[1])))[0][1]
    raise FileNotFoundError('No Qwen LoRA adapter_config.json found. Add the Stage 2 gold LoRA Kaggle input or set USE_QWEN_LORA=False.')
"""
    new_score = """                    low = str(root_path).lower()
                    score = sum(token in low for token in ('ukrainian', 'curriculum', 'stage2', 'final', 'rukopys', 'qwen3vl'))
                    score += 3 * int('qwen3vl_ukrainian_stage2_final_lora_final' in low)
                    score -= 3 * sum(token in low for token in ('stage0', 'stage1', 'silver'))
                    candidates.append((score, root_path))
            if candidates:
                return sorted(candidates, key=lambda x: (-x[0], str(x[1])))[0][1]
    raise FileNotFoundError('No Qwen LoRA adapter_config.json found. Add the Ukrainian Stage 2 final LoRA Kaggle input or set USE_QWEN_LORA=False.')
"""
    helpers = replace_once(helpers, old_score, new_score)
    helpers = helpers.replace(
        "print('Using LoRA adapter:', lora_dir, flush=True)",
        "print('Using Ukrainian Stage 2 LoRA adapter:', lora_dir, flush=True)",
    )
    set_cell_text(nb["cells"][3], helpers)

    write_nb(DST, nb)
    print("Wrote", DST)


if __name__ == "__main__":
    main()
