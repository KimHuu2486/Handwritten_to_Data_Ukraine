from __future__ import annotations

from .schema import STRUCTURAL_TYPES, normalize_type


STAGE_A_LAYOUT_ONLY_BASE = """Extract layout regions from the full page.
Return only a compact JSON array.
Each item must have keys bbox,type.
bbox is [x1,y1,x2,y2] on a 0-1000 grid.
type is one of handwritten, printed, formula, table, annotation, image, graph.

Prioritize bbox and type accuracy over transcription.
Do not include text or text_draft.

Granularity rules:
- handwritten, printed, formula, annotation: one bbox per visual line or standalone item.
- table: one bbox for the full table, not one bbox per cell.
- image: one bbox for drawings, stamps, seals, illustrations, or non-text figures.
- graph: one bbox for charts, coordinate plots, axes-based plots, or data visualizations.

Do not merge separate text lines.
Do not split a single visual table into cells.
Do not force drawings, diagrams, or graphs into handwritten/printed/formula.
Detect small but meaningful regions such as numbering, teacher marks, dates, grades, and short annotations.
Preserve reading order.
Return only JSON."""


SOURCE_LAYOUT_HINTS = {
    "dictation": (
        "This is a Ukrainian national dictation page, usually prose handwriting captured by phone. "
        "Detect all visible document regions in reading order. Return bbox and type for each region. "
        "Do not infer missing lines from the canonical dictation text."
    ),
    "archive": (
        "This is an archival document from 1919-1935. It may contain handwriting, typewritten or printed headers, "
        "stamps, old Cyrillic/Ukrainian orthography, and dense administrative layout. "
        "Detect all meaningful text and non-text regions. Classify handwritten vs printed carefully. "
        "Do not modernize or complete content from context."
    ),
    "school": (
        "This is a school homework page. It may contain handwritten answers, printed fragments, formulas, tables, "
        "teacher annotations, drawings, diagrams, coordinate plots, and charts. Detect all meaningful regions. "
        "Classify drawings/illustrations as image, and axes-based plots/charts as graph. "
        "Do not force non-text visuals into handwritten, printed, or formula."
    ),
    "university": (
        "This is a university exam or coursework page. It may contain handwritten text, printed text, "
        "mathematical or chemical formulas, tables, diagrams, plotted graphs, coordinate charts, and scientific figures. "
        "Detect all meaningful regions. Classify standalone equations or chemistry notation as formula, "
        "tabular structures as table, plotted axes/charts as graph, and diagrams/figures as image."
    ),
}

SOURCE_OCR_HINTS = {
    "dictation": "Ukrainian dictation handwriting. Do not complete from canonical text; read only visible characters.",
    "archive": "Historical Ukrainian/Cyrillic document. Preserve old spelling; do not modernize.",
    "school": "School homework. It may contain corrections, teacher marks, formulas, and mixed handwriting/print.",
    "university": "University exam/coursework. It may contain formulas, tables, chemistry notation, and technical symbols.",
}

TYPE_OCR_INSTRUCTIONS = {
    "handwritten": "Transcribe the visible text exactly. Preserve punctuation, line content, corrections, and strikethrough markers. Return only text.",
    "printed": "Transcribe the visible printed or typewritten text exactly. Preserve punctuation and visible spelling. Return only text.",
    "formula": (
        "Read this standalone math, logic, vector, matrix, determinant, set/relation, statistics, physics, "
        "or chemistry expression exactly as written. Return only formula text, using LaTeX when it is the clearest "
        "representation and plain Unicode when it better matches the handwriting. Preserve visible symbols, indices, "
        "superscripts, subscripts, arrows, fractions, matrix/determinant structure, punctuation, numbering, and "
        "strikethrough/correction markers. Do not solve, simplify, normalize, explain, or convert old notation "
        "into a different style."
    ),
    "table": (
        "Read this table region exactly. Return only pipe-separated table text. Use one output line per visual row "
        "and `|` between cells. Preserve empty cells with empty fields, e.g. `A||C`. Preserve row order, column order, "
        "multi-word cell text, wrapped cell text, numbers, units, punctuation, dashes, and visible spelling mistakes. "
        "Do not infer missing cells, do not rebalance columns, do not summarize, and do not explain."
    ),
    "annotation": "Read this short annotation or teacher mark. Return only the exact visible text.",
    "image": "Return an empty string.",
    "graph": "Return an empty string.",
}


def stage_a_prompt(source: str) -> str:
    """Build the source-aware layout-only prompt for Phase 1 Stage A."""
    source_hint = SOURCE_LAYOUT_HINTS.get(str(source or "").lower(), "")
    return f"{STAGE_A_LAYOUT_ONLY_BASE}\n\n{source_hint}".strip()


def stage_b_prompt(source: str, region_type: str) -> str:
    """Build the type-aware crop OCR prompt for Phase 1 Stage B."""
    normalized_type = normalize_type(region_type)
    if normalized_type in STRUCTURAL_TYPES:
        return TYPE_OCR_INSTRUCTIONS[normalized_type]
    source_hint = SOURCE_OCR_HINTS.get(str(source or "").lower(), "")
    type_instruction = TYPE_OCR_INSTRUCTIONS[normalized_type]
    return (
        f"{source_hint}\n"
        f"{type_instruction}\n"
        "Use [illegible] only for unreadable words inside an otherwise legible text region.\n"
        "Use ~~word~~ for visible strikethrough and ~~old~~{new} for visible correction.\n"
        "Do not explain."
    ).strip()
