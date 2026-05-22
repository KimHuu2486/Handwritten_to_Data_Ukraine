# Stage B — Phase 1 Crop OCR Prompt

Phase 1 Stage B receives:

```text
crop image
type
short source hint
type-specific instruction
```

It returns exact text only. For `image` and `graph`, the target and final output are always an empty string.

The executable prompt lives in `MainPipeline/src/common/prompts.py` so train and inference use exactly the same text.

