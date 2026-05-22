# Stage A — Phase 1 Layout-only Prompt

Phase 1 Stage A returns only `bbox,type`.

It must not return:

```text
text
text_draft
reasoning
markdown
```

The executable prompt lives in `MainPipeline/src/common/prompts.py` so train and inference use exactly the same text.

