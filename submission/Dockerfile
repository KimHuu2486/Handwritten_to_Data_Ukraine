FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HPA_MODEL_DIR=/app/trocr_model \
    QWEN_BASE_MODEL_DIR=/app/qwen3vl_8b_instruct \
    QWEN_LORA_DIR=/app/qwen3vl_lora_adapter \
    YOLO_WEIGHTS=/app/DoclayoutYoloV4.1.pt \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    YOLO_AUTOINSTALL=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential cmake ninja-build pkg-config libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt

COPY inference.py /app/inference.py
COPY entrypoint.sh /app/entrypoint.sh
COPY trocr_model /app/trocr_model
COPY qwen3vl_lora_adapter /app/qwen3vl_lora_adapter
COPY DoclayoutYoloV4*.pt /app/

# Place the full Qwen3-VL-8B-Instruct local Hugging Face folder here before building.
# The directory must contain config.json and the model shard files.
COPY qwen3vl_8b_instruct /app/qwen3vl_8b_instruct

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
