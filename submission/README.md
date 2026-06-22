# Handwritten to Data - Offline Submission

This Docker pipeline runs fully offline for the Handwritten to Data task:

1. DocLayout-YOLO V4.1 and V4.2 detect bounding boxes and region types.
2. TrOCR processes `handwritten`, `printed`, and `annotation` regions.
3. Qwen3-VL-8B-Instruct with a LoRA adapter processes `formula` and `table` regions.
4. Results are written to `/data/output/submission.csv`.

## Requirements

- Docker.
- An NVIDIA GPU with CUDA and NVIDIA Container Toolkit.
- Enough disk space for the models and Docker image.
- No Internet connection is required at runtime; all models must be available locally before building.

## Prepare the Weights

The model weights are not stored in this GitHub repository. Download the [HTD Boustoichoi Weights dataset](https://www.kaggle.com/datasets/huylhn1810/htd-boustoichoi-weights) from Kaggle:

```bash
python -m pip install kaggle
kaggle datasets download -d huylhn1810/htd-boustoichoi-weights -p .
unzip htd-boustoichoi-weights.zip -d .
```

After extraction, the `submission` directory should have the following structure:

```text
submission/
|-- Dockerfile
|-- entrypoint.sh
|-- inference.py
|-- requirements.txt
|-- DoclayoutYoloV4.1.pt
|-- DoclayoutYoloV4.2.pt
|-- trocr_model/
|   |-- config.json
|   `-- model.safetensors
|-- qwen3vl_8b_instruct/
|   |-- config.json
|   |-- model-00001-of-00004.safetensors
|   |-- model-00002-of-00004.safetensors
|   |-- model-00003-of-00004.safetensors
|   `-- model-00004-of-00004.safetensors
`-- qwen3vl_lora_adapter/
    |-- adapter_config.json
    `-- adapter_model.safetensors
```

Keep all tokenizer, processor, and configuration files included with each model, not only the files shown above.

## Build the Image

Run the following command from the `submission` directory:

```bash
docker build --progress=plain -t my-htr:latest .
```

## Run GPU Inference

The input can be a directory containing images directly, or a directory containing an `images/` folder and a `metadata.jsonl` file.

### Linux or WSL

```bash
mkdir -p output

docker run --rm --gpus all --shm-size=8g --network none \
  -v "$(realpath ./data_test):/data/input:ro" \
  -v "$(realpath ./output):/data/output" \
  my-htr:latest
```

### Windows PowerShell

```powershell
New-Item -ItemType Directory -Force .\output | Out-Null

docker run --rm --gpus all --shm-size=8g --network none `
  -v "$(Resolve-Path .\data_test):/data/input:ro" `
  -v "$(Resolve-Path .\output):/data/output" `
  my-htr:latest
```

The result is written to `output/submission.csv`. On successful completion, the pipeline validates the row count, writes one row for every input image, and exits with code `0`.
