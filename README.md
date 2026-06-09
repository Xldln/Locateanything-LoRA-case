# LocateAnything LoRA Fine-tuning Case

Fine-tune [NVIDIA LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) on custom detection datasets.

## Structure

```
├── pkg/                           # Dependency packages (LocateAnything, YOLO)
├── datasets/robot-detection/      # Custom dataset
├── locateanything_worker.py       # Inference worker
├── locate-anything-streaming.sh   # Training launcher
├── build_env.sh                   # Environment setup script
├── generate_datasets.ipynb        # Generate annotations via YOLO
├── check_model_predict.ipynb      # Verify predictions
└── process_data.ipynb             # Data preprocessing
```

## Setup

```bash
bash build_env.sh
```

This creates a `lam` conda environment with Python 3.10, installs torch (CUDA 12.8), EagleVL, and extra dependencies (`hf_xet`, `sortedcontainers`, `pynvml`, `tensorboard`).

## Training

```bash
bash locate-anything-streaming.sh
```

The script auto-activates the `lam` conda environment and launches training with DeepSpeed ZeRO-1, MTP, and stream packing. Config via env: `GPUS`, `MODEL_PATH`, `OUTPUT_DIR`.

Default: 1-GPU LoRA fine-tuning. Override with `GPUS=8 bash locate-anything-streaming.sh`.

## Inference

```python
from locateanything_worker import LocateAnythingWorker

worker = LocateAnythingWorker("nvidia/LocateAnything-3B")
result = worker.detect(img, ["person", "car"])
result = worker.ground_multi(img, "people wearing red shirts")
boxes = LocateAnythingWorker.parse_boxes(result["answer"], w, h)
```

Tasks: detection, phrase grounding, text grounding, GUI grounding, pointing.

## Dataset Format

```jsonl
{"conversations": [{"from": "human", "value": "Locate all the instances that matches the following description: pencilbag</c>pig zipper"}, {"from": "gpt", "value": "<ref>pencilbag</ref><box><442><487><732><735></box><ref>pig zipper</ref><box><369><583><448><664></box>"}], "image": "path/to/image.png"}
```

Coordinates normalized to `[0, 1000]`.
