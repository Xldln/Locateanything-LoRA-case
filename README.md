# LocateAnything LoRA Fine-tuning Case

Fine-tune [NVIDIA LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) on custom detection datasets.

## Structure

```
├── pkg/                           # Dependency packages (LocateAnything, YOLO)
├── datasets/robot-detection/      # Custom dataset
├── locateanything_worker.py       # Inference worker
├── locate-anything-streaming.sh   # Training launcher
├── generate_datasets.ipynb        # Generate annotations via YOLO
├── check_model_predict.ipynb      # Verify predictions
└── process_data.ipynb             # Data preprocessing
```

## Setup

```bash
conda create -n lam python=3.10 -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

cd pkg/yolo && pip install -e .
cd pkg/eaglevl && pip install -e .
```

## Training

```bash
cd pkg/eaglevl
bash ../../locate-anything-streaming.sh
```

8-GPU training with DeepSpeed ZeRO-1, MTP, and stream packing. Config via env: `GPUS`, `MODEL_PATH`, `OUTPUT_DIR`.

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
