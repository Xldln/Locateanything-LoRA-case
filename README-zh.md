# LocateAnything LoRA 微调示例

在自定义检测数据集上微调 [NVIDIA LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B)。

## 目录结构

```
├── pkg/                           # 依赖包（LocateAnything、YOLO）
├── datasets/robot-detection/      # 自定义数据集
├── locateanything_worker.py       # 推理封装
├── locate-anything-streaming.sh   # 训练启动脚本
├── generate_datasets.ipynb        # 用 YOLO 生成标注
├── check_model_predict.ipynb      # 验证预测结果
└── process_data.ipynb             # 数据预处理
```

## 环境配置

```bash
conda create -n lam python=3.10 -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

cd pkg/yolo && pip install -e .
cd pkg/locateanything && pip install -e .
```

## 训练

```bash
cd pkg/locateanything
bash ../../locate-anything-streaming.sh
```

8 卡训练，使用 DeepSpeed ZeRO-1、MTP 多 token 预测、stream packing。通过环境变量配置：`GPUS`、`MODEL_PATH`、`OUTPUT_DIR`。

## 推理

```python
from locateanything_worker import LocateAnythingWorker

worker = LocateAnythingWorker("nvidia/LocateAnything-3B")
result = worker.detect(img, ["person", "car"])
result = worker.ground_multi(img, "people wearing red shirts")
boxes = LocateAnythingWorker.parse_boxes(result["answer"], w, h)
```

支持：目标检测、短语定位、文字定位、GUI 定位、点选。

## 数据集格式

```jsonl
{"conversations": [{"from": "human", "value": "Locate all the instances that matches the following description: pencilbag</c>pig zipper"}, {"from": "gpt", "value": "<ref>pencilbag</ref><box><442><487><732><735></box><ref>pig zipper</ref><box><369><583><448><664></box>"}], "image": "path/to/image.png"}
```

坐标归一化到 `[0, 1000]`。
