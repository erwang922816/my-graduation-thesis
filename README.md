# Fine-Grained Image Retrieval Baseline

This project is refactored around a `ResNet50`-based fine-grained image retrieval pipeline.

## What is included

- `ResNet50` retrieval baseline with embedding head
- `SE` and `CBAM` attention variants
- `Triplet Loss` and `CrossEntropy + Triplet Loss`
- Evaluation with `mAP` and `Recall@K`
- Retrieval result visualization and feature projection
- Experiment comparison template for thesis writing

## Supported dataset layouts

### 1. Generic folder layout

```text
dataset_root/
  train/
    class_a/*.jpg
    class_b/*.jpg
  query/
    class_a/*.jpg
    class_b/*.jpg
  gallery/
    class_a/*.jpg
    class_b/*.jpg
```

If `query/` and `gallery/` do not exist, the code falls back to `val/`.

### 2. Stanford Online Products

Place the official files under one root:

```text
Stanford_Online_Products/
  Ebay_train.txt
  Ebay_test.txt
  bicycle_final/...
  chair_final/...
```

## Quick start

### Train baseline

```bash
python train.py \
  --data-root /path/to/dataset \
  --dataset-type folder \
  --model-name resnet50_retrieval \
  --pretrained \
  --loss-type joint \
  --output-dir outputs/resnet50_baseline
```

### Train on Stanford Online Products

Recommended dataset path:

```text
/Users/zhoutao/PycharmProjects/pytorch_train/data/Stanford_Online_Products
```

Quick run:

```bash
bash scripts/train_sop_baseline.sh
```

Detailed SOP setup:

- [SOP数据接入与环境配置说明.md](/Users/zhoutao/PycharmProjects/pytorch_train/SOP数据接入与环境配置说明.md:1)

### Train SE / CBAM model

```bash
python train.py --data-root /path/to/dataset --model-name resnet50_se --pretrained
python train.py --data-root /path/to/dataset --model-name resnet50_cbam --pretrained
```

### Evaluate

```bash
python test.py \
  --data-root /path/to/dataset \
  --checkpoint outputs/resnet50_baseline/best.pth \
  --model-name resnet50_retrieval
```

### Visualize retrieval results

```bash
python visualize_retrieval.py \
  --data-root /path/to/dataset \
  --checkpoint outputs/resnet50_baseline/best.pth \
  --model-name resnet50_retrieval \
  --output-dir outputs/vis_baseline
```

## Suggested experimental order

1. Train `resnet50_retrieval` as the baseline.
2. Train `resnet50_se` or `resnet50_cbam`.
3. Compare `triplet` and `joint` losses.
4. Fill `experiments/comparison_template.md`.
