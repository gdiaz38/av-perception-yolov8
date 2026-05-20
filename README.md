# 👁️ Autonomous Vehicle Perception System — YOLOv8

A real-time object detection pipeline for autonomous vehicle perception, built on YOLOv8n trained on COCO128. Achieves mAP50 of 0.868 with 4.1ms inference per image — optimized for the speed/accuracy tradeoff required in production ADAS systems.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8n-mAP50_0.868-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-live-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📊 Live Dashboard

👉 **[View Live App](https://gdiaz38-av-perception-yolov8.streamlit.app)**

---

## Overview

Object detection is the foundation of every autonomous vehicle perception stack. This project trains YOLOv8n on COCO128, runs inference across 50 test images, and builds a full analytics dashboard over the 420 resulting detections — examining class frequency, confidence distributions, ADAS-critical object performance, and model benchmarks.

Key question it answers: *Which objects does the model detect most reliably, and how does it perform on ADAS-critical classes like cars, pedestrians, and traffic signs?*

---

## Key Results

| Metric | Value |
|---|---|
| mAP50 | **0.868** |
| mAP50-95 | **0.687** |
| Inference speed | **4.1ms / image** |
| Total detections | 420 across 50 images |
| Unique classes detected | 59 |
| Avg confidence | 0.643 |
| Person precision | 0.678 |
| Car precision | 0.561 |
| Traffic light detection | weakest ADAS class (0.361) |

---

## Features

- **Detection overview** — top 20 classes by count, confidence distribution boxplots, detections per image
- **Class performance** — confidence vs frequency scatter, highest confidence classes, ADAS-critical class breakdown
- **Detection explorer** — browse any of 50 test images, see all detections with confidence bars
- **Model summary** — YOLOv8 variant benchmark comparison, training config, top class breakdown
- **Search and filter** — search by class name, sort by count or confidence

---

## Data

| Source | Description |
|---|---|
| [COCO128](https://www.kaggle.com/datasets/ultralytics/coco128) | 128 images from MS COCO, used as training set |
| Test set | 50 held-out images from COCO128 |
| Inference hardware | Kaggle T4 GPU |

---

## Project Structure

```
av-perception-yolov8/
├── dashboard.py              # Streamlit dashboard
├── detections.csv            # 420 per-detection rows (image, class, confidence)
├── class_performance.csv     # 59 classes with count + avg confidence
├── model_summary.json        # mAP50, mAP50-95, inference time, top classes
├── load_mongodb.py           # MongoDB ingestion pipeline (original storage layer)
└── requirements.txt
```

---

## How It Works

```
COCO128 dataset (128 images)
        ↓
YOLOv8n training — 30 epochs on Kaggle T4 GPU
        ↓
Inference on 50 test images
  → 420 detections across 59 classes
  → per-detection confidence scores
        ↓
MongoDB storage layer (load_mongodb.py)
  → detections collection
  → class_performance collection
  → model_summary collection
        ↓
Streamlit dashboard — detection analytics
```

---

## Model Details

| Parameter | Value |
|---|---|
| Architecture | YOLOv8n |
| Parameters | 3.2M |
| Dataset | COCO128 |
| Epochs | 30 |
| Hardware | Kaggle T4 GPU |
| mAP50 | 0.868 |
| mAP50-95 | 0.687 |
| Inference | 4.1ms/image |

**YOLOv8 variant comparison:**

| Model | mAP50 | Params |
|---|---|---|
| **YOLOv8n (this)** | **0.868** | 3.2M |
| YOLOv8s | 0.892 | 11.2M |
| YOLOv8m | 0.905 | 25.9M |
| YOLOv8l | 0.918 | 43.7M |
| YOLOv8x | 0.925 | 68.2M |

YOLOv8n achieves 94% of YOLOv8x's accuracy at 5% of the parameter count — making it the right choice for real-time ADAS deployment.

---

## Top Detected Classes

| Class | Detections | Avg Confidence |
|---|---|---|
| person | 79 | 0.678 |
| book | 44 | 0.524 |
| chair | 41 | 0.529 |
| umbrella | 28 | 0.527 |
| donut | 20 | 0.865 |
| bottle | 17 | 0.570 |
| car | 16 | 0.561 |
| wine glass | 14 | 0.443 |
| elephant | 9 | 0.897 |
| dog | 4 | 0.932 |

---

## ADAS-Critical Class Performance

| Class | Detections | Avg Confidence | Notes |
|---|---|---|---|
| person | 79 | 0.678 | Most detected — critical for pedestrian safety |
| car | 16 | 0.561 | Core AV detection target |
| truck | 8 | 0.838 | High confidence |
| bus | 1 | 0.921 | Rare but high confidence |
| motorcycle | 3 | 0.530 | Mid confidence |
| traffic light | 4 | 0.361 | Weakest ADAS class — known challenge |
| stop sign | 1 | 0.910 | High confidence on single detection |

Traffic light detection at 0.361 confidence is a known weakness of YOLOv8n — larger variants (YOLOv8m+) improve this significantly.

---

## Dashboard Tabs

**Detection Overview** — top 20 class bar chart colored by confidence, confidence distribution boxplots for top 10 classes, detections per image histogram

**Class Performance** — confidence vs frequency scatter for all 59 classes, highest confidence class ranking, ADAS-critical dual-axis chart, searchable full class table

**Detection Explorer** — select any of 50 test images, view all detections with confidence bars, overall confidence distribution histogram

**Model Summary** — training configuration, top 10 class breakdown, YOLOv8 variant benchmark comparison chart

---

## Local Setup

```bash
git clone https://github.com/gdiaz38/av-perception-yolov8
cd av-perception-yolov8
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run dashboard.py
```

To retrain and re-run inference (requires COCO128 via kagglehub):

```bash
# Install ultralytics
pip install ultralytics kagglehub pymongo

# Train
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="coco128.yaml", epochs=30)

# Run inference and load results
python3 load_mongodb.py
```

---

## Tech Stack

`Python 3.11` · `YOLOv8 (Ultralytics)` · `Streamlit` · `Plotly` · `Pandas` · `NumPy` · `MongoDB`

---

## Affiliation

University of California, Riverside — MS in Engineering Management
Part of a portfolio of 10 live data science projects spanning computer vision, NLP, supply chain, and healthcare ML.

---

## License

MIT