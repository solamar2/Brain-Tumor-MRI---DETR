# Brain Tumor Detection and Segmentation in MRI

## Overview
The objective of this project is to develop an artificial intelligence system capable of detecting and localizing brain tumors in two-dimensional MRI images. The system combines classical image processing techniques with modern deep learning models, particularly Transformer-based architectures designed for object detection.
The final system is designed to perform two main tasks:
•	Tumor Localization: Identify the spatial location of tumors within the MRI image using bounding boxes.
•	Tumor Classification: Classify the detected tumor according to its type.
The system aims to automate and assist in the early detection of brain tumors, which may support medical professionals by providing a fast and consistent preliminary analysis of MRI scans.

It includes:
1. **EDA** – exploratory data analysis for MRI datasets.  
2. **Segmentation** – brain extraction to remove irrelevant background and isolate brain tissue.  
3. **Preprocessing** – cropping images according to brain masks and preparing dataset for training.  
4. **Dataset & DataLoader** – PyTorch Dataset and DataLoader with weighted sampling and custom collate function for detection tasks.  
5. **Detection** – model training DETR and evaluation loops.

---
## Dataset
Kaggle: MRI for Brain Tumor with Bounding Boxes
https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes/data
- Input: MRI images in `.jpg` format with corresponding **YOLO-style labels** and brain masks.  
- Labels format: `<class_id> <x_center> <y_center> <width> <height>` (normalized).  
---
## Project Structure
Brain Tumor MRI/
│
├── data/
│   ├── TRAIN/
│   │   ├── images/       # MRI scans in JPG format
│   │   ├── labels/       # Annotation files describing tumor bounding boxes
│   │   └── brain_masks/  # Masks generated during preprocessing in this project
│   │
│   └── TEST/
│       ├── images/
│       ├── labels/
│       └── brain_masks/
│
├── scripts/
│   ├── dataloader.py       # Custom PyTorch DataLoader with weighted sampling
│   ├── detection.py        # Model definition and training loop
│   ├── EDA.py              # Exploratory Data Analysis tools
│   ├── preprocessing.py    # Preprocessing functions (crop, resize, augmentation)
│   ├── segmentation.py     # Brain extraction / mask generation
│   └── main.py             # Entry point for training/evaluation
