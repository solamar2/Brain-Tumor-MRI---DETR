from scripts.EDA import BrainTumorEDA
from scripts.segmentation import BrainMaskSimple
from scripts.preprocessing import BrainTumorDataset
from scripts.dataloader import BrainTumorDataLoader
import os
import random
from pathlib import Path

main_path = "../"
main_train_path= os.path.join(main_path, 'data/Train')

# ------------------------------
## 0. EDA:

eda = BrainTumorEDA(main_train_path)
eda.plot_class_counts_and_examples()
eda.plot_bb_analysis()
eda.plot_image_stats()

# ------------------------------
## 1. Segment brain vs background
# Collect class dirs that need segmentation

segmenter = BrainMaskSimple(train_dir=main_train_path)
segmenter.run()

# Random visualization (one image per class)
for cls_name in os.listdir(main_train_path):
    cls_dir = os.path.join(main_train_path, cls_name)
    images_dir = os.path.join(cls_dir, "images")
    masks_dir = os.path.join(cls_dir, "brain_masks")
    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        continue

    img_file = random.choice([f for f in os.listdir(images_dir) if f.endswith(".jpg")])
    img_path = os.path.join(images_dir, img_file)
    mask_path = os.path.join(masks_dir, img_file)
    segmenter.create_overlay(img_path, mask_path, alpha=0.3)
    
# ------------------------------
## 2. Pre process the data
traindataset = BrainTumorDataset(train_dir=main_train_path,image_size=(256, 256),augment=False)
for i in range (0,5):
    traindataset.visualize_sample()

# ------------------------------
## 3. Dataloader
loader_builder = BrainTumorDataLoader(traindataset, batch_size=8, weighted_sampling=True)
train_loader = loader_builder.get_loader()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




