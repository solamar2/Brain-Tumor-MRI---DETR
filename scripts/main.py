from scripts.EDA import BrainTumorEDA
from scripts.segmentation import BrainMaskSimple
from scripts.preprocessing import BrainTumorDataset
from scripts.dataloader import BrainTumorDataLoader
from scripts.detection_model_DETR import DETRTrainer
from scripts.evaluation import BrainTumorEvaluator
import torch
import os
import random
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

main_path = "../"
main_train_path= os.path.join(main_path, 'data/Train')
main_test_path = os.path.join(main_path, "data/Test")

# ------------------------------
## 0. EDA:
eda = BrainTumorEDA(main_train_path)
eda.plot_class_counts_and_examples()
eda.plot_bb_analysis()
eda.plot_image_stats()

# ------------------------------
# Class to idx: (same to train and test)
class_names = ['Glioma','Meningioma', 'No Tumor', 'Pituitary']
class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
num_classes = 4

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
train_dataset = BrainTumorDataset(main_train_path, class_to_idx, image_size=(256, 256), augment=True)
for i in range (0,5):
    train_dataset.visualize_sample()

# ------------------------------
## 3. Dataloader
train_loader = BrainTumorDataLoader(train_dataset, class_to_idx, batch_size=4, weighted_sampling=True).get_loader()

# ------------------------------
# 4. Train model
device = "cuda" if torch.cuda.is_available() else "cpu"
trainer = DETRTrainer(num_classes=num_classes, device=device, learning_rate=1e-4, weight_decay=1e-4, num_epochs=5, BB_weight=0.001)
trainer.train(train_loader, print_every=1)

# ------------------------------
# 5. Test:
segmenter = BrainMaskSimple(train_dir=main_test_path)
segmenter.run()
test_dataset  = BrainTumorDataset(main_test_path,  class_to_idx, image_size=(256, 256), augment=False)
test_loader  = BrainTumorDataLoader(test_dataset,  class_to_idx, batch_size=4, weighted_sampling=False, shuffle=False).get_loader()

# ------------------------------
# 6. Evaluation
evaluator = BrainTumorEvaluator(model_trainer=trainer, dataset=test_dataset, class_names=class_names, iou_thresh=0.5)

# Run evaluation on the entire test dataset
metrics_per_class, all_iou_values, all_gt, all_pred = evaluator.evaluate_dataset(conf_thresh=0.5)

# Print mAP@0.5 per class
print("=== mAP@0.5 per class ===")
for cls, m in metrics_per_class.items():
    print(f"{cls}: AP@0.5 = {m['AP@0.5']:.3f}, n_GT={m['n_GT']}, n_pred={m['n_pred']}")

# Plot IoU histograms for all classes
evaluator.plot_iou_histograms(all_iou_values)

# Plot Precision-Recall curves for each class
evaluator.plot_precision_recall_curves(all_gt, all_pred)

# Visualize a random sample with ground truth and predicted boxes
evaluator.visualize_sample(idx=random.randint(0, len(test_dataset)-1), conf_thresh=0.5, show_gt=True, show_pred=True)


    
    
    
    
    
    
    
    
    
    
    
    




