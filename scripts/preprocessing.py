import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class BrainTumorDataset(Dataset):
    """
    Dataset for brain tumor detection using:
    - brain mask based cropping
    - correct bounding box adaptation
    - fixed-size resize
    """

    def __init__(self, train_dir, class_to_idx, mask_dir_name="brain_masks", image_size=(256, 256), augment=True):
        self.samples = []
        self.image_size = image_size
        self.augment = augment
    
        
        for cls_name in os.listdir(train_dir):
            cls_dir = os.path.join(train_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue

            images_dir = os.path.join(cls_dir, "images")
            labels_dir = os.path.join(cls_dir, "labels")
            masks_dir = os.path.join(cls_dir, mask_dir_name)

            if not (os.path.exists(images_dir) and os.path.exists(labels_dir) and os.path.exists(masks_dir)):
                continue

            for img_file in os.listdir(images_dir):
                if not img_file.lower().endswith(".jpg"):
                    continue

                img_path = os.path.join(images_dir, img_file)
                label_path = os.path.join(labels_dir, img_file.replace(".jpg", ".txt"))
                mask_path = os.path.join(masks_dir, img_file)

                if os.path.exists(label_path) and os.path.exists(mask_path):
                    self.samples.append((img_path, label_path, mask_path, class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def _crop_by_mask(self, img, mask, padding=0):
        """
        Crop image to the tightest bounding box around the mask.
        """
        ys, xs = np.where(mask > 0)
        
        # Safety check: empty mask
        if xs.size == 0 or ys.size == 0:
            h, w = img.shape
            return img, mask, (0, 0, w, h)

        xmin = int(max(xs.min() - padding, 0))
        xmax = int(min(xs.max() + padding, img.shape[1]))
        ymin = int(max(ys.min() - padding, 0))
        ymax = int(min(ys.max() + padding, img.shape[0]))
    
        img_cropped = img[ymin:ymax, xmin:xmax]
        mask_cropped = mask[ymin:ymax, xmin:xmax]
    
        return img_cropped, mask_cropped, (xmin, ymin, xmax, ymax)

    def _adjust_boxes_after_crop(self, boxes, crop_coords):
        """
        Adjust bounding boxes after cropping.
        """
        xmin, ymin, _, _ = crop_coords
        boxes[:, [0, 2]] -= xmin
        boxes[:, [1, 3]] -= ymin
        return boxes

    def __getitem__(self, idx):
        img_path, label_path, mask_path, _ = self.samples[idx]

        # Load image and mask
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        h_orig, w_orig = img.shape

        # Load YOLO labels (normalized → absolute)
        boxes = []
        labels = []

        with open(label_path, "r") as f:
            for line in f:
                class_id, x_c, y_c, w, h = map(float, line.split())

                x_c *= w_orig
                y_c *= h_orig
                w *= w_orig
                h *= h_orig

                xmin = x_c - w / 2
                ymin = y_c - h / 2
                xmax = x_c + w / 2
                ymax = y_c + h / 2

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(class_id))

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # Crop by brain mask
        img, mask, crop_coords = self._crop_by_mask(img, mask)
        boxes = self._adjust_boxes_after_crop(boxes, crop_coords)

        # Clip boxes to image boundaries
        h_crop, w_crop = img.shape
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w_crop)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h_crop)

        # Resize image
        """
        img = cv2.resize(img, self.image_size)
        scale_x = self.image_size[1] / w_crop
        scale_y = self.image_size[0] / h_crop

        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        """
        
        # Data augmentation
        if self.augment and random.random() < 0.5:
            img = cv2.flip(img, 1)
            boxes[:, [0, 2]] = self.image_size[1] - boxes[:, [2, 0]]

        # Convert to tensor
        img_tensor = TF.to_tensor(img)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        return img_tensor, target

    def visualize_sample(self, idx=None):
            """
            Visualize one sample after full preprocessing pipeline.
            Shows image and bounding boxes.
            """
            if idx is None:
                idx = random.randint(0, len(self.samples) - 1)
    
            img_tensor, target = self[idx]
    
            img = img_tensor.squeeze().numpy()
            boxes = target["boxes"]
            labels = target["labels"]
    
            fig, ax = plt.subplots(1, figsize=(6, 6))
            ax.imshow(img, cmap="gray")
    
            for box, label in zip(boxes, labels):
                xmin, ymin, xmax, ymax = box.tolist()
                rect = patches.Rectangle(
                    (xmin, ymin),
                    xmax - xmin,
                    ymax - ymin,
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none"
                )
                ax.add_patch(rect)
    
                ax.text(
                    xmin,
                    ymin - 5,
                    f"Class {label.item()}",
                    color="yellow",
                    fontsize=10,
                    backgroundcolor="black"
                )
    
            ax.set_title(f"Dataset sample index: {idx}")
            ax.axis("off")
            plt.show()