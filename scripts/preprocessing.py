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
    - augmentation
    - supports multiple bounding boxes per image
    - keeps only boxes matching the true class of the image
    """

    def __init__(self, train_dir, class_to_idx, mask_dir_name="brain_masks", image_size=(256, 256), augment=True):
        self.samples = []
        self.image_size = image_size
        self.augment = augment
        self.class_to_idx = class_to_idx 
    
        for cls_name in os.listdir(train_dir):
            cls_dir = os.path.join(train_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue

            true_class = self.class_to_idx[cls_name]

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
                    self.samples.append((img_path, label_path, mask_path, true_class))

    def __len__(self):
        return len(self.samples)

    def _crop_by_mask(self, img, mask, padding=0):
        ys, xs = np.where(mask > 0)
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
        xmin, ymin, _, _ = crop_coords
        boxes[:, [0, 2]] -= xmin
        boxes[:, [1, 3]] -= ymin
        return boxes

    def _augment_image_boxes(self, img, boxes):
        """
        Apply random augmentations to image and boxes.
        """
        h, w = img.shape

        # Horizontal flip
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

        # Zoom / scale
        if random.random() < 0.4:
            scale = random.uniform(0.9, 1.1)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            for i in range(len(boxes)):
                box = boxes[i]
                corners = torch.tensor([
                    [box[0], box[1]],
                    [box[2], box[1]],
                    [box[2], box[3]],
                    [box[0], box[3]]
                ], dtype=torch.float32)
                ones = torch.ones((4,1))
                corners_h = torch.cat([corners, ones], dim=1)
                transformed = torch.from_numpy(M).float().mm(corners_h.T).T
                x_coords = transformed[:, 0]
                y_coords = transformed[:, 1]
                boxes[i, 0] = x_coords.min()
                boxes[i, 1] = y_coords.min()
                boxes[i, 2] = x_coords.max()
                boxes[i, 3] = y_coords.max()
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)

        # Brightness / contrast
        if random.random() < 0.3:
            alpha = random.uniform(0.9, 1.1)
            beta = random.uniform(-15, 15)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # Gaussian noise
        if random.random() < 0.2:
            noise = np.random.normal(0, 5, img.shape)
            img = img.astype(np.float32) + noise
            img = np.clip(img, 0, 255).astype(np.uint8)

        # Gaussian blur
        if random.random() < 0.2:
            img = cv2.GaussianBlur(img, (3, 3), 0)

        return img, boxes

    def __getitem__(self, idx):
        img_path, label_path, mask_path, true_class = self.samples[idx]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        h_orig, w_orig = img.shape

        # Load YOLO labels and filter by true class
        boxes = []
        labels = []
        with open(label_path, "r") as f:
            for line in f:
                class_id, x_c, y_c, w, h = map(float, line.split())
                if int(class_id) != true_class:
                    continue
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

        if len(boxes) == 0:
            boxes = torch.tensor([[0, 0, w_orig, h_orig]], dtype=torch.float32)
            labels = torch.tensor([true_class], dtype=torch.int64)
        else:
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
        img = cv2.resize(img, self.image_size)
        scale_x = self.image_size[1] / w_crop
        scale_y = self.image_size[0] / h_crop
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        # Data augmentation
        if self.augment:
            img, boxes = self._augment_image_boxes(img, boxes)

        # Convert to tensor
        img_tensor = TF.to_tensor(img)
        target = {"boxes": boxes, "labels": labels}
        return img_tensor, target

    def visualize_sample(self, idx=None):
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

