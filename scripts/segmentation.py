import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

class BrainMaskSimple:
    """
    Brain extraction from 2D MRI JPG images using edge detection.
    Canny + morphology + largest contour.
    """

    def __init__(self, train_dir):
        self.train_dir = train_dir

    def _predict_mask(self, img_gray):
        """
        Generate binary brain mask using edge detection.
        """
        # 1. Smooth image
        #img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

        # 2. Canny edge detection
        edges = cv2.Canny(img_gray, threshold1=30, threshold2=100)

        # 3. Morphological closing to connect edges
        kernel = np.ones((9, 9), np.uint8)
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # 4. Find contours
        contours, _ = cv2.findContours(
            edges_closed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # 5. Create mask for all contours 
        mask = np.zeros_like(img_gray)
        if contours:
            cv2.drawContours(mask, contours, -1, 255, thickness=-1)

        return mask

    def create_overlay(self, img_path, mask_path, alpha=0.3):
        """
        Visualize overlay of mask on original image.
        """
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img_norm = img_gray / 255.0
        mask_norm = mask / 255.0

        img_rgb = np.stack([img_norm] * 3, axis=-1)
        img_rgb[mask_norm > 0] = (
            img_rgb[mask_norm > 0] * (1 - alpha)
            + np.array([1, 0, 0]) * alpha
        )

        plt.figure(figsize=(5, 5))
        plt.imshow(img_rgb)
        plt.axis("off")
        plt.title(f"Overlay: {os.path.basename(img_path)}")
        plt.show()

    def run(self):
        """
        Generate brain masks for all images in train_dir.
        """
        for cls_name in os.listdir(self.train_dir):
            cls_dir = os.path.join(self.train_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue

            images_dir = os.path.join(cls_dir, "images")
            masks_dir = os.path.join(cls_dir, "brain_masks")
            os.makedirs(masks_dir, exist_ok=True)

            if not os.path.exists(images_dir):
                continue

            for img_file in os.listdir(images_dir):
                if not img_file.lower().endswith(".jpg"):
                    continue

                img_path = os.path.join(images_dir, img_file)
                mask_path = os.path.join(masks_dir, img_file)

                # Skip if mask already exists
                if os.path.exists(mask_path):
                    continue

                img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img_gray is None:
                    continue

                mask_bin = self._predict_mask(img_gray)
                cv2.imwrite(mask_path, mask_bin)

            print(f"Masks saved: {cls_name}")

