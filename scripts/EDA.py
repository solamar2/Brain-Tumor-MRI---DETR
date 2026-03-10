import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

class BrainTumorEDA:
    def __init__(self, data_dir):
        """
        data_dir: path to Train directory
        Labels are YOLO-style: <class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
        """
        self.data_dir = data_dir
        self.classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        self.images = {}  # dict[class] = list of image paths
        self.labels = {}  # dict[class] = list of label paths
        self.image_sizes = {}  # dict[img_path] = (h, w)
        self._load_paths()
        self._cache_image_sizes()

    def _load_paths(self):
        for cls in self.classes:
            img_folder = os.path.join(self.data_dir, cls, "Images")
            lbl_folder = os.path.join(self.data_dir, cls, "labels")
            self.images[cls] = glob.glob(os.path.join(img_folder, "*.jpg")) 
            self.labels[cls] = glob.glob(os.path.join(lbl_folder, "*.txt"))

    def _cache_image_sizes(self):
        for cls in self.classes:
            for img_path in self.images[cls]:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self.image_sizes[img_path] = img.shape

    # ---------------------------
    # Label Parsing
    # ---------------------------
    def parse_label(self, lbl_path):
        """Return list of bounding boxes in pixels: [xmin,ymin,xmax,ymax]"""
        img_path = lbl_path.replace(os.path.join("labels",""), "Images/").replace(".txt",".jpg")
        if not os.path.exists(img_path):
            img_path = lbl_path.replace(os.path.join("labels",""), "Images/").replace(".txt",".png")
        if not os.path.exists(img_path):
            return [], None
        h, w = self.image_sizes.get(img_path, cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).shape)
        boxes = []
        with open(lbl_path, "r") as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) != 5:
                    continue
                _, x_c, y_c, bw, bh = parts
                xmin = int((x_c - bw/2) * w)
                ymin = int((y_c - bh/2) * h)
                xmax = int((x_c + bw/2) * w)
                ymax = int((y_c + bh/2) * h)
                boxes.append([xmin, ymin, xmax, ymax])
        return boxes, (h, w)

    # ---------------------------
    # Class Counts & Example Images
    # ---------------------------
    def plot_class_counts_and_examples(self):
        counts = [len(self.images[cls]) for cls in self.classes]
        plt.figure(figsize=(8,5))
        plt.bar(self.classes, counts, color='skyblue')
        plt.ylabel("Number of Images")
        plt.title("Number of Images per Class")
        plt.show()

        plt.figure(figsize=(10,8))
        n_cols, n_rows = 2, 2
        for i, cls in enumerate(self.classes):
            if not self.images[cls]:
                continue
            img_path = self.images[cls][0]
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            lbl_path = self.labels[cls][0] if self.labels[cls] else ""
            boxes, _ = self.parse_label(lbl_path) if lbl_path else ([], None)

            plt.subplot(n_rows, n_cols, i+1)
            plt.imshow(img, cmap='gray')
            for box in boxes:
                xmin, ymin, xmax, ymax = box
                plt.gca().add_patch(plt.Rectangle((xmin, ymin),
                                                  xmax-xmin, ymax-ymin,
                                                  edgecolor='red', facecolor='none', linewidth=2))
            plt.title(cls)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    # ---------------------------
    # Bounding Box Analysis
    # ---------------------------
    def plot_bb_analysis(self, bins=50):
        # Area & Aspect Ratio
        plt.figure(figsize=(8,5))
        for cls in self.classes:
            areas, ratios = [], []
            for lbl_path in self.labels[cls]:
                boxes, size = self.parse_label(lbl_path)
                if not boxes:
                    continue
                h, w = size
                for xmin, ymin, xmax, ymax in boxes:
                    areas.append((xmax - xmin)*(ymax - ymin))
                    if (ymax - ymin) > 0:
                        ratios.append((xmax - xmin)/(ymax - ymin))
            plt.hist(areas, bins=bins, alpha=0.5, label=cls)
        plt.xlabel("Bounding Box Area (pixels^2)")
        plt.ylabel("Count")
        plt.title("Bounding Box Areas per Class")
        plt.legend()
        plt.show()

        plt.figure(figsize=(8,5))
        for cls in self.classes:
            _, ratios = [], []
            for lbl_path in self.labels[cls]:
                boxes, size = self.parse_label(lbl_path)
                if not boxes:
                    continue
                for xmin, ymin, xmax, ymax in boxes:
                    if (ymax - ymin) > 0:
                        ratios.append((xmax - xmin)/(ymax - ymin))
            plt.hist(ratios, bins=bins, alpha=0.5, label=cls)
        plt.xlabel("Aspect Ratio (Width / Height)")
        plt.ylabel("Count")
        plt.title("Bounding Box Aspect Ratios per Class")
        plt.legend()
        plt.show()

        # BB Centers Heatmap
        plt.figure(figsize=(12,5))
        for idx, cls in enumerate(self.classes):
            centers_x, centers_y = [], []
            for lbl_path in self.labels[cls]:
                boxes, _ = self.parse_label(lbl_path)
                for xmin, ymin, xmax, ymax in boxes:
                    centers_x.append((xmin+xmax)/2)
                    centers_y.append((ymin+ymax)/2)
            plt.subplot(1, len(self.classes), idx+1)
            plt.hist2d(centers_x, centers_y, bins=bins, cmap='hot')
            plt.xlabel("Center X")
            plt.ylabel("Center Y")
            plt.title(f"{cls} BB Centers")
            plt.colorbar(label='Count')
        plt.tight_layout()
        plt.show()

    # ---------------------------
    # Image Stats
    # ---------------------------
    def plot_image_stats(self, bins=256):
        hist_per_class = {cls: np.zeros(bins, dtype=np.int64) for cls in self.classes}
        stds_per_class = {cls: [] for cls in self.classes}
        widths_per_class, heights_per_class = {cls: [] for cls in self.classes}, {cls: [] for cls in self.classes}

        for cls in self.classes:
            for img_path in self.images[cls]:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                h, w = img.shape
                widths_per_class[cls].append(w)
                heights_per_class[cls].append(h)
                stds_per_class[cls].append(np.std(img))
                hist_per_class[cls] += np.histogram(img, bins=bins, range=(0,256))[0]


        # Std Boxplot
        plt.figure(figsize=(8,5))
        data = [stds_per_class[cls] for cls in self.classes]
        plt.boxplot(data, labels=self.classes)
        plt.ylabel("Pixel Intensity Std")
        plt.title("Image Std per Class")
        plt.show()      

        # Width vs Height Scatter
        plt.figure(figsize=(7,7))
        for cls in self.classes:
            plt.scatter(widths_per_class[cls], heights_per_class[cls], alpha=0.5, label=cls)
        plt.xlabel("Width")
        plt.ylabel("Height")
        plt.title("Width vs Height per Class")
        plt.legend()
        plt.show()