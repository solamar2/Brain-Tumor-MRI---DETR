import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.patches as patches
import random

class BrainTumorEvaluator:
    """
    Evaluation and visualization for DETR-based brain tumor detection.
    Supports:
    - Per-class metrics (Precision, Recall, F1, IoU)
    - mAP@0.5
    - Precision-Recall curves
    - Visualizing GT and predicted bounding boxes with IoU
    """

    def __init__(self, model_trainer, dataset, class_names, iou_thresh=0.5):
        """
        model_trainer: DETRTrainer instance
        dataset: BrainTumorDataset instance
        class_names: list of class names
        iou_thresh: IoU threshold for TP/FP (default 0.5)
        """
        self.model = model_trainer
        self.dataset = dataset
        self.class_names = class_names
        self.iou_thresh = iou_thresh

    def compute_iou(boxA, boxB):
        """Compute IoU between two boxes [x_min, y_min, x_max, y_max]"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def evaluate_dataset(self, conf_thresh=0.0):
        """
        Evaluate the entire dataset.
        Returns:
        - metrics_per_class: dict
        - all_iou_values: dict for histogram
        - all_predictions: list of predictions (for mAP)
        """
        # Store all GT and predictions per class
        all_gt = defaultdict(list)       # {class_idx: list of boxes per image}
        all_pred = defaultdict(list)     # {class_idx: list of (box, score) per image}

        metrics_per_class = {}
        all_iou_values = defaultdict(list)

        for idx in range(len(self.dataset)):
            img_tensor, target = self.dataset[idx]
            gt_boxes = target["boxes"]
            gt_labels = target["labels"]

            # Prediction
            img_input = img_tensor.unsqueeze(0).to(self.model.device)
            preds = self.model.predict(img_input, conf_thresh=conf_thresh)
            pred_boxes = preds[0]["boxes"]
            pred_labels = preds[0]["labels"]
            pred_scores = preds[0]["scores"]

            # Organize per class
            for cls_idx in range(len(self.class_names)):
                # GT
                cls_gt_boxes = [b.cpu().numpy() for b, l in zip(gt_boxes, gt_labels) if l.item() == cls_idx]
                all_gt[cls_idx].extend(cls_gt_boxes)

                # Predictions
                cls_pred_boxes_scores = [(b.cpu().numpy(), s.item()) for b, l, s in zip(pred_boxes, pred_labels, pred_scores) if l.item() == cls_idx]
                all_pred[cls_idx].extend(cls_pred_boxes_scores)

                # IoU list for plotting
                for pb, s in cls_pred_boxes_scores:
                    if cls_gt_boxes:
                        ious = [self.compute_iou(pb, gt) for gt in cls_gt_boxes]
                        all_iou_values[self.class_names[cls_idx]].append(max(ious))
                    else:
                        all_iou_values[self.class_names[cls_idx]].append(0.0)

        # Compute mAP@0.5 per class
        for cls_idx, cls_name in enumerate(self.class_names):
            # Sort predictions by score descending
            preds_sorted = sorted(all_pred[cls_idx], key=lambda x: x[1], reverse=True)
            gt_boxes_cls = all_gt[cls_idx].copy()
            TP, FP = [], []

            matched_gt = set()
            for pb, score in preds_sorted:
                ious = [self.compute_iou(pb, gt) for gt_idx, gt in enumerate(gt_boxes_cls) if gt_idx not in matched_gt]
                if ious and max(ious) >= self.iou_thresh:
                    TP.append(1)
                    FP.append(0)
                    matched_gt.add(np.argmax(ious))
                else:
                    TP.append(0)
                    FP.append(1)

            TP_cum = np.cumsum(TP)
            FP_cum = np.cumsum(FP)
            recalls = TP_cum / (len(gt_boxes_cls) + 1e-6)
            precisions = TP_cum / (TP_cum + FP_cum + 1e-6)

            # AP as area under Precision-Recall curve
            AP = 0.0
            if len(recalls) > 0:
                recalls = np.concatenate(([0.0], recalls, [1.0]))
                precisions = np.concatenate(([1.0], precisions, [0.0]))
                for i in range(len(recalls)-1):
                    AP += (recalls[i+1]-recalls[i]) * precisions[i+1]

            metrics_per_class[cls_name] = {
                "AP@0.5": AP,
                "n_GT": len(gt_boxes_cls),
                "n_pred": len(preds_sorted)
            }

        return metrics_per_class, all_iou_values, all_gt, all_pred

    def plot_iou_histograms(self, all_iou_values):
        """Plot histogram of IoU per class"""
        plt.figure(figsize=(8,4))
        for cls, ious in all_iou_values.items():
            plt.hist(ious, bins=10, alpha=0.5, label=cls)
        plt.xlabel("IoU")
        plt.ylabel("Number of boxes")
        plt.title("IoU Histogram per Class")
        plt.legend()
        plt.show()

    def plot_precision_recall_curves(self, all_gt, all_pred):
        """Plot Precision-Recall curve per class"""
        for cls_idx, cls_name in enumerate(self.class_names):
            # Sort predictions
            preds_sorted = sorted(all_pred[cls_idx], key=lambda x: x[1], reverse=True)
            gt_boxes_cls = all_gt[cls_idx].copy()
            TP, FP = [], []

            matched_gt = set()
            for pb, score in preds_sorted:
                ious = [self.compute_iou(pb, gt) for gt_idx, gt in enumerate(gt_boxes_cls) if gt_idx not in matched_gt]
                if ious and max(ious) >= self.iou_thresh:
                    TP.append(1)
                    FP.append(0)
                    matched_gt.add(np.argmax(ious))
                else:
                    TP.append(0)
                    FP.append(1)

            TP_cum = np.cumsum(TP)
            FP_cum = np.cumsum(FP)
            recalls = TP_cum / (len(gt_boxes_cls) + 1e-6)
            precisions = TP_cum / (TP_cum + FP_cum + 1e-6)

            recalls = np.concatenate(([0.0], recalls, [1.0]))
            precisions = np.concatenate(([1.0], precisions, [0.0]))

            plt.figure(figsize=(6,4))
            plt.plot(recalls, precisions, marker='o')
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall Curve: {cls_name}")
            plt.grid(True)
            plt.show()

    def visualize_sample(self, idx=None, conf_thresh=0.5, show_gt=True, show_pred=True):
        """
        Visualize one image with GT and predicted BBs including IoU with closest GT
        """
        if idx is None:
            idx = random.randint(0, len(self.dataset) - 1)

        img_tensor, target = self.dataset[idx]
        img = img_tensor.squeeze().numpy()
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]

        # Predict
        img_input = img_tensor.unsqueeze(0).to(self.model.device)
        preds = self.model.predict(img_input, conf_thresh=conf_thresh)
        pred_boxes = preds[0]["boxes"]
        pred_labels = preds[0]["labels"]
        pred_scores = preds[0]["scores"]

        fig, ax = plt.subplots(1, figsize=(6,6))
        ax.imshow(img, cmap="gray")

        # Ground truth
        if show_gt:
            for box, label in zip(gt_boxes, gt_labels):
                xmin, ymin, xmax, ymax = box.tolist()
                rect = patches.Rectangle((xmin,ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor="red", facecolor="none")
                ax.add_patch(rect)
                ax.text(xmin, ymin-5, f"GT Class {label.item()}", color="yellow", fontsize=10, backgroundcolor="black")

        # Predictions
        if show_pred and len(pred_boxes) > 0:
            for pb, pl, ps in zip(pred_boxes, pred_labels, pred_scores):
                # Compute IoU with closest GT of same class
                gt_same_class = [b for b, l in zip(gt_boxes, gt_labels) if l.item() == pl.item()]
                iou_val = max([self.compute_iou(pb.cpu().numpy(), gt.cpu().numpy()) for gt in gt_same_class] + [0.0])
                xmin, ymin, xmax, ymax = pb.tolist()
                rect = patches.Rectangle((xmin,ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor="blue", facecolor="none")
                ax.add_patch(rect)
                ax.text(xmin, ymax+5, f"Pred {pl.item()} ({ps:.2f}, IoU={iou_val:.2f})", color="cyan", fontsize=10, backgroundcolor="black")

        ax.set_title(f"Dataset sample index: {idx}")
        ax.axis("off")
        plt.show()