import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from timm import create_model

# =========================
# Model
# =========================
class ViTSingleObject(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        # ViT backbone
        self.vit = create_model('vit_base_patch16_224', pretrained=pretrained)
        self.vit.head = nn.Identity()  # Drop original classifier head
        hidden_dim = self.vit.embed_dim
        
        # Classification head
        self.class_head = nn.Linear(hidden_dim, num_classes)
        
        # Bounding box head
        self.bbox_head = nn.Linear(hidden_dim, 4)  # [cx, cy, w, h] normalized

    def forward(self, x):
        feat = self.vit(x)  # [B, hidden_dim]
        cls_logits = self.class_head(feat)  # [B, num_classes]
        bbox_pred  = self.bbox_head(feat)   # [B, 4]
        return cls_logits, bbox_pred

# =========================
# Trainer
# =========================
class TrainerViT:
    def __init__(self, num_classes, device=None, learning_rate=1e-4, weight_decay=1e-4, num_epochs=10):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs

        # Model
        self.model = ViTSingleObject(num_classes).to(self.device)

        # Freeze all backbone parameters
        for param in self.model.vit.parameters():
            param.requires_grad = False

        # Only heads are trainable
        for param in self.model.class_head.parameters():
            param.requires_grad = True
        for param in self.model.bbox_head.parameters():
            param.requires_grad = True

        # Optimizer
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Loss functions
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.history = {"train_loss": []}

    def train(self, train_loader, print_every=1, bbox_weight=5.0):
        self.model.train()
        for epoch in range(self.num_epochs):
            torch.cuda.empty_cache()
            epoch_loss = 0.0

            for batch_idx, (imgs, targets) in enumerate(train_loader):
                imgs = [img.repeat(3, 1, 1).to(self.device) if img.shape[0]==1 else img.to(self.device)
                        for img in imgs]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                self.optimizer.zero_grad()

                # Prepare batch tensors
                imgs_tensor = torch.stack(imgs)  # [B,3,H,W]
                target_labels = torch.stack([t['labels'].squeeze() for t in targets]).to(self.device)
                target_boxes  = torch.stack([t['boxes'].squeeze() for t in targets]).to(self.device)

                # Forward
                cls_logits, bbox_pred = self.model(imgs_tensor)

                # Loss
                cls_loss  = self.cls_loss_fn(cls_logits, target_labels)
                bbox_loss = nn.functional.l1_loss(bbox_pred, target_boxes)
                loss = cls_loss + bbox_weight * bbox_loss

                # Backprop
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(train_loader)
            self.history["train_loss"].append(epoch_loss)

            if (epoch + 1) % print_every == 0 or epoch == self.num_epochs - 1:
                print(f"Epoch [{epoch+1}/{self.num_epochs}] Train Loss: {epoch_loss:.4f}")

        print("Training finished.")
        self.plot_loss()

    def predict(self, imgs, conf_thresh=0.5):
        self.model.eval()
        with torch.no_grad():
            if isinstance(imgs, torch.Tensor) and imgs.shape[1] == 1:
                imgs = imgs.repeat(1, 3, 1, 1)
            elif isinstance(imgs, list):
                imgs = [img.repeat(3, 1, 1) if img.shape[0]==1 else img for img in imgs]

            imgs_tensor = torch.stack(imgs) if isinstance(imgs, list) else imgs.to(self.device)
            cls_logits, bbox_pred = self.model(imgs_tensor)

            results = []
            for i in range(cls_logits.shape[0]):
                probs = cls_logits[i].softmax(dim=-1)
                score, label = probs.max(dim=-1)

                if score >= conf_thresh:
                    boxes_out  = bbox_pred[i].cpu().unsqueeze(0)
                    labels_out = label.cpu().unsqueeze(0)
                    scores_out = score.cpu().unsqueeze(0)
                else:
                    boxes_out  = torch.empty((0,4))
                    labels_out = torch.empty((0,), dtype=torch.long)
                    scores_out = torch.empty((0,))

                results.append({
                    "boxes": boxes_out,
                    "labels": labels_out,
                    "scores": scores_out
                })
        return results

    def plot_loss(self):
        plt.figure(figsize=(6,4))
        plt.plot(self.history["train_loss"], label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.show()

    @staticmethod
    def visualize_prediction(dataset, model_trainer, idx=None, conf_thresh=0.5, dataset_type="Train"):
        if idx is None:
            idx = random.randint(0, len(dataset) - 1)

        img_tensor, target = dataset[idx]
        img = img_tensor.squeeze().numpy()
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]

        img_input = img_tensor.unsqueeze(0).to(model_trainer.device)
        preds = model_trainer.predict([img_input.squeeze(0)], conf_thresh=conf_thresh)
        pred_boxes = preds[0]["boxes"]
        pred_labels = preds[0]["labels"]
        pred_scores = preds[0]["scores"]

        fig, ax = plt.subplots(1, figsize=(6,6))
        ax.imshow(img, cmap="gray")

        for box, label in zip(gt_boxes, gt_labels):
            xmin, ymin, xmax, ymax = box.tolist()
            rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor="red", facecolor="none")
            ax.add_patch(rect)
            ax.text(xmin, ymin-5, f"GT Class {label.item()}", color="yellow", fontsize=10, backgroundcolor="black")

        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            xmin, ymin, xmax, ymax = box.tolist()
            rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor="blue", facecolor="none")
            ax.add_patch(rect)
            ax.text(xmin, ymax+5, f"Pred {label.item()} ({score:.2f})", color="cyan", fontsize=10, backgroundcolor="black")

        ax.set_title(f"{dataset_type} Dataset sample index: {idx}")
        ax.axis("off")
        plt.show()