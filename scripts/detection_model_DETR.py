import torch
from torch import nn, optim
import matplotlib.pyplot as plt

class DETRTrainer:
    def __init__(self, num_classes, device=None, num_queries=5, learning_rate=1e-4, weight_decay=1e-4, num_epochs=20, BB_weight=1.0):
        """
        DETR trainer for 2D grayscale MRI images with multiple bounding boxes.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.BB_weight = BB_weight

        # Load pretrained DETR
        self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)

        # Replace query embedding
        hidden_dim = self.model.query_embed.weight.shape[1]
        self.model.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Replace classifier
        in_features = self.model.class_embed.in_features
        self.model.class_embed = nn.Linear(in_features, num_classes)

        self.model.to(self.device)

        # Freeze all except query_embed, class_embed and backbone layer4
        for param in self.model.parameters():
            param.requires_grad = False
        for p in self.model.query_embed.parameters():
            p.requires_grad = True
        for p in self.model.class_embed.parameters():
            p.requires_grad = True
        for name, param in self.model.backbone.named_parameters():
            if "layer4" in name:
                param.requires_grad = True

        # Optimizer: only trainable params
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Classification loss
        self.cls_loss_fn = nn.CrossEntropyLoss()
        # L1 for bounding boxes
        self.bbox_loss_fn = nn.L1Loss()

        self.history = {"train_loss": []}

    def train(self, train_loader, print_every=1):
        """
        Train DETR for multiple BB per image.
        """
        self.model.train()

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0

            for batch_idx, (imgs, targets) in enumerate(train_loader):
                # Convert grayscale → 3 channels
                imgs = [img.repeat(3,1,1).to(self.device) if img.shape[0]==1 else img.to(self.device) for img in imgs]

                # Move targets to device
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                self.optimizer.zero_grad()
                outputs = self.model(imgs)

                # DETR outputs: [B, num_queries, num_classes], [B, num_queries, 4]
                pred_logits = outputs['pred_logits']
                pred_boxes  = outputs['pred_boxes']

                total_loss = 0.0
                for i in range(len(targets)):
                    tgt_labels = targets[i]['labels']
                    tgt_boxes  = targets[i]['boxes']

                    # Number of BB in this image
                    n_targets = tgt_labels.shape[0]

                    # Pad target BB to num_queries
                    if n_targets < self.num_queries:
                        pad = self.num_queries - n_targets
                        tgt_labels = torch.cat([tgt_labels, torch.tensor([2]*pad, device=self.device)])  # class_id=2 → No Tumor
                        tgt_boxes  = torch.cat([tgt_boxes, torch.zeros((pad,4), device=self.device)])

                    # Take only first num_queries if more (rare)
                    tgt_labels = tgt_labels[:self.num_queries]
                    tgt_boxes  = tgt_boxes[:self.num_queries]

                    # Classification loss (all queries)
                    cls_loss = self.cls_loss_fn(pred_logits[i], tgt_labels)

                    # Bounding box loss only for true tumors (exclude class_id=No Tumor=2)
                    mask = tgt_labels != 2
                    if mask.sum() > 0:
                        bbox_loss = self.bbox_loss_fn(pred_boxes[i][mask], tgt_boxes[mask])
                    else:
                        bbox_loss = 0.0

                    loss = cls_loss + self.BB_weight * bbox_loss
                    total_loss += loss

                total_loss /= len(targets)
                total_loss.backward()
                self.optimizer.step()

                epoch_loss += total_loss.item()

            epoch_loss /= len(train_loader)
            self.history["train_loss"].append(epoch_loss)

            if (epoch+1) % print_every == 0 or epoch == self.num_epochs-1:
                print(f"Epoch [{epoch+1}/{self.num_epochs}] Train Loss: {epoch_loss:.4f}")

        print("Training finished.")
        self.plot_loss()

    def predict(self, imgs, conf_thresh=0.5):
        """
        Predict multiple bounding boxes per image.
        Returns list of dicts: boxes, labels, scores
        """
        self.model.eval()
        results = []

        with torch.no_grad():
            if isinstance(imgs, torch.Tensor) and imgs.shape[1]==1:
                imgs = imgs.repeat(1,3,1,1)
            elif isinstance(imgs, list):
                imgs = [img.repeat(3,1,1) if img.shape[0]==1 else img for img in imgs]

            imgs = [img.to(self.device) for img in imgs] if isinstance(imgs, list) else imgs.to(self.device)
            outputs = self.model(imgs)

            batch_size = len(imgs) if isinstance(imgs, list) else imgs.shape[0]
            for i in range(batch_size):
                logits = outputs['pred_logits'][i]   # [num_queries, num_classes]
                boxes  = outputs['pred_boxes'][i]    # [num_queries,4]

                scores, labels = logits.softmax(-1).max(-1)
                mask = scores > conf_thresh

                results.append({
                    "boxes": boxes[mask].cpu(),
                    "labels": labels[mask].cpu(),
                    "scores": scores[mask].cpu()
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