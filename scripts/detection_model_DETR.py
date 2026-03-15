import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

class DETRTrainer:
    def __init__(self, num_classes, device=None, learning_rate=1e-4, weight_decay=1e-4, num_epochs=10):
        """
        DETR trainer for 2D grayscale MRI images.
        Automatically adapts 1-channel MRI to 3-channel input.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs

        # Load pretrained DETR
        self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        
        # Change number of object queries to 1
        num_queries = 1
        hidden_dim = self.model.query_embed.weight.shape[1]
        self.model.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Replace classifier 
        in_features = self.model.class_embed.in_features
        self.model.class_embed = nn.Linear(in_features, num_classes)
        
        self.model.to(self.device)

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Enable training only for query_embed and class_embed
        for p in self.model.query_embed.parameters():
            p.requires_grad = True
        for p in self.model.class_embed.parameters():
            p.requires_grad = True
    
        # Optimizer: only trainable parameters
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )


        # Loss function (classification only; BB loss added in train loop)
        self.cls_loss_fn = nn.CrossEntropyLoss()

        # Training history
        self.history = {"train_loss": []}

    def train(self, train_loader, print_every=1):
        """
        Train DETR on the given DataLoader.
        """
        self.model.train()

        for epoch in range(self.num_epochs):
            torch.cuda.empty_cache()
            epoch_loss = 0.0

            for batch_idx, (imgs, targets) in enumerate(train_loader):
                # Convert grayscale to 3 channels
                imgs = [img.repeat(3, 1, 1).to(self.device) if img.shape[0]==1 else img.to(self.device)
                        for img in imgs]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                self.optimizer.zero_grad()
                outputs = self.model(imgs)

                pred_logits = outputs['pred_logits']  # [B,1,num_classes]
                pred_boxes  = outputs['pred_boxes']   # [B,1,4]

                # Prepare targets
                target_labels = torch.stack([t['labels'].squeeze() for t in targets]).to(self.device)  # [B]
                target_boxes  = torch.stack([t['boxes'].squeeze() for t in targets]).to(self.device)   # [B,4]

                # Classification loss
                cls_loss = self.cls_loss_fn(pred_logits.squeeze(1), target_labels)

                # Bounding box L1 loss
                bbox_loss = nn.functional.l1_loss(pred_boxes.squeeze(1), target_boxes)

                # Total loss (adjust weight of bbox_loss if needed)
                loss = cls_loss + 5.0 * bbox_loss

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
        """
        Predict bounding boxes and labels for a batch of images.
        imgs: torch.Tensor [B,C,H,W] or list of tensors
        Returns: list of dicts per image: boxes, labels, scores
        """
        self.model.eval()
        with torch.no_grad():
            # Convert grayscale to 3-channel
            if isinstance(imgs, torch.Tensor) and imgs.shape[1] == 1:
                imgs = imgs.repeat(1, 3, 1, 1)
            elif isinstance(imgs, list):
                imgs = [img.repeat(3, 1, 1) if img.shape[0]==1 else img for img in imgs]

            imgs = [img.to(self.device) for img in imgs] if isinstance(imgs, list) else imgs.to(self.device)
            outputs = self.model(imgs)

            results = []
            batch_size = len(imgs) if isinstance(imgs, list) else imgs.shape[0]

            for i in range(batch_size):
                logits = outputs['pred_logits'][i]  # [num_queries=1, num_classes]
                boxes  = outputs['pred_boxes'][i]   # [num_queries=1, 4]

                scores, labels = logits.softmax(-1).max(-1)
                mask = scores > conf_thresh

                boxes_out  = boxes[mask].cpu()
                labels_out = labels[mask].cpu()
                scores_out = scores[mask].cpu()

                results.append({
                    "boxes": boxes_out,
                    "labels": labels_out,
                    "scores": scores_out
                })

        return results

    def plot_loss(self):
        """
        Plot training loss curve.
        """
        plt.figure(figsize=(6,4))
        plt.plot(self.history["train_loss"], label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.show()
        
    def visualize_prediction(dataset, model_trainer, idx=None, conf_thresh=0.5, dataset_type="Train"):
        """
        Visualize one sample from dataset with ground truth and predicted bounding box.
        - dataset_type: "Train" or "Test", shown in the title
        - Ground truth BB: red
        - Predicted BB: blue
        """
        if idx is None:
            idx = random.randint(0, len(dataset) - 1)
    
        # Get image and ground truth
        img_tensor, target = dataset[idx]
        img = img_tensor.squeeze().numpy()
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]
    
        # Prepare image for prediction: add batch dimension
        img_input = img_tensor.unsqueeze(0).to(model_trainer.device)
    
        # Get prediction
        preds = model_trainer.predict(img_input, conf_thresh=conf_thresh)
        pred_boxes = preds[0]["boxes"]
        pred_labels = preds[0]["labels"]
        pred_scores = preds[0]["scores"]
    
        # Plot
        fig, ax = plt.subplots(1, figsize=(6,6))
        ax.imshow(img, cmap="gray")
    
        # Ground truth boxes
        for box, label in zip(gt_boxes, gt_labels):
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
                f"GT Class {label.item()}",
                color="yellow",
                fontsize=10,
                backgroundcolor="black"
            )
    
        # Predicted boxes
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            xmin, ymin, xmax, ymax = box.tolist()
            rect = patches.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                linewidth=2,
                edgecolor="blue",
                facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(
                xmin,
                ymax + 5,
                f"Pred {label.item()} ({score:.2f})",
                color="cyan",
                fontsize=10,
                backgroundcolor="black"
            )
    
        ax.set_title(f"{dataset_type} Dataset sample index: {idx}")
        ax.axis("off")
        plt.show()
    
    