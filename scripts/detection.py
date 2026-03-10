import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
from torchvision.models.detection import detr_resnet50
import matplotlib.pyplot as plt

class DETRTrainer:
    def __init__(self, train_dir, device="cuda", image_size=(256,256), batch_size=4, lr=1e-4):
        self.device = torch.device(device)
        self.dataset = BrainTumorDataset(train_dir, image_size=image_size)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
        # DETR model (pretrained COCO, fine-tune for brain tumors)
        self.model = detr_resnet50(pretrained=True, num_classes=5).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = None  # handled internally by DETR

    def collate_fn(self, batch):
        images, targets = list(zip(*batch))
        return list(images), list(targets)

    def train_epoch(self, num_epochs=1):
        self.model.train()
        for epoch in range(num_epochs):
            for imgs, targets in self.dataloader:
                imgs = [img.to(self.device) for img in imgs]
                targets = [{k: v.to(self.device) for k,v in t.items()} for t in targets]

                loss_dict = self.model(imgs, targets)
                losses = sum(loss for loss in loss_dict.values())

                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {losses.item():.4f}")

    def predict(self, img_path, mask_path):
        self.model.eval()
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        img_masked = cv2.bitwise_and(img, mask)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256,256))])
        img_tensor = transform(img_masked).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)[0]  # DETR returns list of dicts
        return outputs

    def visualize_prediction(self, img_path, mask_path, threshold=0.5):
        outputs = self.predict(img_path, mask_path)
        img = cv2.imread(img_path)
        plt.figure(figsize=(6,6))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        boxes = outputs["boxes"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()
        for box, score, label in zip(boxes, scores, labels):
            if score < threshold:
                continue
            xmin, ymin, xmax, ymax = box
            plt.gca().add_patch(plt.Rectangle((xmin,ymin), xmax-xmin, ymax-ymin,
                                              edgecolor='red', facecolor='none', linewidth=2))
            plt.text(xmin, ymin, f"{label}:{score:.2f}", color='yellow', fontsize=8)
        plt.axis('off')
        plt.show()