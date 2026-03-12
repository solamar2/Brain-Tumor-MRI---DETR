import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

class BrainTumorDataLoader:
    """
    Builds PyTorch DataLoaders for BrainTumorDataset.

    Features:
    - Weighted sampling for class imbalance
    - Custom collate function for detection models
    """

    def __init__(self, dataset, class_to_idx, batch_size=8, num_workers=0, weighted_sampling=False, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.weighted_sampling = weighted_sampling
        self.shuffle = shuffle

    def build_sampler(self):
        """
        Create WeightedRandomSampler.
        """
        class_ids = [sample[3] for sample in self.dataset.samples]

        class_counts = np.bincount(class_ids)
        class_weights = 1.0 / class_counts

        sample_weights = [class_weights[c] for c in class_ids]
        
        sampler = WeightedRandomSampler( weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        return sampler

    def collate_fn(self, batch):
        """
        Custom collate function for detection tasks.
        Allows variable number of bounding boxes per image.
        """
        images, targets = zip(*batch)
        return list(images), list(targets)

    def get_loader(self):
        """
        Create DataLoader instance.
        """
        sampler = self.build_sampler() if self.weighted_sampling else None
        
        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=(sampler is None and self.shuffle),
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn)         
        
        return loader
    
    
    
    