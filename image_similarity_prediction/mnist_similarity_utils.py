import torch
from torch.utils.data import Dataset, Subset, random_split
from torchvision.datasets import MNIST
from collections import defaultdict
import random
import os
import matplotlib.pyplot as plt



class MNISTSimilarityPairs(Dataset):
    def __init__(self, mnist_dataset, num_pairs=60000, transform=None, seed=42, cache_path=None):
        self.mnist = mnist_dataset
        self.transform = transform
        self.num_pairs = num_pairs
        self.seed = seed
        self.cache_path = cache_path

        if self.cache_path and os.path.exists(self.cache_path):
            print(f"Loading cached pairs from {self.cache_path}")
            data = torch.load(self.cache_path)
            if data["seed"] != self.seed or data["num_pairs"] != self.num_pairs:
                raise ValueError("Cached dataset seed or num_pairs mismatch. Delete the cache or set the right seed.")
            self.pairs = data["pairs"]
            self.targets = data["targets"]
        else:
            self.label_to_indices = defaultdict(list)
            for idx, (_, label) in enumerate(self.mnist):
                self.label_to_indices[label].append(idx)

            self.pairs = []
            self.targets = []

            self._create_pairs()

            if self.cache_path:
                print(f"Saving generated pairs to {self.cache_path}")
                torch.save({
                    "pairs": self.pairs,
                    "targets": self.targets,
                    "seed": self.seed,
                    "num_pairs": self.num_pairs
                }, self.cache_path)

    def _create_pairs(self):
        random.seed(self.seed)
        labels = list(self.label_to_indices.keys())

        for _ in range(self.num_pairs):
            if random.random() < 0.5:
                label = random.choice(labels)
                idx1, idx2 = random.sample(self.label_to_indices[label], 2)
                target = 0
            else:
                label1, label2 = random.sample(labels, 2)
                idx1 = random.choice(self.label_to_indices[label1])
                idx2 = random.choice(self.label_to_indices[label2])
                target = 1

            self.pairs.append((idx1, idx2))
            self.targets.append(target)

    def __getitem__(self, index):
        idx1, idx2 = self.pairs[index]
        img1, _ = self.mnist[idx1]
        img2, _ = self.mnist[idx2]
        target = torch.tensor(self.targets[index], dtype=torch.float32)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, target

    def __len__(self):
        return len(self.pairs)


def get_or_create_splits(dataset, split_sizes, split_path="mnist_splits.pt", seeds=(42, 123)):
    if os.path.exists(split_path):
        print(f"Loading dataset splits from {split_path}")
        saved = torch.load(split_path)
        train_idx = saved["train"]
        val_idx = saved["val"]
        test_idx = saved["test"]
    else:
        print("Creating new dataset splits...")
        total_size = sum(split_sizes)
        assert len(dataset) >= total_size, "Split sizes exceed dataset length."

        train_size, val_size, test_size = split_sizes

        gen1 = torch.Generator().manual_seed(seeds[0])
        train_base, val_test_base = random_split(dataset, [train_size, val_size + test_size], generator=gen1)

        gen2 = torch.Generator().manual_seed(seeds[1])
        val_base, test_base = random_split(val_test_base, [val_size, test_size], generator=gen2)

        train_idx = train_base.indices
        val_idx = val_base.indices
        test_idx = test_base.indices

        torch.save({
            "train": train_idx,
            "val": val_idx,
            "test": test_idx
        }, split_path)
        print(f"Splits saved to {split_path}")

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    return train_set, val_set, test_set


def visualize_similarity_pairs(dataset, num_pairs=5):
    """
    Visualize a few similarity pairs from the dataset.
    Args:
        dataset: The MNISTSimilarityPairs dataset.
        num_pairs: Number of pairs to visualize.
    """
    fig, axes = plt.subplots(num_pairs, 2, figsize=(10, 2 * num_pairs))

    for i in range(num_pairs):
        img1, img2, target = dataset[i]

        ax1, ax2 = axes[i]

        # Plot the first image
        ax1.imshow(img1.permute(1, 2, 0).numpy(), cmap='gray')
        ax1.set_title(f"Image 1 - Label: {target.item()}")
        ax1.axis('off')

        # Plot the second image
        ax2.imshow(img2.permute(1, 2, 0).numpy(), cmap='gray')
        ax2.set_title(f"Image 2 - Label: {target.item()}")
        ax2.axis('off')

    plt.tight_layout()
    plt.show()

