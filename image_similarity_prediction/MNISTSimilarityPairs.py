import random
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Dataset, Subset, random_split, DataLoader


class MNISTSimilarityPairs(Dataset):
    def __init__(self, mnist_dataset, seed=42):
        self.mnist = mnist_dataset
        self.label_to_indices = {label: [] for label in range(10)}
        for idx, (_, label) in enumerate(self.mnist):
            self.label_to_indices[label].append(idx)
        random.seed(seed)

        # Generate pairs
        self.pairs = []
        for idx, (_, label1) in enumerate(self.mnist):
            # Positive pair (label = 1)
            idx2 = random.choice(self.label_to_indices[label1])
            self.pairs.append(((idx, idx2), 1))

            # Negative pair (label = 0)
            different_label = random.choice([l for l in range(10) if l != label1])
            idx3 = random.choice(self.label_to_indices[different_label])
            self.pairs.append(((idx, idx3), 0))

    def __getitem__(self, index):
        (idx1, idx2), label = self.pairs[index]
        img1, _ = self.mnist[idx1]
        img2, _ = self.mnist[idx2]
        return (img1, img2), label

    def __len__(self):
        return len(self.pairs)



if __name__ == "__main__":
    base_dir_for_MNIST = "/home/user/data"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    full_train_dataset = MNIST(root=base_dir_for_MNIST, train=True, download=False, transform=transform)

    ## split into 80% train, 20% validation
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    ## for reproducability, set a fixed seed
    generator = torch.Generator().manual_seed(42)
    train_base, val_base = random_split(full_train_dataset, [train_size, val_size], generator=generator)

    # Create similarity datasets
    train_dataset = MNISTSimilarityPairs(train_base)
    val_dataset = MNISTSimilarityPairs(val_base)

    # Test set (from official MNIST test set)
    test_base = MNIST(root=base_dir_for_MNIST, train=False, download=False, transform=transform)
    test_dataset = MNISTSimilarityPairs(test_base)

    ## dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(train_loader)