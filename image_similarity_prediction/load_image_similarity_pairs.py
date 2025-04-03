from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import ConcatDataset
from mnist_similarity_utils import MNISTSimilarityPairs, get_or_create_splits, visualize_similarity_pairs
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

base_dir_for_MNIST = "/home/mjgtdj/workspace/image_classification_dataset/MNIST_similarity/"
mnist_train = MNIST(root=base_dir_for_MNIST, train=True, download=False)
mnist_test = MNIST(root=base_dir_for_MNIST, train=False, download=False)
mnist_full = ConcatDataset([mnist_train, mnist_test])

pair_dataset = MNISTSimilarityPairs(
    mnist_dataset=mnist_full,
    num_pairs=70000,
    transform=transform,
    seed=42,
    cache_path=os.path.join(base_dir_for_MNIST, "mnist_pairs.pt")
)

train_size = int(0.8 * len(pair_dataset))
val_size = int(0.1 * len(pair_dataset))
test_size = len(pair_dataset) - train_size - val_size

train_set, val_set, test_set = get_or_create_splits(
    pair_dataset,
    split_sizes=(train_size, val_size, test_size),
    split_path=os.path.join(base_dir_for_MNIST, "mnist_split_indices.pt"),
)

visualize_similarity_pairs(pair_dataset, num_pairs=10)

# Set your batch size
batch_size = 4

# Define DataLoaders for each subset
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

# Optionally, check the sizes
print(f"Train loader size: {len(train_loader)}")
print(f"Validation loader size: {len(val_loader)}")
print(f"Test loader size: {len(test_loader)}")



