import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Standard input size for ViTs
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_dir = "/home/user/workspace/image_classification_dataset/"

# Load train and test sets
train_dataset = datasets.StanfordCars(root=data_dir, split='train', transform=transform, download=False)
test_dataset = datasets.StanfordCars(root=data_dir, split='test', transform=transform, download=False)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
