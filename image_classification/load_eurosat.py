import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

data_dir = "/home/mjgtdj/workspace/image_classification_dataset/EuroSAT/"

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to ViT-compatible size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset using ImageFolder
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

## Extract targets (class labels) for stratified split
targets = np.array(dataset.targets)

# Define stratified split sizes (70% train, 15% val, 15% test)
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, temp_idx = next(splitter.split(np.zeros(len(targets)), targets))

# Further split temp set into validation (15%) and test (15%)
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_idx, test_idx = next(splitter.split(np.zeros(len(temp_idx)), targets[temp_idx]))

# Create subsets
train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)
test_dataset = Subset(dataset, test_idx)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Check dataset size
print(f"Training Data set: {len(train_dataset)}")
print(f"Testing Data set: {len(test_dataset)}")
print(f"Validation Data set: {len(val_dataset)}")

##checking some examples
#data_iter = iter(dataloader)
#images, labels = next(data_iter)
#print(images.shape, labels.shape)

