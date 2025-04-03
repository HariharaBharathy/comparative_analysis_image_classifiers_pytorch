import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from mnist_similarity_utils import MNISTSimilarityPairs, get_or_create_splits, visualize_similarity_pairs
from image_similarity_VGG import SiameseNetwork_MNIST
import torchvision.transforms as transforms
import os
from torchvision.datasets import MNIST
from torch.utils.data import ConcatDataset, DataLoader


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook to store gradients during backprop
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]  # Gradients from target layer

        # Hook to store feature map activations
        def forward_hook(module, input, output):
            self.activations = output  # Feature maps

        # Register the hooks
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, img, device="cuda"):
        """ Generate Grad-CAM heatmap for a single image. """
        self.model.eval()
        img = img.to(device).unsqueeze(0)  # Add batch dim [1, 1, H, W]

        # Forward pass
        output = self.model.cnn_base(img)  # Pass only through the encoder
        output = output.squeeze()  # Ensure it's a scalar
        target_score = output.mean()
        target_score.backward()  # Compute gradients

        # Compute Grad-CAM
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])  # Global Average Pooling
        activations = self.activations.squeeze(0).detach()  # [C, H, W]

        # Weight feature maps by gradients
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        # Average over channels and normalize
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)  # ReLU (remove negatives)
        heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1  # Normalize to [0,1]

        return heatmap



def apply_gradcam(model, dataset, target_layer, num_samples=5, device="cuda"):
    """ Apply Grad-CAM to both images in a pair and visualize them. """
    gradcam = GradCAM(model, target_layer)

    fig, axes = plt.subplots(num_samples, 4, figsize=(12, 3 * num_samples))

    for i in range(num_samples):
        img1, img2, label = dataset[i]

        # Generate heatmaps for both images in the pair
        heatmap1 = gradcam.generate_cam(img1, device)
        heatmap2 = gradcam.generate_cam(img2, device)

        # Plot original images
        axes[i, 0].imshow(img1.squeeze(0), cmap="gray")
        axes[i, 0].set_title(f"Image 1")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(img2.squeeze(0), cmap="gray")
        axes[i, 1].set_title(f"Image 2")
        axes[i, 1].axis("off")

        # Overlay Grad-CAM heatmap on Image 1
        axes[i, 2].imshow(img1.squeeze(0), cmap="gray")
        axes[i, 2].imshow(heatmap1, cmap="jet", alpha=0.5)  # Blend heatmap
        axes[i, 2].set_title(f"Grad-CAM 1")
        axes[i, 2].axis("off")

        # Overlay Grad-CAM heatmap on Image 2
        axes[i, 3].imshow(img2.squeeze(0), cmap="gray")
        axes[i, 3].imshow(heatmap2, cmap="jet", alpha=0.5)  # Blend heatmap
        axes[i, 3].set_title(f"Grad-CAM 2")
        axes[i, 3].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    path_to_trained_models = "/home/mjgtdj/workspace/trained_nn_models/Pretrained_with_ImageNet_weights/MNIST_similarity/"
    base_dir_for_MNIST = "/home/mjgtdj/workspace/image_classification_dataset/MNIST_similarity/"
    experiment_name = "MNIST_similarity_vgg16_without_early_stopping_lightweight_with_bn_dropout"

    this_model = torch.load(os.path.join(path_to_trained_models, experiment_name, experiment_name + '.pt'),
                            weights_only=False)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

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

    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)

    # Example usage
    apply_gradcam(this_model, test_loader, target_layer=this_model.cnn_base[0][42], num_samples=5, device="cuda")


