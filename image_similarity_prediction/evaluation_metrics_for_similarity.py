import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import MNIST
from mnist_similarity_utils import MNISTSimilarityPairs, get_or_create_splits, visualize_similarity_pairs
import seaborn as sns
from image_similarity_VGG import SiameseNetwork_MNIST
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader
from compute_gradCAM import GradCAM
import os
import pandas as pd
import copy

def evaluate_model(model, dataloader, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []
    all_images_1 = []
    all_images_2 = []

    with torch.no_grad():
        for batch_idx, (images_1, images_2, targets) in enumerate(dataloader):
            for i in range(images_1.size(0)):

                ## save unnormalized image for later Grad-CAM
                orig_image_1 = copy.deepcopy(images_1[i])
                orig_image_1 = transforms.ToPILImage()(orig_image_1)

                orig_image_2 = copy.deepcopy(images_2[i])
                orig_image_2 = transforms.ToPILImage()(orig_image_2)

                input_tensor_1 = images_1[i].unsqueeze(0).to(device)
                input_tensor_2 = images_2[i].unsqueeze(0).to(device)

                output = model(input_tensor_1, input_tensor_2)
                pred_class = torch.where(torch.sigmoid(output) > 0.5, 1, 0)

                all_preds.append(pred_class.cpu().numpy())
                all_labels.append(targets[i].numpy())
                all_images_1.append(orig_image_1)
                all_images_2.append(orig_image_2)

    return np.array(all_preds), np.array(all_labels), all_images_1, all_images_2

def print_classification_report(y_true, y_pred, class_names):
    this_classification_report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    print(this_classification_report)

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

    plt.figure(figsize=(12, 10))
    sns.heatmap(df_cm, cmap='Blues', annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()

def show_misclassified_images(all_images_1, all_images_2, y_true, y_pred, class_names, save_mispredictions=False, n_mispredictions = 10, visualize_mispredictions=True):
    misclassified_indices = np.where(y_true != y_pred)[0]
    print(f"Total misclassified: {len(misclassified_indices)}")

    if len(misclassified_indices) == 0:
        print("No misclassified samples found.")
        return

    this_transform = transforms.ToPILImage()

    if save_mispredictions:
        for misclassified_idx in misclassified_indices:

            ## create a folder to save the mispredictions
            path_to_mispredictions = os.path.join(path_to_trained_models, experiment_name, 'mispredictions')
            isExist = os.path.exists(path_to_mispredictions)

            if not isExist:
                os.makedirs(path_to_mispredictions)

            misclassified_image = this_transform(images[misclassified_idx])
            img_name = f"{y_true[misclassified_idx]}_{y_pred[misclassified_idx]}"
            misclassified_image.save(os.path.join(path_to_mispredictions, img_name+'.jpg'))


    if visualize_mispredictions:
        fig, ax = plt.subplots(n_mispredictions, 2, figsize=(10, 2 * n_mispredictions))
        for i, idx in enumerate(misclassified_indices[0: n_mispredictions]):
            img1 = all_images_1[idx]
            img2 = all_images_2[idx]

            predicted_class = class_names[int(y_pred[idx])]
            true_class = class_names[int(y_true[idx])]

            ## plot first image
            ax[i, 0].imshow(img1, cmap='gray')
            ax[i, 0].axis("off")
            ax[i, 0].set_title(f"True: {true_class}\nPred: {predicted_class}")

            ## plot second image
            ax[i, 1].imshow(img2, cmap='gray')
            ax[i, 1].axis("off")
        plt.tight_layout()
        plt.show()

    return "Saved and visualized mispredictions"



if __name__ == "__main__":

    path_to_trained_models = "/home/user/workspace/trained_nn_models/Pretrained_with_ImageNet_weights/MNIST_similarity/"
    base_dir_for_MNIST = "/home/user/workspace/image_classification_dataset/MNIST_similarity/"
    experiment_name = "MNIST_similarity_vgg16_without_early_stopping_lightweight_with_bn_dropout"

    this_model = torch.load(os.path.join(path_to_trained_models, experiment_name, experiment_name+'.pt'), weights_only=False)

    plot_gradcam_on_misclassified = True
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

    y_pred, y_true, images_1, images_2 = evaluate_model(this_model, test_loader)
    #plot_confusion_matrix(y_true, y_pred, ["SAME", "DIFFERENT"])
    #print_classification_report(y_true, y_pred, ["SAME", "DIFFERENT"])
    #show_misclassified_images(images_1, images_2, y_true, y_pred, ["SAME", "DIFFERENT"])

    ## plotting gradCAM on misclassified images
    if plot_gradcam_on_misclassified:
        misclassified_indices = np.where(y_true == y_pred)[0]

        print(f"Total misclassified: {len(misclassified_indices)}")

        if len(misclassified_indices) == 0:
            print("No misclassified samples found.")
        else:
            num_samples = 4
            fig, axes = plt.subplots(num_samples, 4, figsize=(12, 3 * num_samples))
            for i, idx in enumerate(misclassified_indices[0:num_samples]):
                """ Apply Grad-CAM to both images in a pair and visualize them. """
                gradcam = GradCAM(this_model, this_model.cnn_base[0][42])

                # Generate heatmaps for both images in the pair
                this_transform = transforms.Compose([transforms.ToTensor()])

                ## convert PIL image to tensor
                this_image_1 = this_transform(images_1[idx])
                this_image_2 = this_transform(images_2[idx])

                heatmap1 = gradcam.generate_cam(this_image_1, device="cuda")
                heatmap2 = gradcam.generate_cam(this_image_2, device="cuda")

                # Plot original images
                axes[i, 0].imshow(this_image_1.squeeze(0), cmap="gray")
                axes[i, 0].set_title(f"Image 1")
                axes[i, 0].axis("off")

                axes[i, 1].imshow(this_image_2.squeeze(0), cmap="gray")
                axes[i, 1].set_title(f"Image 2")
                axes[i, 1].axis("off")

                # Overlay Grad-CAM heatmap on Image 1
                axes[i, 2].imshow(this_image_1.squeeze(0), cmap="gray")
                axes[i, 2].imshow(heatmap1, cmap="jet", alpha=0.5)  # Blend heatmap
                axes[i, 2].set_title(f"Grad-CAM 1")
                axes[i, 2].axis("off")

                # Overlay Grad-CAM heatmap on Image 2
                axes[i, 3].imshow(this_image_2.squeeze(0), cmap="gray")
                axes[i, 3].imshow(heatmap2, cmap="jet", alpha=0.5)  # Blend heatmap
                axes[i, 3].set_title(f"Grad-CAM 2")
                axes[i, 3].axis("off")

            plt.tight_layout()
            plt.show()






