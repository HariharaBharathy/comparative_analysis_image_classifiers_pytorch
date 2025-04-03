import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
import seaborn as sns
from image_classification_VGG import ClassifierNetwork
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from scipy.io import loadmat
import pandas as pd
import copy

def evaluate_model(model, dataloader, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []
    all_images = []

    with torch.no_grad():
        for this_mini_batch_images, this_mini_batch_labels in dataloader:
            for i in range(this_mini_batch_images.size(0)):

                ## save unnormalized image for later Grad-CAM
                orig_image = copy.deepcopy(this_mini_batch_images[i])
                orig_image = transforms.ToPILImage()(orig_image)

                ## prepare normalized input
                normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                input_tensor = normalize(this_mini_batch_images[i]).unsqueeze(0).to(device)

                output = model(input_tensor)
                pred_class = torch.argmax(output, dim=1)

                all_preds.append(pred_class[0].cpu().numpy())
                all_labels.append(this_mini_batch_labels[i].numpy())
                all_images.append(orig_image)

    return np.array(all_preds), np.array(all_labels), all_images

def print_classification_report(y_true, y_pred, class_names):
    this_classification_report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    print(this_classification_report)

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

    plt.figure(figsize=(12, 10))
    sns.heatmap(df_cm, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()

def show_misclassified_images(images, y_true, y_pred, class_names, max_display=16, save_mispredictions=False, visualize_mispredictions=True):
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
        plt.figure(figsize=(15, 10))
        for i, idx in enumerate(misclassified_indices[:max_display]):
            plt.subplot(4, 4, i+1)
            img = images[idx]
            plt.imshow(img)
            plt.title(f"True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    return "Saved and visualized mispredictions"



if __name__ == "__main__":

    path_to_trained_models = "/home/mjgtdj/workspace/trained_nn_models/Pretrained_with_ImageNet_weights/Stanford_cars_classification/"
    path_to_datasets = "/home/mjgtdj/workspace/image_classification_dataset/"
    experiment_name = "Stanford_cars_vgg16_without_early_stopping_lightweight_with_bn_dropout"

    this_model = torch.load(os.path.join(path_to_trained_models, experiment_name, experiment_name+'.pt'), weights_only=False)

    # Define transformations
    just_resize = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    test_dataset = datasets.StanfordCars(root=path_to_datasets, split='test', transform=just_resize, download=False)

    # Create DataLoaders
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    ## loading class names from database folders in a python list object
    devkit_path = '/home/mjgtdj/workspace/image_classification_dataset/stanford_cars/devkit'
    cars_meta = loadmat(os.path.join(devkit_path, 'cars_meta.mat'))
    labels_list = [str(c[0]) for c in cars_meta['class_names'][0]]

    y_pred, y_true, images = evaluate_model(this_model, test_loader)
    plot_confusion_matrix(y_true, y_pred, labels_list)
    print_classification_report(y_true, y_pred, labels_list)
    show_misclassified_images(images, y_true, y_pred, labels_list)

