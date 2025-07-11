import torch
import os
from image_classification_VGG import ClassifierNetwork
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
import copy
from scipy.io import loadmat
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask

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

def show_misclassified_images(images, y_true, y_pred, class_names, max_display=16, save_mispredictions=True, visualize_mispredictions=False):
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
            img = this_transform(images[idx])
            plt.imshow(img)
            plt.title(f"True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    return "Saved and visualized mispredictions"


def forward_pass_generate_CAM(model, input_img, predicted_class, true_class,  device='cuda'):

    # Initialize GradCAM from the last convolution layer
    model.eval()

    ## prepare normalized input
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(input_img).unsqueeze(0).to(device)

    cam_extractor = GradCAM(model, target_layer='cnn_base.features.42')  # Last conv layer in VGG16

    # Forward pass
    out = model(input_tensor)
    pred_class = out.argmax().item()

    # Extract CAM
    activation_map = cam_extractor(pred_class, out)[0]

    ## convert output tensor into a PIL image
    this_transform = transforms.ToPILImage()
    this_activation_map_img = this_transform(activation_map)

    # Overlay CAM on image
    result = overlay_mask(input_img, this_activation_map_img, alpha=0.5)

    ## create a folder to save the mispredictions
    path_to_mispredictions = os.path.join(path_to_trained_models, experiment_name, 'mispredictions')
    isExist = os.path.exists(path_to_mispredictions)

    if not isExist:
        os.makedirs(path_to_mispredictions)

    img_name = f"{true_class}_{predicted_class}"
    result.save(os.path.join(path_to_mispredictions, img_name + '.jpg'))

    # Show it
    plt.imshow(result)
    plt.axis('off')
    plt.title(f'Grad-CAM for class {predicted_class}')
    plt.show()

    return 0


if __name__ == "__main__":
    path_to_trained_models = "/home/user/workspace/trained_nn_models/Pretrained_with_ImageNet_weights/Stanford_cars_classification/"
    experiment_name = "Stanford_cars_vgg16_without_early_stopping_lightweight_with_bn_dropout"
    path_to_datasets = "/home/user/workspace/image_classification_dataset/"


    this_model = torch.load(os.path.join(path_to_trained_models, experiment_name, experiment_name + '.pt'), weights_only=False)

    # Define transformations
    just_resize = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_dataset = datasets.StanfordCars(root=path_to_datasets, split='test', transform=just_resize, download=False)

    # Create DataLoaders
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    ## loading class names from database folders in a python list object
    devkit_path = '/home/user/workspace/image_classification_dataset/stanford_cars/devkit'
    cars_meta = loadmat(os.path.join(devkit_path, 'cars_meta.mat'))
    labels_list = [str(c[0]) for c in cars_meta['class_names'][0]]

    y_pred, y_true, images = evaluate_model(this_model, test_loader)

    ## visualize gradcam for miscalssified image
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            ## function to forward pass and generate CAM
            forward_pass_generate_CAM(model=this_model,
                                      input_img=images[i],
                                      predicted_class= y_pred[i],
                                      true_class=y_true[i])





