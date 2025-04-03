import numpy as np
import torch
import torch.nn as nn
import argparse
import os
import json
import torchvision.models
from torchvision.datasets import MNIST
from mnist_similarity_utils import MNISTSimilarityPairs, get_or_create_splits, visualize_similarity_pairs
from sklearn.metrics import confusion_matrix
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader

class SiameseNetwork_MNIST(nn.Module):
    def __init__(self):
        super(SiameseNetwork_MNIST, self).__init__()

        self.cnn_base = torchvision.models.vgg16_bn(weights="VGG16_BN_Weights.IMAGENET1K_V1")
        self.cnn_base.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.cnn_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cnn_base = torch.nn.Sequential(*(list(self.cnn_base.children())[:-1]))

        self.fc_in_features = 512
        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        # initialize the weights
        self.cnn_base.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, this_input):
        output = self.cnn_base(this_input).squeeze()
        return output

    def forward(self, input1, input2):

        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        ## concatenate features from both images
        if len(output1.size()) == 1 :
            output = torch.cat((output1, output2))
        else:
            output = torch.cat((output1, output2), 1)

        # pass the concatenation to the linear layers
        output = self.fc(output).squeeze()

        return output

def train(args, model, device, train_loader, optimizer, epoch):
    train_loss, train_correct, total_size = 0.0, 0, 0
    model.train()

    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (images_1, images_2, targets) in enumerate(train_loader):
        images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(images_1, images_2)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        predictions = torch.where(torch.sigmoid(output) > 0.5, 1, 0)
        train_correct += predictions.eq(targets.view_as(predictions)).sum().item()
        total_size += targets.shape[0]
        print("Epoch {}, Training Mini-batch : {}/{}, loss value : {}, Acc. values : {} ".format(epoch + 1, batch_idx, len(train_loader), loss.item(), (predictions == targets).sum().item()))

    return train_loss, train_correct, total_size

def test(model, device, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0
    total_test_size = 0

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch_idx, (images_1, images_2, targets) in enumerate(test_loader):
            images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
            output = model(images_1, images_2)
            test_loss += criterion(output, targets)  # sum up batch loss
            pred = torch.where(torch.sigmoid(output) > 0.5, 1, 0) # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()
            total_test_size += targets.shape[0]


    return test_loss, correct, total_test_size


def valid(model, device, valid_loader):
    model.eval()
    valid_loss = 0.0
    correct = 0
    total_valid_size = 0

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch_idx, (images_1, images_2, targets) in enumerate(valid_loader):
            images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
            output = model(images_1, images_2)
            valid_loss += criterion(output, targets)  # sum up batch loss
            pred = torch.where(torch.sigmoid(output) > 0.5, 1, 0)  # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()
            total_valid_size += targets.shape[0]

    return valid_loss, correct, total_valid_size


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Image classifier network based on transformer VIT backbone')
    parser.add_argument('--experiment-name', type=str, default="MNIST_similarity_vgg16_without_early_stopping_lightweight_with_bn_dropout", metavar='N',
                        help='experiment-name-to-store-model')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--optim-type', type=str, default="adam", metavar='N',
                        help='optimizer to use in this experiment')
    parser.add_argument('--loss-type', type=str, default="BCE", metavar='N',
                        help='loss function to use in this experiment')
    args = parser.parse_args()
    converted_dict_args = vars(args)

    base_dir_for_MNIST = "/home/user/workspace/image_classification_dataset/MNIST_similarity/"
    path_to_training_runs = "/home/user/trained_nn_models"
    args_path = os.path.join(path_to_training_runs, args.experiment_name)
    if not os.path.exists(args_path):
        os.makedirs(args_path)
        with open(os.path.join(args_path, 'params.json'), 'w') as file_object:
            json.dump(converted_dict_args, file_object)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

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

    # Define DataLoaders for each subset
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(val_set, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

    writer = SummaryWriter(os.path.join(path_to_training_runs, args.experiment_name))

    ## create an instance of siamese network, move it to the device and print its structure
    model = SiameseNetwork_MNIST().to(device=device)

    scheduler = None

    if args.optim_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    elif args.optim_type == "SGD":
        optimizer = optim.SGD(model.parameters(),  lr=args.lr, momentum=0.9)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)
    else:
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    best_model_acc = 0

    for epoch in range(args.epochs):

        print("{} / {}".format(epoch + 1,  args.epochs))
        train_loss, train_correct, total_train = train(args, model, device, train_loader, optimizer, epoch)
        train_loss = train_loss / len(train_loader)
        writer.add_scalar("Training loss", train_loss, epoch + 1)

        train_acc = (train_correct / total_train) * 100
        writer.add_scalar("Training accuracy", train_acc, epoch + 1)

        valid_loss, valid_correct, total_valid = valid(model=model, device=device, valid_loader=valid_loader)
        valid_loss = valid_loss / len(valid_loader)
        writer.add_scalar("validation loss", valid_loss, epoch + 1)

        valid_acc = (valid_correct / total_valid) * 100
        writer.add_scalar("valid accuracy", valid_acc, epoch + 1)

        print("Epoch:{}/{} AVG Training Loss:{:.3f} Training Acc {:.2f} AVG Valid Loss:{:.3f} Valid Acc {:.2f}%".format(epoch + 1,
                                                                                                                            args.epochs,
                                                                                                                            train_loss,
                                                                                                                            train_acc,
                                                                                                                            valid_loss,
                                                                                                                            valid_acc))

        if valid_acc >= best_model_acc and args.save_model:
            best_model_acc = valid_acc
            torch.save(model, os.path.join(args_path, args.experiment_name + '.pt'))


    ## perform testing here
    test_loss, test_correct, total_test = test(model=model, device=device, test_loader=test_loader)
    test_acc = (test_correct / total_test) * 100
    print("AVG Test Loss:{:.3f}, Test Acc {:.2f}%".format(test_loss, test_acc))
    writer.close()

if __name__ == '__main__':
    main()
