import numpy as np
import torch
import torch.nn as nn
import argparse
import os
import json
import torchvision.models
from sklearn.metrics import confusion_matrix
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

class ClassifierNetwork(nn.Module):
    """
            Backbone of classifier is VIT - Transformer for vision problems
    """

    def __init__(self):
        super(ClassifierNetwork, self).__init__()

        self.vit_base = torchvision.models.vit_b_16(weights='ViT_B_16_Weights.IMAGENET1K_V1')
        self.vit_base.heads = nn.Linear(in_features=768, out_features=196, bias=True)
        self.vit_base.num_classes = 196

        # initialize the weights
        self.vit_base.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, input):
        # get two images' features
        output = self.vit_base(input)
        return output

def train(args, model, device, train_loader, optimizer, epoch):
    train_loss, train_correct, total_size = 0.0, 0, 0
    model.train()

    criterion = nn.CrossEntropyLoss()

    for batch_idx, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        predictions = torch.argmax(output, 1)
        train_correct += predictions.eq(targets.view_as(predictions)).sum().item()
        total_size += targets.shape[0]
        print("Epoch {}, Training Mini-batch : {}/{}, loss value : {}, Acc. values : {} ".format(epoch + 1, batch_idx, len(train_loader), loss.item(), (predictions == targets).sum().item()))

    return train_loss, train_correct, total_size

def test(model, device, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0
    total_test_size = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images, targets = images.to(device), targets.to(device)
            output = model(images)
            test_loss += criterion(output, targets)  # sum up batch loss
            pred = torch.argmax(output, 1) # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()
            total_test_size += targets.shape[0]


    return test_loss, correct, total_test_size


def valid(model, device, valid_loader):
    model.eval()
    valid_loss = 0.0
    correct = 0
    total_valid_size = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(valid_loader):
            images, targets = images.to(device), targets.to(device)
            output = model(images)
            valid_loss += criterion(output, targets)  # sum up batch loss
            pred = torch.argmax(output, 1)  # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()
            total_valid_size += targets.shape[0]

    return valid_loss, correct, total_valid_size


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Image classifier network based on transformer VIT backbone')
    parser.add_argument('--experiment-name', type=str, default="Classify_stanfordcars_with_VIT", metavar='N',
                        help='experiment-name-to-store-model')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
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

    data_dir = "/home/user/workspace/image_classification_dataset/"

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

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to ViT-compatible size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset using ImageFolder
    #dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    ## Extract targets (class labels) for stratified split
    #targets = np.array(dataset.targets)

    # Define stratified split sizes (70% train, 15% val, 15% test)
    #splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    #train_idx, temp_idx = next(splitter.split(np.zeros(len(targets)), targets))

    # Further split temp set into validation (15%) and test (15%)
    #splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    #val_idx, test_idx = next(splitter.split(np.zeros(len(temp_idx)), targets[temp_idx]))

    # Create subsets
    #train_dataset = Subset(dataset, train_idx)
    #val_dataset = Subset(dataset, val_idx)
    #test_dataset = Subset(dataset, test_idx)

    # Create DataLoaders
    #train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    #valid_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=4)
    #test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=4)

    # Load train and test sets
    train_dataset = datasets.StanfordCars(root=data_dir, split='train', transform=transform, download=False)
    val_dataset = datasets.StanfordCars(root=data_dir, split='test', transform=transform, download=False)
    test_dataset = datasets.StanfordCars(root=data_dir, split='test', transform=transform, download=False)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=4)
    valid_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=4)



    writer = SummaryWriter(os.path.join(path_to_training_runs, args.experiment_name))

    ## create an instance of siamese network, move it to the device and print its structure
    model = ClassifierNetwork().to(device=device)

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
