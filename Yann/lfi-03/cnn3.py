# source code inspireed by
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#model-training-and-validation-code

import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CATEGORIES = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

# implement your own NNs
class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()
        
        # Conv2D: 32 Filters, 3x3 Kernel, Padding 'same'
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # Conv2D: 32 Filters. 
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=0)
        # MaxPooling 2x2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Dropout 25%
        self.drop1 = nn.Dropout(p=0.25)
        # Conv2D: 64 Filters, padding =1
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Conv2D: 64 Filters
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0)
        # MaxPooling 2x2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Dropout 25%
        self.drop2 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(64 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.drop2(x)

        # Flatten
        x = x.view(x.size(0), -1) 
        
        # Dense
        x = F.relu(self.fc1(x))
        x = self.fc2(x) 
        return x

    def name(self):
        return "MyNeuralNetwork"

class LeNet5Adapted(nn.Module):
    def __init__(self):
        super(LeNet5Adapted, self).__init__()
        
        # Conv 1 : Input 28x28 -> Output 26x26 (kernel 3x3, valid padding)
        # 6 filters
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
        
        # AvgPool 1 : 26x26 -> 13x13
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Conv 2 : Input 13x13 -> Output 11x11 
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        
        # AvgPool 2 : 11x11 -> 5x5
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)
        
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)
        
        x = x.view(-1, 16 * 5 * 5)
        
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        
        return x

    def name(self):
        return "LeNet5Adapted"

class AlexNetAdapted(nn.Module):
    def __init__(self):
        super(AlexNetAdapted, self).__init__()
        
        # --- PARTIE CONVOLUTION (5 Couches) ---
        self.features = nn.Sequential(
            # Conv 1
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # -> 14x14

            # Conv 2
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # -> 7x7

            # Conv 3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv 4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv 5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # -> 3x3
        )
        
        # --- PARTIE CLASSIFIER (3 Couches Denses) ---
        # La sortie de features est 256 canaux * 3 * 3 = 2304
        self.classifier = nn.Sequential(
            # Dropout pour éviter l'overfitting (classique AlexNet)
            nn.Dropout(p=0.5),
            nn.Linear(256 * 3 * 3, 4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), # On garde la largeur typique d'AlexNet
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, 10), # Sortie 10 classes
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 3 * 3) # Flatten
        x = self.classifier(x)
        return x

    def name(self):
        return "AlexNetAdapted"
    


def training(model, data_loader, optimizer, criterion, device):
    model.train()

    running_loss = 0.0
    running_corrects = 0

    for batch_idx, (inputs, labels) in enumerate(data_loader):

        # zero the parameter gradients
        optimizer.zero_grad()

        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)

        # backward
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        if batch_idx % 10 == 0:
            print(f'Training Batch: {batch_idx:4} of {len(data_loader)}')

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = running_corrects.double() / len(data_loader.dataset)

    print('-' * 10)
    print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')

    return epoch_loss, epoch_acc


def test(model, data_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    # do not compute gradients
    with torch.no_grad():

        for batch_idx, (inputs, labels) in enumerate(data_loader):

            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            if batch_idx % 10 == 0:
                print(f'Test Batch: {batch_idx:4} of {len(data_loader)}')

        epoch_loss = running_loss / len(data_loader.dataset)
        epoch_acc = running_corrects.double() / len(data_loader.dataset)

    print('-' * 10)
    print(f'Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')

    return epoch_loss, epoch_acc


def plot(train_history, test_history, metric, num_epochs, model_name):

    plt.title(f"Validation/Test {metric}_{model_name} vs. Number of Training Epochs")
    plt.xlabel(f"Training Epochs")
    plt.ylabel(f"Validation/Test {metric}_{model_name}")
    plt.plot(range(1, num_epochs + 1), train_history, label="Train")
    plt.plot(range(1, num_epochs + 1), test_history, label="Test")
    plt.ylim((0, 1.))
    plt.xticks(np.arange(1, num_epochs + 1, 1.0))
    plt.legend()
    plt.savefig(f"{metric}_{model_name}.png")
    # plt.show()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set seed for reproducability
torch.manual_seed(0)

# hyperparameter
# TODO: find good hyperparameters
batch_size = 50
num_epochs = 12
learning_rate = 0.01
momentum = 0.9

transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()
])

# load train and test data
root = './data'
train_set = datasets.FashionMNIST(root=root,
                                  train=True,
                                  transform=transform,
                                  download=True)
test_set = datasets.FashionMNIST(root=root,
                                 train=False,
                                 transform=transform,
                                 download=True)

loader_params = {
    'batch_size': batch_size,
    'num_workers': 0  # increase this value to use multiprocess data loading

}

train_loader = DataLoader(dataset=train_set, shuffle=True, **loader_params)
test_loader = DataLoader(dataset=test_set, shuffle=False, **loader_params)

## model setup
Model = [MyNeuralNetwork().to(device), LeNet5Adapted().to(device), AlexNetAdapted().to(device)]
for model in Model:
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    train_acc_history = []
    test_acc_history = []

    train_loss_history = []
    test_loss_history = []

    best_acc = 0.0
    since = time.time()

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # train
        training_loss, training_acc = training(model, train_loader, optimizer,
                                            criterion, device)
        train_loss_history.append(training_loss)
        train_acc_history.append(training_acc)

        # test
        test_loss, test_acc = test(model, test_loader, criterion, device)
        test_loss_history.append(test_loss)
        test_acc_history.append(test_acc)

        # overall best model
        if test_acc > best_acc:
            best_acc = test_acc
            #  best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(
        f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s'
    )
    print(f'Best val Acc: {best_acc:4f}')

    # plot loss and accuracy curves
    train_acc_history = [h.cpu().numpy() for h in train_acc_history]
    test_acc_history = [h.cpu().numpy() for h in test_acc_history]

    plot(train_acc_history, test_acc_history, 'accuracy', num_epochs, model.name())
    plot(train_loss_history, test_loss_history, 'loss', num_epochs, model.name())

    # plot examples
    example_data, _ = next(iter(test_loader))
    with torch.no_grad():
        output = model(example_data)

        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.tight_layout()
            plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
            plt.title("Pred: {}".format(CATEGORIES[output.data.max(
                1, keepdim=True)[1][i].item()]))
            plt.xticks([])
            plt.yticks([])
        plt.savefig(f"examples_{model.name()}.png")
        # plt.show()
