# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms


class LeNet(nn.Module):
    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()
        # certain definitions
        self.conv1 = nn.Conv2d(3,6,5,1,0)
        self.conv2 = nn.Conv2d(6,16,5,1,0)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(400,256)
        self.linear2 = nn.Linear(256,128)
        self.linear3 = nn.Linear(128,num_classes)

    def forward(self, x):
        shape_dict = {}
        # certain operations
        x = F.max_pool2d(F.relu(self.conv1(x)), 2,2,0)
        shape_dict[1] = x.shape
        x = F.max_pool2d(F.relu(self.conv2(x)), 2,2,0)
        shape_dict[2] = x.shape
        x = self.flatten(x)
        shape_dict[3] = x.shape
        x = F.relu(self.linear1(x))
        shape_dict[4] = x.shape
        x = F.relu(self.linear2(x))
        shape_dict[5] = x.shape
        x = self.linear3(x)
        shape_dict[6] = x.shape
        out = x
        return out, shape_dict

def count_model_params():
    '''
    return the number of trainable parameters of LeNet.
    '''
    model = LeNet()
    model_params = 0.0
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model_params / 1e6    


def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        ###################################
        # fill in the standard training loop of forward pass,
        # backward pass, loss computation and optimizer step
        ###################################

        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc
