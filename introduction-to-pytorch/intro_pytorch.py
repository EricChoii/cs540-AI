import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])    
    train_set=datasets.FashionMNIST('./data',train=True, download=True,transform=custom_transform)
    test_set=datasets.FashionMNIST('./data', train=False, transform=custom_transform)
    if training == True:
      train_loader = torch.utils.data.DataLoader(train_set, batch_size = 64)
      return train_loader
    else:
      test_loader = torch.utils.data.DataLoader(test_set, batch_size = 64)
      return test_loader



def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128,64),
        nn.ReLU(),
        nn.Linear(64,10)
    )
    return model




def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    size = len(train_loader.dataset)
    model.train()
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    running_loss = 0.0
    correct = 0
    for t in range(T):
      for batch, (X,y) in enumerate(train_loader):
        pred = model(X)
        opt.zero_grad()
        loss = criterion(pred, y)
        loss.backward()
        opt.step()
        running_loss += loss.item()
        correct += int((pred.argmax(1) == y).type(torch.float).sum().item())        
      print("Train Epoch: {} 	 Accuracy: {}/{}({:.2f}%)	Loss: {:.3f}".format(t,correct,size,correct/size * 100,running_loss/(size/50)))
      running_loss = 0.0
      correct = 0
    


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    model.eval()
    test_loss, correct = 0.0, 0
    with torch.no_grad():
      for X, y in test_loader:
        pred = model(X)
        loss = criterion(pred, y)
        test_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    if(show_loss == False):
      print("Accuracy: {:.2f}%".format(correct * 100))
    else:
      print("Average loss: {:.4f}\nAccuracy: {:.2f}%".format(test_loss, correct * 100))
    



def predict_label(model, test_images, index):  
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """ 
    img = test_images[index]
    with torch.no_grad():
      logits = model(img)
    prob = F.softmax(logits, dim=1)
    prob_list = list(prob.numpy()[0])
    dic = {}
    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
    for i in range(10):
      dic[prob_list[i]] = class_names[i]
    prob_sorted = np.sort(prob_list)[::-1]
    for p in prob_sorted[0:3]:
      print("{}: {:.2f}%".format(dic[p], p*100))


if __name__ == '__main__':  
    # Q1
    train_loader = get_data_loader()
    print(type(train_loader))
    print(train_loader.dataset)
    test_loader = get_data_loader(training=False)
    print("-"*20)
    # Q2
    model = build_model()
    print(model)
    print("-"*20)
    # Q3
    criterion = nn.CrossEntropyLoss()
    train_model(model, train_loader, criterion, T = 5)
    print("-"*20)
    # Q4
    evaluate_model(model, test_loader, criterion, show_loss = False)
    evaluate_model(model, test_loader, criterion, show_loss = True)
    print("-"*20)
    # Q5
    pred_set, _ = iter(test_loader).next()
    predict_label(model, pred_set, 1)
