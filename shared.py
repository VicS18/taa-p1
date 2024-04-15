import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import torch.nn.functional as F

import numpy as np

import os
import pickle

import pandas as pd

from torch.utils.data import Dataset

print("Is cuda available?: ", torch.cuda.is_available())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def simple_plot(x, x_label, y_label, dir_path, batch_size, title, lr):
    plt.plot(x)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.figtext(.8, .8, f"Batch size = {batch_size}")
    plt.title(title)

    fig = plt.figure(1)
    
    dir_path = f"{dir_path}/lr_{lr}"
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    plt.savefig(f"{dir_path}/bsz_{batch_size}.png")

    with open(f"{dir_path}/bsz_{batch_size}_fig.obj", 'wb') as file:
        pickle.dump(fig, file)

    plt.close(fig)
    plt.clf()
    plt.cla()

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class CustomDataset(Dataset):
    def __init__(self, data, labels, transform = False):
        self.transform = transform
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image,label = self.data[index], self.labels[index]
        if self.transform:
            image = self.transform(image) 
        return image,label
    
class ConvNet(nn.Module):
    def __init__(self, file = None):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 128, 3)
        self.conv3 = nn.Conv2d(128, 512, 3)
        # Adjusted the input size for the fully connected layers
        self.fc1 = nn.Linear(128*5*5, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Adjusted the view size based on the output size of the last convolutional layer
        x = x.view(-1, 128*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def train_one_epoch(model, train_loader, criterion, optimizer): 
    model.train()
    train_loss = 0

    for _, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        loss = criterion(outputs, labels)
        train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # len(train_loader) = number of batches in total
    return train_loss / len(train_loader) 

def validate_one_epoch(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for _, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss
    # len(val_loader) = number of batches in total
    return val_loss / len(val_loader)

def predict_acc(model, test_loader):
    with torch.no_grad(): # Predict Test Dataset
        n_correct = 0
        n_samples = 0

        for images, labels in test_loader:
            images = images.to(device)  # Add a channel dimension for grayscale images
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

    return 100 * n_correct / n_samples

def predict_confusion(model, test_loader):
    with torch.no_grad(): # Predict Test Dataset
        n_correct = 0
        n_samples = 0
        confusion_matrix = np.zeros((26,26))
        
        for images, labels in test_loader:
            images = images.to(device)  # Add a channel dimension for grayscale images
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            # Update confusion matrix
            for i in range(labels.size(0)):
                confusion_matrix[labels[i]][predicted[i]] += 1
    return confusion_matrix

def predict_scores(model, test_loader):
    with torch.no_grad(): # Predict Test Dataset
        scores = np.zeros((0, 25))
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            scores = np.concatenate((scores,outputs.cpu()), axis=0)
        
    return scores

# Function to parse test_performance.csv files and find the best accuracy
def find_best_model(directory):
    best_accuracy = 0.0
    best_model_info = None
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == 'test_performance.csv':
                df = pd.read_csv(os.path.join(root, file))
                max_accuracy = df['accuracy'].max()
                if max_accuracy > best_accuracy:
                    best_accuracy = max_accuracy
                    learning_rate, batch_size = df.loc[df['accuracy'].idxmax(), ['learning_rate', 'batch_size']]
                    batch_size = int(batch_size)
                    optimizer = os.path.basename(os.path.dirname(root))
                    data_augmentation = os.path.basename(root)
                    best_model_info = (learning_rate, batch_size, optimizer, data_augmentation)
    
    return best_model_info

# Function to load parameters from the best model
def load_model_parameters(model_info):
    learning_rate, batch_size, optimizer, data_augmentation = model_info
    model_path = f"{optimizer}/{data_augmentation}/lr_{learning_rate}/bsz_{batch_size}.pth"
    
    return torch.load(model_path)