import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import csv 
import pandas as pd

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

print("Is cuda available?: ", torch.cuda.is_available())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Path to your CSV files
train_csv_file = "dataset/sign_mnist_train.csv"
test_csv_file = "dataset/sign_mnist_test.csv"

# Create datasets
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5,0.5)]) #Transform features to Tensor and normalize them
train_data = pd.read_csv(train_csv_file)
test_data = pd.read_csv(test_csv_file)

#Create and Normalize Data
train_dataset = np.array(train_data.drop(labels=['label'], axis=1))/255
train_labels =  np.array(train_data['label'])
test_dataset =  np.array(test_data.drop(labels=['label'],axis=1))/255
test_labels =  np.array(test_data['label'])
classes = torch.tensor(np.arange(0, 26), dtype=torch.long)  # 26 classes in total


test_labels = test_labels.ravel()
train_labels = train_labels.ravel()
train_dataset = train_dataset.reshape((-1, 1, 28, 28))
test_dataset = test_dataset.reshape((-1, 1, 28, 28))

print (train_dataset.shape)
print(train_labels.shape)
print(test_dataset.shape)
print(test_labels.shape)


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

# Create custom datasets
train_dataset = CustomDataset(train_dataset, train_labels)
test_dataset = CustomDataset(test_dataset, test_labels)

# HyperParameters
num_epochs = 10
batch_size = 4
learn_rate = 0.01

print("Hyper - Parameters:")
print("Epochs: ", num_epochs)
print("Batch-Size: ", batch_size)
print("Learning Rate: ", learn_rate)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Adjusted the input size for the fully connected layers
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # Output size from conv2 is 16x4x4
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 26)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Adjusted the view size based on the output size of the last convolutional layer
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Reshape the input images to have shape [batch_size, 1, 28, 28]
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print (f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Finished Training!")

with torch.no_grad(): # Predict Test Dataset
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(26)]
    n_class_samples = [0 for i in range(26)]

    for images, labels in test_loader:
        images = images.to(device)  # Add a channel dimension for grayscale images
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(labels.size(0)):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f"Accuracy of the network: {acc} %")

    for i in range(26):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i] if n_class_samples[i] > 0 else 0
        print(f"Accuracy of the class {classes[i]}: {acc} %")

