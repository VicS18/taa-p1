import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader, Dataset, random_split
from time import time

global file_counter
file_counter = 1

print("Is cuda available?: ", torch.cuda.is_available())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Path to your CSV files
train_csv_file = "dataset/sign_mnist_train.csv"
test_csv_file = "dataset/sign_mnist_test.csv"


# Create datasets
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

train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size

# Split the training dataset
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

def train_one_epoch(model, train_loader, criterion, optimizer): 
    model.train()
    train_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss

def validate_one_epoch(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    num_samples = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss
            num_samples += images.size(0)
            avg_val_loss = val_loss / num_samples
    return avg_val_loss


class ConvNet(nn.Module):
    def __init__(self, file):
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

def predict(model, test_loader):
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
    return acc
    #    for i in range(26):
    #        acc = 100.0 * n_class_correct[i] / n_class_samples[i] if n_class_samples[i] > 0 else 0
    #        print(f"Accuracy of the class {classes[i]}: {acc} %")

def loop():   
    global file_counter
    file = open(f"NAdam{file_counter}.txt", "a+")
    file_counter += 1
    file.write("\n\n\n")
    # Hypearameters
    for lrn in (0.0001,):
        for bsz in (16, 32, 64, 128):
            gen_train_loss =[]
            gen_val_loss = []
            gen_train_acc =[]
            gen_val_acc = []
            gen_test_acc = []
            early_stopper = EarlyStopper(patience=5, min_delta=1e-4)

            batch_size = bsz
            learn_rate = lrn
            timestamp = time()

            print("Hyper - Parameters:")
            print("Batch-Size: ", batch_size)
            print("Learning Rate: ", learn_rate)
            file.write("Hyper - Parameters:\n")
            file.write(f"Batch-Size: {batch_size}\n")
            file.write(f"Learning Rate: {learn_rate}\n")
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            model = ConvNet(file).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.NAdam(model.parameters(), lr=learn_rate, weight_decay=1e-5)

            n_total_steps = len(train_loader)
            for epoch in np.arange(200): #max of 200 epochs
                train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
                validation_loss = validate_one_epoch(model, test_loader, criterion)
                gen_train_loss.append(str(train_loss.item()))
                gen_val_loss.append(str(validation_loss.item()))
                print(f"Train Loss: {train_loss}")
                print(f"Validation Loss: {validation_loss}")
                print(f"Epoch [{epoch+1}/{200}]")
                #file.write(f"Train Loss: {train_loss}\n")
                #file.write(f"Validation Loss: {validation_loss}\n")
                #file.write(f"Epoch [{epoch+1}/{200}]\n")
                #file.write("\n")
                gen_train_acc.append(predict(model, train_loader))
                print(f"Accuracy on Training Data: {gen_train_acc[-1]} %")

                gen_val_acc.append(predict(model, val_loader))
                print(f"Accuracy on Validation Data: {gen_val_acc[-1]} %")
                
                gen_test_acc.append(predict(model, test_loader))
                print(f"Accuracy on Test Data: {gen_test_acc[-1]} %")

                print()

                if early_stopper.early_stop(validation_loss):             
                    break

            print(f"Finished Training at epoch {epoch+1}!")
            file.write(f"Finished Training at epoch {epoch+1}!\n")

            print("Elapsed Time: " , time()-timestamp," seconds." )
            file.write(f"Elapsed Time: {time()-timestamp,} seconds.\n")

            print(f"Accuracy of the network: {predict(model, test_loader)} %")
            file.write(f"Accuracy of the network: {predict(model, test_loader)} %\n")

            file.write(f"Training Loss:[")
            for value in gen_train_loss:
                file.write(str(value) + ",")
            file.write("]\n")
            file.write(f"Validation Loss:[")
            for value in gen_val_loss:
                file.write(str(value) + ",")
            file.write("]\n")
            file.write(f"\nTraining Accuracy:[")
            for value in gen_train_acc:
                file.write(str(value) + ",")
            file.write("]\n")
            file.write(f"Validation Accuracy:[")
            for value in gen_val_acc:
                file.write(str(value) + ",")
            file.write("]\n")


loop()

