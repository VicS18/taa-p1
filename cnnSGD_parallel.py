import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shared import *

import pickle
import csv

from torch.utils.data import DataLoader, Dataset, random_split
from time import time

DATA_AUG = False

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

train_transform = False
if DATA_AUG:
    # Define transformations for data augmentation
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),       # Random rotation within [-10, 10] degrees
        transforms.RandomResizedCrop(28, scale=(0.9, 1.0), ratio=(0.9, 1.1)),  # Random resized crop
    ])

# Create custom datasets
train_dataset = CustomDataset(train_dataset, train_labels, transform=train_transform)
test_dataset = CustomDataset(test_dataset, test_labels)

train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size

# Split the training dataset
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])


def loop():   
    global file_counter
    
    BASE_DIR = f"SGD/{DATA_AUG}_DATA_AUG"

    if not os.path.isdir(BASE_DIR):
        os.makedirs(BASE_DIR)

    file = open(f"{BASE_DIR}{file_counter}.txt", "a+")
    file_counter += 1
    file.write("\n\n\n")
    
    PLOT_DIR = f"{BASE_DIR}/plots"

    # Hyperparameters
    with open(f'{BASE_DIR}/test_performance.csv', 'w', newline='') as csvfile:
        # Prepare test performance file
        fieldnames = ['learning_rate', 'batch_size', 'accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for lrn in (0.01, 0.05):
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

                model = nn.DataParallel(ConvNet(file)).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate, weight_decay=1e-5)

                n_total_steps = len(train_loader)
                for epoch in np.arange(100): #max of 100 epochs
                    train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
                    validation_loss = validate_one_epoch(model, test_loader, criterion)
                    gen_train_loss.append(train_loss.item())
                    gen_val_loss.append(validation_loss.item())
                    print(f"Train Loss: {train_loss}")
                    print(f"Validation Loss: {validation_loss}")
                    print(f"Epoch [{epoch+1}/{100}]")
                    # file.write(f"Train Loss: {train_loss}\n")
                    # file.write(f"Validation Loss: {validation_loss}\n")
                    # file.write(f"Epoch [{epoch+1}/{100}]\n")
                    # file.write("\n")

                    gen_train_acc.append(predict_acc(model, train_loader))
                    print(f"Accuracy on Training Data: {gen_train_acc[-1]} %")

                    gen_val_acc.append(predict_acc(model, val_loader))
                    print(f"Accuracy on Validation Data: {gen_val_acc[-1]} %")

                    print()

                    if early_stopper.early_stop(validation_loss):             
                        break
                test_confusion = predict_confusion(model, test_loader)

                test_acc = 100 * np.trace(test_confusion) / np.sum(test_confusion)
                print(f"Finished Training at epoch {epoch+1}!")
                # file.write(f"Finished Training at epoch {epoch+1}!\n")

                print("Elapsed Time: " , time()-timestamp," seconds." )
                # file.write(f"Elapsed Time: {time()-timestamp,} seconds.\n")

                print(f"Accuracy of the network: {test_acc} %")
                # file.write(f"Accuracy of the network: {predict(model, test_loader)} %\n")
                # file.write(f"Accuracy of the network: {predict(model, test_loader)} %\n")

                #
                # Individual plots
                #

                # Training accuracy 
                simple_plot(gen_train_acc, "Epoch", "Accuracy (%)", f"{PLOT_DIR}/train/accuracy", bsz, "Training accuracy", lrn)

                # Training loss 
                simple_plot(gen_train_loss, "Epoch", "$J(\\Theta)$", f"{PLOT_DIR}/train/loss", bsz, "Training loss", lrn)

                # Validation accuracy 
                simple_plot(gen_val_acc, "Epoch", "Accuracy (%)", f"{PLOT_DIR}/validation/accuracy", bsz, "Validation accuracy", lrn)

                # Validation loss
                simple_plot(gen_val_loss, "Epoch", "$J(\\Theta)$", f"{PLOT_DIR}/validation/loss", bsz, "Validation loss", lrn)

                #
                # Save model
                #

                dir_path = f"{BASE_DIR}/lr_{lrn}"
                if not os.path.isdir(dir_path):
                    os.makedirs(dir_path)

                np.save(f"{dir_path}/bsz_{bsz}.npy", test_confusion)

                torch.save(model.module.state_dict(), f"{dir_path}/bsz_{bsz}.pth")

                #
                # Register data from test set predictions
                #

                writer.writerow({
                    'learning_rate': lrn, 
                    'batch_size': bsz, 
                    'accuracy': test_acc
                })

                # file.write(f"Training Loss:[")
                # for value in gen_train_loss:
                #     file.write(str(value) + ",")
                # file.write("]\n")
                # file.write(f"Validation Loss:[")
                # for value in gen_val_loss:
                #     file.write(str(value) + ",")
                # file.write("]\n")
                # file.write(f"\nTraining Accuracy:[")
                # for value in gen_train_acc:
                #     file.write(str(value) + ",")
                # file.write("]\n")
                # file.write(f"Validation Accuracy:[")
                # for value in gen_val_acc:
                #     file.write(str(value) + ",")
                # file.write("]\n")


loop()

