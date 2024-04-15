import os
import pandas as pd
import torch
import numpy as np

from shared import *

from torch.utils.data import DataLoader

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt

def plot_roc_curve(y_true, y_score, num_classes):
    fig, axs = plt.subplots(num_classes // 5, 5, figsize=(20, 16))

    # Compute ROC curve and ROC area for each class
    for i in range(num_classes):
        ax = axs[i // 5, i % 5]

        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve for Class %d' % i)
        ax.legend(loc="lower right")

    # Adjust layout
    plt.tight_layout()
    plt.show()


# Path to the directory containing the model directories
directory = "."

# Find the best model
best_model_info = find_best_model(directory)

print("BEST MODEL:", best_model_info)

_, batch_size, ___, ____ = best_model_info

# Load the parameters from the best model
model_parameters = load_model_parameters(best_model_info)

# Apply the model parameters to the test dataset and obtain scores
test_dataset = "dataset/sign_mnist_test.csv"

test_data = pd.read_csv(test_dataset)
test_labels_all = np.array(test_data['label'])

# Set class 9 to have no examples
test_labels_all[test_labels_all == 9] = 25

test_dataset = np.array(test_data.drop(labels=['label'], axis=1)) / 255
test_dataset = test_dataset.reshape((-1, 1, 28, 28))
test_dataset = CustomDataset(test_dataset, test_labels_all)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load model
model = ConvNet().to(device)
model.load_state_dict(model_parameters)
model.eval()

test_results = predict_scores(model, test_loader)

confusion_matrix = predict_confusion(model, test_loader)

class_accuracy = {}
for i in range(26):
    class_correct = confusion_matrix[i, i]
    class_total = confusion_matrix[i, :].sum()
    class_accuracy[i] = class_correct / class_total if class_total > 0 else 0

test_labels_all = test_labels_all.reshape((test_labels_all.shape[0], 1))

test_labels_all = np.append(test_labels_all, np.array([[25]]), axis=0)
test_results = np.append(test_results, np.zeros((1, 25)), axis=0)

encoder = OneHotEncoder(sparse=False)
# Fit and transform y_true to one-hot encoding array
y_true_onehot = encoder.fit_transform(test_labels_all.reshape(-1, 1))

num_classes = 25

plot_roc_curve(y_true_onehot, test_results, num_classes)
