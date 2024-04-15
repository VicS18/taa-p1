import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from shared import *

from skorch import NeuralNetClassifier
from skorch.dataset import Dataset
from skorch.helper import predefined_split
from skorch.callbacks import EpochScoring

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

# Path to the directory containing the model directories
directory = "."

# Find the best model
best_model_info = find_best_model(directory)

print("BEST MODEL:", best_model_info)

_, batch_size, ___, ____ = best_model_info

# Load the parameters from the best model
model_path = os.path.join(directory, f"{best_model_info[2]}/{best_model_info[3]}/lr_{best_model_info[0]}/bsz_{best_model_info[1]}.pth")

# Define Skorch classifier
net = NeuralNetClassifier(
    ConvNet,
    device=device
)

# Load the trained model parameters
net.initialize()
net.load_params(f_params=model_path)

# Apply the model parameters to the test dataset and obtain scores
test_dataset = "dataset/sign_mnist_test.csv"
test_data = pd.read_csv(test_dataset)
test_labels_all = np.array(test_data['label'])

# Set class 9 to have no examples
test_labels_all[test_labels_all == 9] = 25

test_dataset = np.array(test_data.drop(labels=['label'], axis=1), dtype=np.float32) / 255
test_dataset = test_dataset.reshape((-1, 1, 28, 28))

# Obtain predicted probabilities using predict_proba
y_proba = net.predict_proba(test_dataset)

# Obtain classes predictions using predict
y_pred = net.predict(test_dataset)

# One hot encode the test labels
encoder = OneHotEncoder(sparse=False)
y_true_onehot = encoder.fit_transform(test_labels_all.reshape(-1, 1))

# Compute ROC curve and ROC area for each class
num_classes = 24
fig, axs = plt.subplots(num_classes // 5 + 1, 5, figsize=(20, 16))
for i in range(num_classes):
    ax = axs[i // 5, i % 5]
    fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_proba[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve for Class %d' % i)
    ax.legend(loc="lower right")

# Adjust layout
plt.tight_layout()
plt.show()
