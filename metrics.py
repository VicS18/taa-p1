from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score

import pandas as pd
import numpy as np
import seaborn as sns

from shared import *

from torch.utils.data import DataLoader

directory = "."

# Load the best model and its parameters
best_model_info = find_best_model(directory)

best_model_info = find_best_model(directory)

print("BEST MODEL:", best_model_info)

_, batch_size, ___, ____ = best_model_info

model_parameters = load_model_parameters(best_model_info)

# Load model
model = ConvNet().to(device)
model.load_state_dict(model_parameters)
model.eval()

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

# Apply the model to the test dataset to obtain predictions
test_results = predict_scores(model, test_loader)
predicted_labels = np.argmax(test_results, axis=1)

confusion_matrix = predict_confusion(model, test_loader)

# Calculate evaluation metrics
accuracy = accuracy_score(test_labels_all, predicted_labels)
precision = precision_score(test_labels_all, predicted_labels, average='weighted')
recall = recall_score(test_labels_all, predicted_labels, average='weighted')
f1 = f1_score(test_labels_all, predicted_labels, average='weighted')
balanced_accuracy = balanced_accuracy_score(test_labels_all, predicted_labels)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Balanced Accuracy:", balanced_accuracy)


confusion_matrix = confusion_matrix.astype(int)

# Plot confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()