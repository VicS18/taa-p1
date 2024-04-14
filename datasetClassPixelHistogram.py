import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('dataset/sign_mnist_train.csv')

# Group the data by class labels
grouped = df.groupby('label')

# Define the number of rows and columns for the subplots
num_rows = 4  
num_cols = 6  

# Create a figure and subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 10))

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Iterate over each class
for i, (label, group) in enumerate(grouped):
    # Extract pixel values for the current class
    pixel_values = group.iloc[:, 1:].values.flatten()
    
    # Plot a histogram of pixel intensities
    axes[i].hist(pixel_values, bins=256, color='blue', alpha=0.7)
    axes[i].set_title(f'Class {label}')
    axes[i].set_xlabel('Pixel Intensity')
    axes[i].set_ylabel('Frequency')

# Adjust layout and show plot
plt.tight_layout()
plt.show()
