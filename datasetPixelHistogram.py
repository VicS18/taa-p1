import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('dataset/sign_mnist_train.csv')

# Extract pixel values (assuming pixel columns start from the second column)
pixel_values = df.iloc[:, 1:].values.flatten()

# Plot a histogram of pixel intensities
plt.hist(pixel_values, bins=256, color='blue', alpha=0.7)
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Histogram of Pixel Intensities')
plt.show()
