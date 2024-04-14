import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('dataset/sign_mnist_train.csv')

pixel_values = df.iloc[:, 0].values.flatten()

# Plot a histogram of class frequencies
plt.hist(pixel_values, bins=26, color='blue', alpha=0.7)
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title('Histogram of Class frequencies')
plt.show()
