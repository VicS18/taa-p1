import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file into a DataFrame
df = pd.read_csv('dataset/sign_mnist_train.csv')

frequencies = df.iloc[:, 0].values

print((frequencies == 9).sum())

hist, bin_edges = np.histogram(frequencies, bins=25)

for i in range(len(hist)):
    print(i,":",hist[i])

# Plot a histogram of class frequencies
plt.hist(frequencies, bins=25, color='blue', alpha=0.7)
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title('Histogram of Class frequencies')
plt.show()
