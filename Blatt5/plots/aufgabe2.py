from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# import numpy as np
# a)
samples, labels = make_blobs(n_samples=1000, centers=2, n_features=4, random_state=0)
samples_x = samples[:, 0]
samples_y = samples[:, 1]
plt.scatter(samples[:, 0], samples[:, 1])
plt.savefig('scatterplot_a.png')
# c)
pca = PCA()
print()
