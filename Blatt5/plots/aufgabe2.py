from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from scipy import linalg
# a)
samples, labels = make_blobs(n_samples=1000, centers=2, n_features=4, random_state=0)
samples_x = samples[:, 0]
samples_y = samples[:, 1]
plt.scatter(samples[:, 0], samples[:, 1])
plt.xlabel('Dim 1')
plt.ylabel('Dim 2')
plt.savefig('scatterplot_a.png')
plt.clf()
# c)
# Zentrieren um Mittelwert
mean = np.mean(samples)
samples_zentriert = samples - mean

# Analyse
pca = PCA()
pca.fit(samples_zentriert)
samples_pca = pca.transform(samples_zentriert)

covariance = pca.get_covariance()
print(covariance)
eigenval, eigenvec = linalg.eigh(covariance)  # eigh diagonalisiert mir die covariance matrix
print('Nicht sortierte Eigenwerte:')
print(eigenval)

# d)

# scatterplot
plt.scatter(samples_pca[:, 0], samples[:, 1])
plt.xlabel('Dim 1')
plt.ylabel('Dim 2')
plt.savefig('scatterplot_d.png')
plt.clf()
# histogram
plt.subplot(2, 2, 1)
plt.hist(samples_pca[:, 0], color='b', normed=True, bins=40, label='Dim 1', alpha=0.6)
plt.xlabel('Dim 1')
plt.ylabel('W.-keit')
plt.tight_layout()

plt.subplot(2, 2, 2)
plt.hist(samples_pca[:, 1], color='b', normed=True, bins=40, label='Dim 2', alpha=0.6)
plt.xlabel('Dim 2')
plt.ylabel('W.-keit')
plt.tight_layout()

plt.subplot(2, 2, 3)
plt.hist(samples_pca[:, 2], color='b', normed=True, bins=40, label='Dim 3', alpha=0.6)
plt.xlabel('Dim 3')
plt.ylabel('W.-keit')
plt.tight_layout()

plt.subplot(2, 2, 4)
plt.hist(samples_pca[:, 3], color='b', normed=True, bins=40, label='Dim 4', alpha=0.6)
plt.xlabel('Dim 4')
plt.ylabel('W.-keit')
plt.tight_layout()
plt.savefig('histogrammi.png')
