# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(seed=7)

# b)
# pop sind Populationen, mean Mittelwerte und cov Kovarianzmatrizen
mean0 = (0, 3)
cov0 = ([12.25, 8.19], [8.19, 6.67])
pop0 = np.random.multivariate_normal(mean0, cov0, 10000)

mean1 = (6, 3.1)
cov1 = ([12.25, 7.351323], [7.351323, 5.410276])
pop1 = np.random.multivariate_normal(mean1, cov1, 10000)

plt.scatter(pop0[:, 0], pop0[:, 1], c='r')
plt.scatter(pop1[:, 0], pop1[:, 1])
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('scatterplot.png')

# c)
# pop sind Populationen, mu_ Mittelwerte und cova Kovarianzmatrizen
mu_0_x = np.mean(pop0[:, 0])
mu_0_y = np.mean(pop0[:, 1])

cova0 = np.cov(pop0[:, 0], pop0[:, 1])

mu_1_x = np.mean(pop1[:, 0])
mu_1_y = np.mean(pop1[:, 1])

cova1 = np.cov(pop1[:, 0], pop1[:, 1])

pop_ges_x = np.array([])
pop_ges_x = np.append(pop_ges_x, pop0[:, 0])
pop_ges_x = np.append(pop_ges_x, pop1[:, 0])
mu_ges_x = np.mean(pop_ges_x)

pop_ges_y = np.array([])
pop_ges_y = np.append(pop_ges_y, pop0[:, 1])
pop_ges_y = np.append(pop_ges_y, pop1[:, 1])
mu_ges_y = np.mean(pop_ges_y)

pop_ges = np.array([pop_ges_x, pop_ges_y])
cova_ges = np.cov(pop_ges_x, pop_ges_y)

# Essensausgabe:
print('Population 0 x : ')
print(mu_0_x)
print('Population 0 y : ')
print(mu_0_y)
print('Kovarianzmatrix 0 :')
print(cova0)
print()
print('Population 1 x : ')
print(mu_1_x)
print('Population 1 y : ')
print(mu_1_y)
print('Kovarianzmatrix 1 :')
print(cova1)
print()
print('Population Gesamt x : ')
print(mu_ges_x)
print('Population Gesamt y : ')
print(mu_ges_y)
print('Kovarianzmatrix Gesamt :')
print(cova_ges)
