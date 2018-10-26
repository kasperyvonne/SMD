import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
import scipy.constants as const
from scipy.optimize import curve_fit
from uncertainties import correlated_values, correlation_matrix
from uncertainties import ufloat
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)

def Maxwell(x):
  return (2*np.exp(-2*(x**2))*12*(x**2))

x = np.linspace(0, 10, 1000)
plt.plot(x, Maxwell(x))

plt.show()
