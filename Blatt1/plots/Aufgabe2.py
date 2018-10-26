import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
import scipy.constants as const
from scipy.optimize import curve_fit
from uncertainties import correlated_values, correlation_matrix
from uncertainties import ufloat
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)

## Sexy functions ##
def stabilGleichung(teta, gamma):
	return (2+ np.sin(teta)**2)/(np.sin(teta)**2 + (np.cos(teta)**2)/gamma**2)
def instabilGleichung(teta,beta):
	return (2+ np.sin(teta)**2)/(1-(beta**2)*np.cos(teta)**2)
def Kondition(teta,beta):
	return np.absolute(teta*(1-3*(beta**2)*np.sin(teta)*np.cos(teta))/(1-(np.cos(teta)**2)*beta**2)*(np.cos(teta)**2 -3))


## Sexy Data ##
tetahalf = np.linspace(0, np.pi, 1000)
teta = np.linspace(0, 2*np.pi, 1000)
beta = .99989
gamma = 0.01022

#Sexy Plots#
plt.subplot(3, 1, 1)
plt.plot(teta,stabilGleichung(teta, gamma), label='Stabil')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$\frac{d\sigma}{d\Omega}$')
plt.legend(loc='best')
plt.subplot(3, 1, 2)
plt.plot(teta,instabilGleichung(teta, beta), label='Instabil')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$\frac{d\sigma}{d\Omega}$')
plt.legend(loc='best')
plt.subplot(3, 1, 3)
plt.plot(tetahalf,Kondition(tetahalf,beta), label = "Kondition")
plt.xlabel(r'$\theta$')
plt.ylabel(r'$K$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('Stabiplot.pdf')


