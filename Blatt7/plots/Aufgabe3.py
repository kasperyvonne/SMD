import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
import scipy.constants as const
from scipy.optimize import curve_fit
from uncertainties import correlated_values, correlation_matrix
from uncertainties import ufloat
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
import pandas as pd

P_0 = pd.read_hdf('populationen.hdf5', key ='P_0')
P_1 = pd.read_hdf('populationen.hdf5', key ='P_1')
Pges = pd.DataFrame({
	'x': np.append(P_0['x'],P_1['x']),
	'y': np.append(P_0['y'],P_1['y']),
	#'X': np.append(Pges['x'],Pges['y']),
	#'classmask' : np.append(np.zeros(len(Pges['x'])),np.ones(len(Pges['y']))),
	'label' : np.append(np.zeros(len(P_0)),np.ones(len(P_1))), })
Wone = np.ones((2,len(Pges['X'])))
bone = np.ones(2) 
def ffunk(k,W,b):
	if k == 0:
		return np.dot(W.T[k],Pges['x']) + b[0]
	else:
		return np.dot(W.T[k],Pges['y']) + b[1]
def softmax(k,i,W,b):
	return np.exp(ffunk(W,x,b)) /(np.exp(W[
for i in range(100):
	


#plt.plot(x, y, label='Kurve')
#plt.xlabel(r'$\alpha \:/\: \si{\ohm}$')
#plt.ylabel(r'$y \:/\: \si{\micro\joule}$')
#plt.legend(loc='best')
#plt.savefig('A3plot.pdf')

