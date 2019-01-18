import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
import scipy.constants as const
from scipy.optimize import curve_fit
from uncertainties import correlated_values, correlation_matrix
from uncertainties import ufloat
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from scipy.stats import chi2
###### Muss nicht mit abgegeben werden denke ich ########
Werte = np.array([31.6 ,31.3 ,32.2 ,30.8 ,31.2 ,31.3 ,31.9 ])
Fehler = 0.5
def chiquadtest(vals,std,hypo):
	return sum(((vals - hypo)**2)/std)
t_A = chiquadtest(Werte,Fehler,31.3)
t_B = chiquadtest(Werte,Fehler,30.7) 
print(t_A)
print(t_B)
