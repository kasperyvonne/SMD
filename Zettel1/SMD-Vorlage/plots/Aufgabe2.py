import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
import scipy.constants as const
from scipy.optimize import curve_fit
from uncertainties import correlated_values, correlation_matrix
from uncertainties import ufloat
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
teta = np.linspace(0, 2*np.pi, 1000)
#mhub = const.value('Bohr magneton') #das gelibete Borhsche Magneton zeigt wie man Scipy Constants benutzt
#def mittel(x):              #the real mean()-ing of life
#    return ufloat(np.mean(x),np.std(x,ddof=1)/np.sqrt(len(x)))
#def relf(l,m):  #in Prozent
#    return (np.absolute(l-m)/l)*100
#
#
##Fit
#params , cov = curve_fit(f , x ,y )
#params = correlated_values(params, cov)
#for p in params:
#    print(p)
def stabilGleichung(teta, gamma):
	return (2+ np.sin(teta)**2)/(np.sin(teta)**2 + (np.cos(teta)**2)/gamma**2)
def instabilGleichung(teta,beta):
	return (2+ np.sin(teta)**2)/(1-(beta**2)*np.cos(teta)**2)

beta = .99989
gamma = 0.01022
plt.subplot(2, 1, 1)
plt.plot(teta,stabilGleichung(teta, gamma), label='Stabil')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$\frac{d\sigma}{d\Omega}$')
plt.legend(loc='best')
#plt.show()
#plt.savefig('Stabilplot.pdf')
#plt.clf()
plt.subplot(2, 1, 2)
plt.plot(teta,instabilGleichung(teta, beta), label='Instabil')
#plt.plot(x, y, label='Kurve')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$\frac{d\sigma}{d\Omega}$')
plt.legend(loc='best')

# in matplotlibrc leider (noch) nicht möglich
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
#plt.show()
plt.savefig('Stabiplot.pdf')


#Tabelle
# np.savetxt('tab.txt',np.column_stack([x,y]), delimiter=' & ',newline= r'\\'+'\n' )
#plt.subplot(1, 2, 1)

