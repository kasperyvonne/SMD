import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
import scipy.constants as const
from scipy.optimize import curve_fit
from uncertainties import correlated_values, correlation_matrix
from uncertainties import ufloat
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
import scipy.stats as stat
## Make some Randoms ##
unicorn = np.random.uniform(0,1,100)
## Dat Algorithm ##
def Metropolis(xzero,num, step):
	position = xzero
	randoms = np.array([])	
	while (num != np.shape(randoms)):
		xnext = np.random.uniform((position - step),(position + step),1)
		P = min(1, (stat.uniform.pdf(xnext,position,step) / stat.uniform.pdf(position,position,step)))
		rand = np.random.uniform(0,1,1)
		if rand <= P:
			position = xnext
			randoms = np.append(randoms,position)
		else:
			print("continued")
			continue
	return randoms
print(Metropolis(10,1,2))		

#x = np.linspace(0, 10, 1000)
#mhub = const.value('Bohr magneton') #das gelibete Borhsche Magneton zeigt wie man Scipy Constants benutzt
#def mittel(x):              #the real mean()-ing of life
#    return ufloat(np.mean(x),np.std(x,ddof=1)/np.sqrt(len(x)))
#def relf(l,m):  #in Prozent
#    return (np.absolute(l-m)/l)*100
#
##
##Fit
#params , cov = curve_fit(f , x ,y )
#params = correlated_values(params, cov)
#for p in params:
#    print(p)
#

#Tabelle
# np.savetxt('tab.txt',np.column_stack([x,y]), delimiter=' & ',newline= r'\\'+'\n' )
#plt.subplot(1, 2, 1)
#plt.plot(x, y, label='Kurve')
#plt.xlabel(r'$\alpha \:/\: \si{\ohm}$')
#plt.ylabel(r'$y \:/\: \si{\micro\joule}$')
#plt.legend(loc='best')
#plt.savefig('build/plot.pdf')
#plt.clf()
##plt.subplot(1, 2, 2)
#plt.plot(x, y, label='Kurve')
#plt.xlabel(r'$\alpha \:/\: \si{\ohm}$')
#plt.ylabel(r'$y \:/\: \si{\micro\joule}$')
#plt.legend(loc='best')
#
## in matplotlibrc leider (noch) nicht möglich
##plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
#plt.savefig('build/plot2.pdf')
