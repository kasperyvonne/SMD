import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
import scipy.constants as const
from scipy.optimize import curve_fit
from uncertainties import correlated_values, correlation_matrix
from uncertainties import ufloat
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
import pandas as pd 
from functools import reduce
##Einlesen der Daten ##
Pzero = pd.read_hdf('zwei_populationen.h5',key='P_0_10000')
Pone = pd.read_hdf('zwei_populationen.h5',key='P_1')
p1x = np.array(Pone['x'])
p1y = np.array(Pone['y'])
p0x = np.array(Pzero['x'])
p0y = np.array(Pzero['y'])
##Berechnung der Mittelwertvektoren##
mup0 = np.array([np.mean(p0x),np.mean(p0y)])
mup1 = np.array([np.mean(p1x),np.mean(p1y)])
##Berechnung der Kovarianzmatrizen##
Vp0 = np.cov(p0x,p0y)
Vp1 = np.cov(p1x,p1y)
Vp01 = np.cov([p0x,p1x],[p0y,p1y])
##Fisher Diskriminanzanalyse##
p0 = np.array([p0x,p0y])
p1 = np.array([p1x,p1y])
#Berechnung der Streuung der Klassen
S0 = np.array([p - mup0 for p in p0.T]).T
S1 = np.array([p - mup0 for p in p1.T]).T
S0 = [np.matrix([s[0],s[1]]).T*(np.matrix([s[0],s[1]])) for s in S0.T]
S0 = reduce(lambda x,y: x+y, S0)
S1 = [np.matrix([s[0],s[1]]).T*(np.matrix([s[0],s[1]])) for s in S1.T]
S1 = reduce(lambda x,y: x+y, S1)
SW = S0 + S1 #Berechnung der Gesamtstreuung = Within class scatter matrix
#SB = (mup0-mup1)**2 #Streumatrix Between class scatter matrix
lamb = SW.I*np.matrix([[(mup0[0]-mup1[0])],[(mup0[1]-mup1[1])]]) #Berechnung der Projektion
lamb = lamb.T
lambnorm = np.sqrt(np.dot(lamb,lamb.T)) #lambda normierung
lamb/= lambnorm
##1dim Hist der Populationen##
def projektion(lam,data):
	return np.squeeze(np.asarray(np.dot(lam,data)/np.dot(lam,lam.T)))
proj0 = projektion(lamb,p0)
proj1 = projektion(lamb,p1)
#histp0 = np.squeeze(np.asarray(np.dot(lamb,[p0x,p0y])))
#histp1 = np.squeeze(np.asarray(np.dot(lamb,[p1x,p1y])))
plt.hist(proj0, label='Population 0')
plt.hist(proj1, label='Population 1')
plt.xlabel(r'$\lambda_x \cdot x$')
plt.legend(loc='best')
plt.savefig('Projektion1dimhist.pdf')
###
plt.clf()
#x = np.linspace(-15,20,1000)
#y = np.linspace(-7.5,13,1000)
#xvec = np.array([x,y])
###xvec = lamb*xvec.T
##xvec = xvec.T
#y = np.squeeze(np.asarray(np.dot(lamb,xvec)))
#plt.scatter(p0x,p0y,marker='.',alpha = 0.1)
#plt.scatter(p1x,p1y,marker='.',alpha = 0.1)
##plt.scatter(x,y)
#plt.show()





#x = np.linspace(0, 10, 1000)
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
#
#
##Tabelle
## np.savetxt('tab.txt',np.column_stack([x,y]), delimiter=' & ',newline= r'\\'+'\n' )
##plt.subplot(1, 2, 1)
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
