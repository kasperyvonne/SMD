import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
import scipy.constants as const
from scipy.optimize import curve_fit
from uncertainties import correlated_values, correlation_matrix
from uncertainties import ufloat
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
x = np.linspace(0, 10, 1000)
mhub = const.value('Bohr magneton') #das gelibete Borhsche Magneton zeigt wie man Scipy Constants benutzt
def mittel(x):              #the real mean()-ing of life
    return ufloat(np.mean(x),np.std(x,ddof=1)/np.sqrt(len(x)))
def relf(l,m):  #in Prozent
    return (np.absolute(l-m)/l)*100
### Some Sweet Random Data ###
TheAnswerToTheUltimateQuestionOfLifeTheUniverseAndEverything = 42
np.random.seed(TheAnswerToTheUltimateQuestionOfLifeTheUniverseAndEverything)
u = np.random.uniform(0,1,100)
### Aufgaben a bis d - Fkt implimentierungen ###
def gleichverteili(uniforms, xmin,xmax):
	return (u*(xmax - xmin) + xmin)

def expotentialverteili(uniform, tau):
	return (- tau * np.log(1-u))

def potenzverteili(uniform,n,xmin,xmax):
	return ((uniform*(xmax**(1-n) - xmin**(1-n)) + xmin**(1-n))**(1/(1-n))) 

def cauchyverteili(uniforms):
	return np.tan(np.pi*(uniforms + 1))
### Aufgabenteil e ###
binmid, counts = np.genfromtxt('empirisches_histogramm.csv',delimiter=',' ,skip_header =1, unpack = True)
wahrschs = counts/sum(counts)
kumuwahr = np.array([])
for w in wahrschs:
	if w == wahrschs[0]:
		kumuwahr = np.append(kumuwahr,w)
	else:
		kumuwahr = np.append(kumuwahr,w + kumuwahr[-1])
def diskretverteili(uniforms):
	vals = np.array([])
	for uni in uniforms:
		if uni > max(kumuwahr):
			continue
		else:
			vals = np.append(vals,binmid[kumuwahr == kumuwahr[kumuwahr >= uni][0]])
	return vals	
plt.title('Diskrete Zufallsverteilung')
plt.hist(diskretverteili(u))
#plt.show()
plt.savefig('A1eplot.pdf')
plt.clf()


### Plots a bis d ### 
plt.subplot(2, 2, 1)
plt.title('Beide Gleichverteilungen')
plt.hist(u, label="Grenzen [0,1]")
plt.hist(gleichverteili(u,2,7), label= "Grenzen [2,7]")
plt.legend(loc='best')

plt.subplot(2, 2, 2)
plt.title('Expotentialverteilung')
plt.hist(expotentialverteili(u, 2),label="Tau = 2")
plt.legend(loc='best')

plt.subplot(2, 2, 3)
plt.title('Potenzverteilung')
plt.hist(potenzverteili(u, 2, 2,7),label="n=2 , Grenzen [2,7]")
plt.legend(loc='best')

plt.subplot(2, 2, 4)
plt.title('Cauchyverteilung')
plt.hist(cauchyverteili(u))

plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
#plt.show()
plt.savefig('A1abcd.pdf')


##Fit
#params , cov = curve_fit(f , x ,y )
#params = correlated_values(params, cov)
#for p in params:
#    print(p)


#plt.plot(x, y, label='Kurve')
#plt.xlabel(r'$\alpha \:/\: \si{\ohm}$')
#plt.ylabel(r'$y \:/\: \si{\micro\joule}$')
#plt.legend(loc='best')
#plt.savefig('build/plot.pdf')
#plt.clf()
#plt.plot(x, y, label='Kurve')
#plt.xlabel(r'$\alpha \:/\: \si{\ohm}$')
#plt.ylabel(r'$y \:/\: \si{\micro\joule}$')
#plt.legend(loc='best')
#
## in matplotlibrc leider (noch) nicht möglich
##plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
#plt.savefig('build/plot2.pdf')
