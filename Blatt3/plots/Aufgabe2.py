import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
import scipy.constants as const
from scipy.optimize import curve_fit
from uncertainties import correlated_values, correlation_matrix
from uncertainties import ufloat
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
import scipy.stats as stat
## Dat Algorithm ##
def Metropolis(xzero,num, step,distr,PDF):
	position = xzero
	countDooku = 0
	randoms = np.array([])	
	while (num != randoms.size):
		xnext = distr(position-step,position + step,1)
		P =  (PDF(xnext,xnext-step,xnext+step) / PDF(position,position-step,position+step))
		rand = np.random.uniform(0,1,1)
		countDooku+=1
		Iteration = np.array([])
		if rand <= P:
			position = xnext
			randoms = np.append(randoms,position)
			Iteration = np.append(Iteration,countDooku)
			
		else:
	#		print("continued")
			continue
	return randoms,Iteration
## Whole lotta Plancking ##
lambda_ = 0.51

def Planckdestr(left,right,num):
	if left <0:
		return stat.planck.rvs(lambda_,size=num)
	else:
		return stat.planck.rvs(lambda_,loc=left,size=num)	

def PlanckPDF(x,left,right):
	if left <0:
		return stat.planck.pmf(x,lambda_)
	else:
		return stat.planck.pmf(x,lambda_,loc = left) 


plancks,planckitis = Metropolis(30,100,2,Planckdestr,PlanckPDF)		
planckhist, planckOnTheEdgeOfSanity = np.histogram(plancks,bins='auto')
plt.hist(plancks)
plt.hist(planckhist, bins= planckOnTheEdgeOfSanity)

plt.show()

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
