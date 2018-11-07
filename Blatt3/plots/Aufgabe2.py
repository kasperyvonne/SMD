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
def Metropolis(xzero,num, step,PDF):
	position = xzero
	countDooku = 0 # d)
	randoms = np.array([])	
	Iteration = np.array([]) # d)
	while (num != randoms.size):
		xnext = np.random.uniform(position-step,position + step,1)
		countDooku+=1 # d)
		P =  min(1,PDF(xnext,xnext-step,xnext+step) / PDF(position,position-step,position+step))
		rand = np.random.uniform(0,1,1)
		if rand <= P:
			position = xnext
			randoms = np.append(randoms,position)
			Iteration = np.append(Iteration,countDooku) # d)
			
		else:
			continue
	return randoms,Iteration # a), d)
## Whole lotta Plancking ##
#def Planckdestr(left,right,num):
#	if left <0:
#		return stat.planck.rvs(lambda_,size=num)
#	else:
#		return stat.planck.rvs(lambda_,loc=left,size=num)	
#
#def PlanckPDF(x,left,right):
#	if left <0:
#		return stat.planck.pmf(x,lambda_)
#	else:
#		return stat.planck.pmf(x,lambda_,loc = left) 
def PlanckPDFBlatt(x,left,right):
	if left <0:
		return 0
	else:
		return (15/np.pi**4)*(x**3)/(np.exp(x) -1) 


## Planck Plott ##
plancks,planckitis = Metropolis(30,10**5,2,PlanckPDFBlatt)		
plt.hist(plancks,bins= 'auto',density = 'True',histtype='step', label="Metropolis-Daten")
x = np.linspace(min(plancks),max(plancks),1000)
plt.plot(x,PlanckPDFBlatt(x,min(plancks),max(plancks)),label= "Planck-Verteilung")
plt.legend(loc='best')
plt.xlabel(r'Zufallsvariablen')
plt.ylabel(r'Wahrscheinlichkeit')
plt.grid()
plt.savefig('Planckvergleich.pdf')
plt.clf()
## Trace Plot ##
plt.title('Trace Plot')
plt.plot(planckitis,plancks,linewidth = 0.001)
plt.xlabel(r'Iterationen')
plt.ylabel(r'Zufallsvariablen')
plt.grid()
#plt.legend(loc='best')
plt.savefig('Traceplot.pdf')
#plt.show()

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
