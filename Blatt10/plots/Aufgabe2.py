import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
import scipy.constants as const
from scipy.optimize import curve_fit
from uncertainties import correlated_values, correlation_matrix
from uncertainties import ufloat
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
f = np.array([193, 485, 664, 763, 804, 805, 779, 736, 684, 626,566, 508, 452, 400, 351, 308, 268, 233, 202, 173])
def antwortmatrix(n,epsi):
	A = np.matrix(np.zeros((n,n)))
	np.fill_diagonal(A,1-2*epsi)
	A[(0,0)]+=epsi
	A[(n-1,n-1)]+=epsi
	for i in range(0,n):
		if i-1 >= 0:
			A[i,i-1] = epsi
		if i+1 <= n-1:
			A[i,i+1] = epsi
	return A

def diago(Ant):
	ew,ev = np.linalg.eig(Ant)
	ewindis = np.argsort(ew)
	ewindis = ewindis[::-1]
	ew = [ew[i] for i in ewindis]
	tmatrix = ev.T[ewindis] 
	tmatrix = tmatrix.T
	diago = np.matrix(np.zeros((Ant.shape)))
	np.fill_diagonal(diago,ew)
	return tmatrix, diago	

A = antwortmatrix(20,0.23)
g = np.dot(A,f)
np.random.seed(42)
gmess = [np.random.poisson(i) for i in g]
gmess = np.squeeze(np.array(gmess))
Atrans, Adiag = diago(A)
c = np.dot(Atrans.I,g.T)
b = np.dot(Atrans.I,f.T)
##Kovm von b  mit V[b] = B V[f] B.T mit B = Atrans.I
Vb = np.dot(np.dot(Atrans.I,np.cov(f)), (Atrans.I).T)
###
bmess = np.dot(np.dot(Adiag.I,Atrans.I),gmess)
B = np.dot(Adiag.I,Atrans.I)
Vbmess = np.dot(np.dot(B,np.cov(gmess)), B.T)
stdbmess = np.sqrt(np.diagonal(Vbmess))
nbmess = np.array(bmess/stdbmess)
nbmess = np.squeeze(nbmess)

plt.plot(range(1,21),nbmess, 'bx', label = 'normierte Koeffizenten')
plt.xlabel(r'Index')
plt.ylabel(r'$\frac{b_j}{\sqrt{Var(b_j )}}$')
plt.grid()
plt.legend(loc='best')
#plt.savefig('bjggIndexplot.pdf')
plt.show()
plt.clf()


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
