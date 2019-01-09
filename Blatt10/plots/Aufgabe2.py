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
#print('A mit n=4 und e=0.23:',antwortmatrix(4,0.23))
A = antwortmatrix(20,0.23)
g = np.dot(A,f)
np.random.seed(42)
gmess = [np.random.poisson(i) for i in g]
gmess = np.squeeze(np.array(gmess))
#print('gmess:', gmess)
Atrans, Adiag = diago(A)
c = np.dot(Atrans.T,g.T)
b = np.dot(Atrans.T,f.T)
#print('bwahr:', b)
#print('c:', c)
##Kovm von b  mit V[b] = B V[f] B.T mit B = Atrans.I
Vb = np.dot(np.dot(Atrans.T,np.cov(f)), (Atrans.I).T)
###
bmess = np.dot(np.dot(Adiag.I,Atrans.T),gmess)
B = np.dot(Adiag.I,Atrans.T)
Vbmess = np.dot(np.dot(B,np.cov(gmess)), B.T)
stdbmess = np.sqrt(np.diagonal(Vbmess))
nbmess = np.array(bmess/stdbmess)
nbmess = np.squeeze(nbmess)

plt.plot(range(1,21),nbmess, 'bx', label = 'normierte Koeffizenten')
plt.xlabel(r'Index')
plt.ylabel(r'$\frac{b_j}{\sqrt{Var(b_j )}}$')
plt.grid()
plt.legend(loc='best')
plt.savefig('bjggIndexplot.pdf')
plt.clf()

fmess = np.dot(Atrans,bmess.T)
breg = np.array(bmess)
np.put(breg,range(8,20),np.zeros(12))
#print('breg:',breg)
freg = np.dot(Atrans,breg.T)
Vfmess = np.dot(np.dot(Atrans,Vbmess) , Atrans.T)
Vfreg = np.dot(np.dot(Atrans,np.cov(breg)) , Atrans.T)
stdfmess = np.sqrt(np.diagonal(Vfmess))
stdfreg = np.sqrt(np.diagonal(Vfreg))

plt.plot(range(1,21),f,'ko', label='wahre Verteilung')
plt.errorbar(range(1,21),fmess,yerr = stdfmess,fmt='bx', label='mess. Verteilung')
plt.errorbar(range(1,21),freg,yerr = stdfreg, fmt='gx', label='reg. Verteilung')
plt.xlabel(r'Index $i$')
plt.ylabel(r'Verteilung $f_j$')
plt.legend(loc='best')
plt.grid()
plt.savefig('regplot.pdf')
