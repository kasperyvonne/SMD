import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from functools import reduce

##Einlesen der Daten ##
Pzero = pd.read_hdf('zwei_populationen.h5',key='P_0_10000')
Pone = pd.read_hdf('zwei_populationen.h5',key='P_1')
P01 = pd.read_hdf('zwei_populationen.h5',key='P_0_1000')
p01x = np.array(P01['x'])
p01y = np.array(P01['y'])
p1x = np.array(Pone['x'])
p1y = np.array(Pone['y'])
p0x = np.array(Pzero['x'])
p0y = np.array(Pzero['y'])
##Berechnung der Mittelwertvektoren##
mup0 = np.array([np.mean(p0x),np.mean(p0y)])
mup1 = np.array([np.mean(p1x),np.mean(p1y)])
mup01 = np.array([np.mean(p01x),np.mean(p01y)])
##Berechnung der Kovarianzmatrizen##
Vp0 = np.cov(p0x,p0y)
Vp1 = np.cov(p1x,p1y)
Vp01 = np.cov(np.append(p0x,p1x),np.append(p0y,p1y))
Vp2 = np.cov(p01x,p01y)
Vp001 = np.cov(np.append(p0x,p01x),np.append(p0y,p01y))
print('Kovarianzmatrix p0:',Vp0)
print('Kovarianzmatrix p1:',Vp1)
print('Kovarianzmatrix p0 + p1:',Vp01) 
print('Kovarianzmatrix p01:',Vp2)
print('Kovarianzmatrix p0 + p01:',Vp001)
##Fisher Diskriminanzanalyse##
p0 = np.array([p0x,p0y]) #shape 2,1000
p1 = np.array([p1x,p1y])
p01 = np.array([p01x,p01y])
#Berechnung der Streuung der Klassen
S0 = np.array([p - mup0 for p in p0.T]).T
S1 = np.array([p - mup1 for p in p1.T]).T
S01 = np.array([p - mup01 for p in p1.T]).T
S0 = [np.matrix([s[0],s[1]]).T*(np.matrix([s[0],s[1]])) for s in S0.T]
S0 = reduce(lambda x,y: x+y, S0)
S1 = [np.matrix([s[0],s[1]]).T*(np.matrix([s[0],s[1]])) for s in S1.T]
S1 = reduce(lambda x,y: x+y, S1)
S01 = [np.matrix([s[0],s[1]]).T*(np.matrix([s[0],s[1]])) for s in S01.T]
S01 = reduce(lambda x,y: x+y, S01)
SW = S0 + S1 #Berechnung der Gesamtstreuung = Within class scatter matrix
SW001 =  S0 + S01

lamb = SW.I*np.matrix([[(mup0[0]-mup1[0])],[(mup0[1]-mup1[1])]]) #Berechnung der Projektion
lamb = lamb.T
lambnorm = np.sqrt(np.dot(lamb,lamb.T)) #lambda normierung
lamb/= lambnorm #Müsste jetzt Einehitsvektor sein
print('Normiertes Lambda p0:',lamb)
print(lambnorm)
lamb001 = SW.I*np.matrix([[(mup0[0]-mup01[0])],[(mup0[1]-mup01[1])]]) #Berechnung der Projektion
lamb001 = lamb001.T
lambnorm001 = np.sqrt(np.dot(lamb001,lamb001.T)) #lambda normierung
lamb001/= lambnorm001 #Müsste jetzt Einehitsvektor sein
print('Normiertes Lambda p01:',lamb001)
print(lambnorm001)

##1dim Hist der Populationen##
def projektion(lam,data):
	return np.squeeze(np.asarray(np.dot(lam,data)/np.dot(lam,lam.T)))
proj0 = projektion(lamb,p0)
proj1 = projektion(lamb,p1)
proj01 = projektion(lamb,p01)

plt.hist(proj0,bins='auto', label='Population 0',histtype = 'step')
plt.hist(proj1,bins='auto', label='Population 1',histtype = 'step')
plt.xlabel(r'$\lambda_x \cdot x$')
plt.legend(loc='best')
plt.savefig('Projektion1dimhist.pdf')
plt.clf()
plt.hist(proj0,bins='auto', label='Population 0',histtype = 'step')
plt.hist(proj01,bins='auto', label='Population 0_1000',histtype = 'step')
plt.xlabel(r'$\lambda_x \cdot x$')
plt.legend(loc='best')
plt.savefig('2Projektion1dimhist.pdf')

## Lambda Cut Funktion ##
def lambdaCut(lamcut,lam,P0,P1):
	reinheit  = np.array([])
	effizienz = np.array([])
	#hier steck ich jetzt rein das ich weiß, dass Population 0 rechts liegt	
	# Population 0 ist signal also "positiv"
	for l in lamcut:
		projzero = projektion(lam,P0)
		projone = projektion(lam,P1)
		fn = projzero[projzero < l].size #false negativ
		tp = projzero[projzero > l].size # true positiv
		tn = projone[projone < l].size	#true negativ
		fp = projone[projone > l].size #false positiv
		if (tp + fp) <= 0 : break
		if (tp + fn) <= 0 : break
		reinheit = np.append(reinheit,[tp/(tp+fp)])
		effizienz = np.append(effizienz,[tp/(tp+fn)])
	return reinheit,effizienz
## reinheit und effizienz plot ##
lambcut = np.linspace(min(min(proj0),min(proj1)),max(max(proj0),max(proj1)),10**3)
lambcut2 = np.linspace(min(min(proj0),min(proj01)),max(max(proj0),max(proj01)),10**3)
rein,effi = lambdaCut(lambcut,lamb,p0,p1)	
rein2,effi2 = lambdaCut(lambcut2,lamb001,p0,p01)	

plt.clf()
plt.plot(lambcut[:rein.size],rein, '-r', label='Reinheit')
plt.plot(lambcut[:effi.size],effi, '-b', label='Effizienz')
plt.xlabel(r'$\lambda_{Cut}$')
plt.ylabel(r'Reinheit & Effizizenz')
plt.legend(loc='best')
plt.savefig('ReinheitEffizienzplot.pdf')
plt.clf()
plt.plot(lambcut2[:rein2.size],rein2, '-r', label='Reinheit')
plt.plot(lambcut2[:effi2.size],effi2, '-b', label='Effizienz')
plt.xlabel(r'$\lambda_{Cut}$')
plt.ylabel(r'Reinheit & Effizizenz')
plt.legend(loc='best')
plt.savefig('2ReinheitEffizienzplot.pdf')
#plt.show()
## S/B Quotient 
def SBratio(lamcut,lam,P0,P1):
	SB = np.array([])
	for l in lamcut:
		projzero = projektion(lam,P0)
		projone = projektion(lam,P1)
		tp = projzero[projzero > l].size # true positiv
		if projone[projone > l].size <= 0: break
		fp = projone[projone > l].size #false positiv
		SB = np.append(SB,tp/fp)
	return SB
SBRatio = SBratio(lambcut, lamb, p0,p1)
SBRatio2 = SBratio(lambcut2, lamb001, p0,p01)
placeholder = lambcut[:SBRatio.size]
print('max sbr 1:',placeholder[SBRatio==max(SBRatio)])
placeholder = lambcut2[:SBRatio2.size]
print('max sbr2:',placeholder[SBRatio2==max(SBRatio2)])

plt.clf()
plt.plot(lambcut[:SBRatio.size],SBRatio, label='S/B Quotient')
plt.xlabel(r'$\lambda_{Cut}$')
plt.ylabel(r'$S/B \; = \; \frac{true  \; positiv}{false \; positiv}$ ')
plt.legend(loc='best')
plt.savefig('SBRatioplot.pdf')
plt.clf()
plt.plot(lambcut2[:SBRatio2.size],SBRatio2, label='S/B Quotient')
plt.xlabel(r'$\lambda_{Cut}$')
plt.ylabel(r'$S/B \; = \; \frac{true  \; positiv}{false \; positiv}$ ')
plt.legend(loc='best')
plt.savefig('2SBRatioplot.pdf')

##Signifikanz
def Signiratio(lamcut, lam, P0, P1):
	Sign = np.array([])
	for l in lamcut:
		projzero = projektion(lam,P0)
		projone = projektion(lam,P1)
		tp = projzero[projzero > l].size # true positiv
		fp = projone[projone > l].size #false positiv
		if np.sqrt((tp+fp))<= 0: break
		Sign = np.append(Sign,tp/np.sqrt((tp+fp)))
	return Sign
Sign = Signiratio(lambcut,lamb,p0,p1)
Sign2 = Signiratio(lambcut2,lamb001,p0,p01)
placeholder = lambcut[:Sign.size]
print('max sign1:',placeholder[Sign==max(Sign)])
placeholder = lambcut2[:Sign2.size]
print('max sign2:',placeholder[Sign2==max(Sign2)])

plt.clf()
plt.plot(lambcut[:Sign.size],Sign, label=r'$S/\sqrt{S+B}$ Quotient')
plt.xlabel(r'$\lambda_{Cut}$')
plt.ylabel(r'$S/B \; = \; \frac{true  \; positiv}{false \; positiv + true  \; positiv}$ ')
plt.legend(loc='best')
plt.savefig('Signifikanzplot.pdf')
plt.clf()
plt.plot(lambcut2[:Sign2.size],Sign2, label=r'$S/\sqrt{S+B}$ Quotient')
plt.xlabel(r'$\lambda_{Cut}$')
plt.ylabel(r'$S/B \; = \; \frac{true  \; positiv}{false \; positiv + true  \; positiv}$ ')
plt.legend(loc='best')
plt.savefig('2Signifikanzplot.pdf')

