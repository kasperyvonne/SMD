import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
import timeit

# Definitionen und so
def Planck(x):
    return(15/(np.pi)**4)*(x**3)/(np.exp(x)-1)


def PlanckAbl(x):
        return((15/(np.pi)**4)
               * ((3*(x**2)/(np.exp(x)-1))
               - (((x**3)*np.exp(x))/(np.exp(x)-1)**2)))


def SchnittMajo(x, ymax):
    return (ymax-200*(15/(np.pi)**4) * (x**(-0.1)) * (np.exp(-x**(0.9))))


def Majorante2(x):
    return(200*(15/(np.pi)**4) * (x**(-0.1)) * (np.exp(-x**(0.9))))


def Majorante1(x):
    return(0.21888647009110665)


# Vorbereitungsbums
x = np.linspace(0.0001, 20, 10000)
xmax = optimize.brentq(PlanckAbl, 2.5, 5)
ymax = Planck(xmax)
xs = optimize.brentq(SchnittMajo, 4.5, 6, args=(ymax))
# plt.plot(x, Majorante2(x))
# plt.plot(x, Planck(x))
# plt.ylim(0, 0.3)
# plt.axhline(y=ymax)
# plt.savefig('Majoranten.png')
# plt.clf()
# Aufgabenteil a)


def Rueckweisung():
    zaehler = 0
    Zufallszahlen = np.empty((100000, 2))
    while zaehler < 100000:
        xwert = np.random.uniform(0, 20)
        ywert = np.random.uniform(0, ymax)
        if(ywert <= Planck(xwert)):
            Zufallszahlen[zaehler] = [xwert, ywert]
            zaehler += 1
    return(Zufallszahlen)


TEST_RUECKWEISUNG = '''
def Rueckweisung():
    zaehler = 0
    Zufallszahlen = np.empty((100000, 2))
    while zaehler < 100000:
        xwert = np.random.uniform(0, 20)
        ywert = np.random.uniform(0, ymax)
        if(ywert <= Planck(xwert)):
            Zufallszahlen[zaehler] = [xwert, ywert]
            zaehler += 1
    return(Zufallszahlen)
'''

Zeit = timeit.timeit(TEST_RUECKWEISUNG, number=1)
Zahlen = Rueckweisung()
plt.hist(Zahlen[:, 0], bins=100)
plt.show()
#print(Zahlen[:,1])
print(Zeit)
