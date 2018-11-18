import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt


def Signal(x):
    return((1-x)**(-(1.0/1.7)))


def Neutrinofluss(E):
    return(1.7*(E**(-2.7)))


def Akzeptanz(E):
    return((1-np.exp(-0.5*E))**3)


# Aufgabenteil a)
Esize = 100000
x = np.random.uniform(0, 1, Esize)
Energie = np.empty(Esize)
Energie = Signal(x)

# Aufgabenteil b)
x2 = np.random.uniform(0, 1, Esize)
Detektiert = np.zeros(Esize, dtype=bool)

for i in np.arange(0, Esize):
    if x2[i] < Akzeptanz(Energie[i]):
        Detektiert[i] = True

DetEnergie = Energie[Detektiert]
plt.hist(Energie, bins=np.logspace(0, 3), histtype='step',
         label='Energie')
plt.hist(DetEnergie, bins=np.logspace(0, 3), histtype='step', linestyle='--',
         label='Detektierte Energie')
plt.loglog()
plt.legend()
plt.xlabel(r'$E/\mathrm{TeV}$')
plt.ylabel(r'Ereignisse')
plt.savefig('Energie.pdf')

# Aufgabenteil c)
