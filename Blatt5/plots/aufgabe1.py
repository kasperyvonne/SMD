import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt


def Signal(x):
    return((1-x)**(-(1.0/1.7)))


def Neutrinofluss(E):
    return(1.7*(E**(-2.7)))


def Akzeptanz(E):
    return((1-np.exp(-0.5*E))**3)


def Polarmethode(E):
    i = 0
    x = -1
    while i == 0:
        u1 = np.random.uniform(0, 1)
        u2 = np.random.uniform(0, 1)
        v1 = 2*u1-1
        v2 = 2*u2-1
        s = v1**2+v2**2
        if (s < 1):
            term = np.sqrt(-(2/s)*np.log(s))
            c1 = v1*term
            c2 = v2*term
            x = 2*E*c1+2*E*c2+10*E
            if (x > 0):
                x = np.round(x)
                i = 1
    return(x)


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
plt.clf()
# Aufgabenteil c)
Hits = np.zeros(len(DetEnergie), dtype=int)
for j in np.arange(0, len(DetEnergie)):
    Hits[j] = Polarmethode(DetEnergie[j])

plt.hist(Hits, bins=50, range=[0, 100], histtype='step')
plt.show()
